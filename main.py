import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Note: meshio is no longer needed since we are running in-memory!

from src.mesh import Mesh
from src.element import Q4Element, T3Element
from src.assembly import assemble_global_system
from src.boundary_conditions import apply_dirichlet_bcs
from src.solver import solve_system
from src.postprocess import compute_flux_fields, export_to_vtk, plot_results_3d


def generate_sample_mesh(num_r=40, num_theta=180):
    """
    Generates a high-density unstructured T3 mesh directly in memory.
    Returns: coords (nodes), elements (connectivity array), and element_types.
    """
    print(f"Generating sample unstructured mesh in-memory ({num_r}x{num_theta})...")
    
    # Domain dimensions and hole geometry
    r_inner = 0.05
    
    # 1. Radial rings and angular divisions around the hole
    radii = np.linspace(r_inner, 0.7, num_r)
    angles = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    
    points = []
    for r in radii:
        for theta in angles:
            points.append([r * np.cos(theta), r * np.sin(theta)])
            
    # 2. Boundary resolution along outer rectangular frame
    x_outer = np.linspace(-1.0, 1.0, 120)
    y_outer = np.linspace(-0.5, 0.5, 60)
    
    for x in x_outer:
        points.append([x, -0.5])
        points.append([x, 0.5])
    for y in y_outer[1:-1]:
        points.append([-1.0, y])
        points.append([1.0, y])
        
    points = np.array(points)
    
    # Delaunay triangulation over point cloud
    triangulation = tri.Triangulation(points[:, 0], points[:, 1])
    
    # Filter out elements lying inside the central hole
    centroids = np.mean(points[triangulation.triangles], axis=1)
    valid_mask = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2) >= (r_inner - 1e-4)
    triangles = triangulation.triangles[valid_mask]
    
    element_types = ["T3"] * len(triangles)
    
    return points, triangles, element_types


def generate_structured_tri_mesh(nx, ny, lx=1.0, ly=1.0):
    """Generates a structured triangular grid for benchmark verification."""
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    xv, yv = np.meshgrid(x, y)
    coords = np.column_stack([xv.ravel(), yv.ravel()])

    elements = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            n0 = i * nx + j
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])

    element_types = ["T3"] * len(elements)
    
    class BenchmarkMesh:
        def __init__(self, coords, elements, element_types):
            self.coords = coords
            self.elements = np.array(elements, dtype=int)
            self.element_types = element_types
            self.num_nodes = len(coords)
            self.num_elements = len(elements)

    return BenchmarkMesh(coords, elements, element_types)


def analytical_solution(x, y, lx=1.0, ly=1.0, t_max=100.0):
    """Analytical Laplace benchmark: T(x,y) = T_max * sin(pi*x/L) * sinh(pi*y/L) / sinh(pi)."""
    return t_max * np.sin(np.pi * x / lx) * (np.sinh(np.pi * y / ly) / np.sinh(np.pi))


def run_benchmark_suite():
    """Runs mesh convergence and solver timing profiling across multiple grid sizes."""
    print("\n--- Benchmark Suite: Error Norm & Performance Scaling ---")
    grid_sizes = [(10, 10), (20, 20), (40, 40), (70, 70), (100, 100)]
    
    node_counts = []
    assembly_times = []
    solve_times = []
    l2_errors = []

    kx, ky = 1.0, 1.0
    lx, ly = 1.0, 1.0
    t_max = 100.0

    print(f"{'Nodes':>8} | {'Elements':>9} | {'Assembly (s)':>12} | {'Solve (s)':>10} | {'L2 Error (%)':>12}")
    print("-" * 62)

    for nx, ny in grid_sizes:
        b_mesh = generate_structured_tri_mesh(nx, ny, lx, ly)

        # 1. Compute Element Matrices & Measure COO Assembly Time
        t0_asm = time.perf_counter()
        k_elem_dict = {}
        for elem_id in range(b_mesh.num_elements):
            elem_coords = b_mesh.coords[b_mesh.elements[elem_id]]
            k_elem_dict[elem_id] = T3Element.compute_ke(elem_coords, kx, ky)

        K_global, F_global = assemble_global_system(b_mesh, k_elem_dict)
        t1_asm = time.perf_counter()
        asm_time = t1_asm - t0_asm

        # 2. Apply Boundary Conditions
        dirichlet_bcs = {}
        for idx, (x_i, y_i) in enumerate(b_mesh.coords):
            if np.isclose(x_i, 0.0) or np.isclose(x_i, lx) or np.isclose(y_i, 0.0):
                dirichlet_bcs[idx] = 0.0
            elif np.isclose(y_i, ly):
                dirichlet_bcs[idx] = t_max * np.sin(np.pi * x_i / lx)

        K_bc, F_bc = apply_dirichlet_bcs(K_global, F_global, dirichlet_bcs)

        # 3. Measure Sparse Direct Solve Time
        t0_solve = time.perf_counter()
        T_num = solve_system(K_bc, F_bc)
        t1_solve = time.perf_counter()
        solve_time = t1_solve - t0_solve

        # 4. Compute Relative L2 Error Norm
        T_exact = analytical_solution(b_mesh.coords[:, 0], b_mesh.coords[:, 1], lx, ly, t_max)
        err_num = np.sqrt(np.sum((T_num - T_exact) ** 2))
        err_den = np.sqrt(np.sum(T_exact ** 2))
        l2_err = (err_num / err_den) * 100.0

        node_counts.append(b_mesh.num_nodes)
        assembly_times.append(asm_time)
        solve_times.append(solve_time)
        l2_errors.append(l2_err)

        print(f"{b_mesh.num_nodes:8d} | {b_mesh.num_elements:9d} | {asm_time:12.4f} | {solve_time:10.4f} | {l2_err:11.3f}%")

    # Plot Convergence and Performance Graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Error Plot
    ax1.plot(node_counts, l2_errors, "ro-", linewidth=2, label="FEA Solver Error")
    ax1.axhline(y=1.5, color="k", linestyle="--", alpha=0.7, label="1.5% Target Line")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Nodes (Log Scale)")
    ax1.set_ylabel("Relative $L_2$ Error (%)")
    ax1.set_title("Mesh Convergence vs. Analytical Benchmark", fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    # Matrix Assembly & Solver Time Plot
    ax2.plot(node_counts, assembly_times, "bs-", linewidth=2, label="Assembly Time (COO)")
    ax2.plot(node_counts, solve_times, "g^-", linewidth=2, label="Solve Time (CSR)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Nodes (Log Scale)")
    ax2.set_ylabel("Execution Time (s, Log Scale)")
    ax2.set_title("Assembly & Solver Scaling Profile", fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 0. Run Formal Benchmark Verification Suite
    run_benchmark_suite()

    # 1. Generate In-Memory Geometry
    print("\n--- Phase 3: Unstructured Geometry & Dual-Element Execution ---")
    coords, elements, element_types = generate_sample_mesh(num_r=40, num_theta=180)
    
    # Initialize the Mesh object directly without saving/reading a file
    # Note: If your src.mesh.Mesh class expects arguments in __init__, adjust this accordingly.
    # We are mimicking the attribute structure your solver expects.
    mesh = Mesh() 
    mesh.coords = coords
    mesh.elements = elements
    mesh.element_types = element_types
    mesh.num_nodes = len(coords)
    mesh.num_elements = len(elements)

    print(f"Loaded Mesh: {mesh.num_nodes} nodes, {mesh.num_elements} elements.")
    print(f"Element Types Present: {set(mesh.element_types)}")

    # Isotropic Thermal Conductivity (W/m·K)
    kx, ky = 25.0, 25.0

    # 2. Compute Element Stiffness Matrices (Dual Support for T3 & Q4)
    k_elem_dict = {}
    for elem_id in range(mesh.num_elements):
        elem_coords = mesh.coords[mesh.elements[elem_id]]
        e_type = mesh.element_types[elem_id]

        if e_type == "T3":
            k_elem_dict[elem_id] = T3Element.compute_ke(elem_coords, kx, ky)
        elif e_type == "Q4":
            k_elem_dict[elem_id] = Q4Element.compute_ke(elem_coords, kx, ky)
        else:
            raise ValueError(f"Unsupported element type encountered: {e_type}")

    # 3. Assemble Global Linear System
    K_global, F_global = assemble_global_system(mesh, k_elem_dict)

    # 4. Apply Dirichlet Boundary Conditions by coordinate location
    left_nodes = np.where(np.isclose(mesh.coords[:, 0], -1.0))[0]
    right_nodes = np.where(np.isclose(mesh.coords[:, 0], 1.0))[0]
    
    dirichlet_bcs = {}
    for node in left_nodes:
        dirichlet_bcs[node] = 20.0
    for node in right_nodes:
        dirichlet_bcs[node] = -20.0

    print(f"Left boundary nodes found: {len(left_nodes)}")
    print(f"Right boundary nodes found: {len(right_nodes)}")

    K_bc, F_bc = apply_dirichlet_bcs(K_global, F_global, dirichlet_bcs)

    # 5. Solve Primary Field (Nodal Temperature Vector T)
    print("Solving global FEA system...")
    T = solve_system(K_bc, F_bc)
    print(f"Computed Nodal Temperature Range: Min = {T.min():.2f} °C, Max = {T.max():.2f} °C")

    # 6. Post-Processing: Compute Heat Flux Vector Field q = -k ∇T
    print("\n--- Phase 4: Thermal Post-Processing & Scientific Export ---")
    centroids, grad_T, q = compute_flux_fields(mesh, T, kx, ky)
    
    q_magnitude = np.linalg.norm(q, axis=1)
    print(f"Computed Heat Flux Magnitude Range: Min = {q_magnitude.min():.2f} W/m², Max = {q_magnitude.max():.2f} W/m²")

    # 7. VTK File Export for ParaView Visualization
    vtk_filename = "solution.vtk"
    export_to_vtk(vtk_filename, mesh, T, q)

    # 8. Render Plots (3D Surface Only)
    plot_results_3d(mesh, T)


if __name__ == "__main__":
    main()