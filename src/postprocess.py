# src/postprocess.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
from src.element import Q4Element, T3Element

def compute_flux_fields(mesh, T, kx, ky):
    """Computes spatial gradients ∇T and heat flux q at element centroids."""
    centroids = np.zeros((mesh.num_elements, 2))
    grad_T = np.zeros((mesh.num_elements, 2))
    q = np.zeros((mesh.num_elements, 2))

    for i in range(mesh.num_elements):
        elem_nodes = mesh.elements[i]
        coords = mesh.coords[elem_nodes]
        T_elem = T[elem_nodes]
        elem_type = mesh.element_types[i]

        if elem_type == "T3":
            c, g, q_vec = T3Element.compute_centroid_flux(coords, T_elem, kx, ky)
        elif elem_type == "Q4":
            c, g, q_vec = Q4Element.compute_centroid_flux(coords, T_elem, kx, ky)
            
        centroids[i] = c
        grad_T[i] = g
        q[i] = q_vec

    return centroids, grad_T, q


def export_to_vtk(filepath, mesh, T, q_element):
    """Exports nodal scalar T and cell vector q to legacy VTK format."""
    # Split elements by type for meshio output
    cells = []
    t3_indices = [i for i, t in enumerate(mesh.element_types) if t == "T3"]
    q4_indices = [i for i, t in enumerate(mesh.element_types) if t == "Q4"]

    if t3_indices:
        cells.append(("triangle", np.array([mesh.elements[i] for i in t3_indices])))
    if q4_indices:
        cells.append(("quad", np.array([mesh.elements[i] for i in q4_indices])))

    # Pad 2D coordinates to 3D for VTK specification
    coords_3d = np.hstack([mesh.coords, np.zeros((len(mesh.coords), 1))])
    
    # Pad 2D flux vectors to 3D for Cell Data
    q_3d = np.hstack([q_element, np.zeros((len(q_element), 1))])

    vtk_mesh = meshio.Mesh(
        points=coords_3d,
        cells=cells,
        point_data={"Temperature": T},
        cell_data={"HeatFlux": [q_3d]}
    )
    vtk_mesh.write(filepath)
    print(f"VTK file successfully written to: {filepath}")


def plot_results(mesh, T, centroids, q):
    plt.figure(figsize=(10, 6))

    # 1. Isolate T3 elements into a clean 2D NumPy integer array
    t3_elements = [
        elem for elem, e_type in zip(mesh.elements, mesh.element_types) if e_type == "T3"
    ]
    triangles = np.array(t3_elements, dtype=int)

    # 2. Build Triangulation passing explicit mesh triangles (fixes boundary distortion)
    triangulation = tri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], triangles=triangles)

    # 3. Plot temperature contour gradient
    contour = plt.tricontourf(triangulation, T, levels=25, cmap="inferno")
    plt.colorbar(contour, label="Temperature (°C)")

    # 4. Correctly scaled Heat Flux Quivers
    # Normalizing arrows for uniform direction display with magnitude scaling
    q_mag = np.linalg.norm(q, axis=1)
    q_mag[q_mag == 0] = 1e-10  # prevent division by zero
    
    # Scale vector lengths to ~0.05 units in plot coordinates
    qx_norm = (q[:, 0] / q_mag) * 0.05
    qy_norm = (q[:, 1] / q_mag) * 0.05

    plt.quiver(
        centroids[:, 0], centroids[:, 1],
        qx_norm, qy_norm,
        color="cyan", scale=1, scale_units="xy", angles="xy",
        width=0.003, alpha=0.9, label="Heat Flux Vector q"
    )

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("2D Thermal Field & Heat Flux Vector Field")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_results_3d(mesh, T):
    """
    Renders a 3D surface plot where Z-coordinate represents Temperature (°C).
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Extract T3 elements
    t3_elements = [
        elem for elem, e_type in zip(mesh.elements, mesh.element_types) if e_type == "T3"
    ]
    triangles = np.array(t3_elements, dtype=int)

    # 2. Build Triangulation
    triangulation = tri.Triangulation(mesh.coords[:, 0], mesh.coords[:, 1], triangles=triangles)

    # 3. Render 3D Surface
    surf = ax.plot_trisurf(
        triangulation, T, 
        cmap='inferno', edgecolor='none', antialiased=True, alpha=0.9
    )

    # Formatting
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Temperature [°C]')
    ax.set_title('3D Temperature Elevation Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Temperature (°C)')

    # Set viewing angle
    ax.view_init(elev=35, azim=-45)
    plt.tight_layout()
    plt.show()