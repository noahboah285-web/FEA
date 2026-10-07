# 2D Unstructured Finite Element Thermal Solver (Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FEA Accuracy](https://img.shields.io/badge/L2%20Error-%3C1.5%25-brightgreen.svg)]()

An end-to-end 2D Steady-State Thermal Finite Element Analysis (FEA) solver engineered in Python. The project features automated mesh generation, dual element formulation (T3/Q4), sparse global system assembly, interactive 3D thermal topography visualization, and a quantitative validation suite benchmarked against analytical Laplace solutions.

Live Dashboard: https://awsbjt3xvdu2c37jyzpubn.streamlit.app/

---

## Project Overview

This application solves 2D steady-state heat conduction across complex geometries with stress risers and cutouts (such as plates with circular holes). Supporting meshes up to 10,000+ nodes and 19,000+ elements, the solver manages the complete scientific computing pipeline:

1. **Mesh Import & Generation:** Reads unstructured geometries via `meshio` and generates high-density Delaunay grids.
2. **Element Formulation:** Computes local stiffness matrices ($K^e$) for Constant Strain Triangles (T3) and Isoparametric Quadrilaterals (Q4).
3. **Sparse Assembly:** Assembles global linear systems using Sparse Coordinate Format (COO) for memory-efficient $\mathcal{O}(N)$ scaling.
4. **Linear System Solution:** Enforces Dirichlet boundary conditions and solves $K \cdot T = F$ via Compressed Sparse Row (CSR) direct solvers (`scipy.sparse.linalg.spsolve`).
5. **Post-Processing & Visualization:** Calculates temperature gradient vectors ($\nabla T$) and heat flux vectors ($\vec{q} = -k \nabla T$), exporting data to `.vtk` files for ParaView and rendering interactive 3D Matplotlib surface elevation plots.

---

## Key Features & Technical Highlights

* **Dual-Element Compatibility (T3 & Q4):** Supports both 3-node triangular (T3) and 4-node quadrilateral (Q4) isoparametric element formulations.
* **Scalable Sparse Matrix Architecture:** Converts COO sparse matrices to CSR format prior to linear system solving, maintaining fast execution times even at high node counts.
* **Automated Benchmarking Suite:** Built-in validation suite that tests solver convergence across 5 grid resolutions ($N \approx 100$ to $10,000+$ nodes) against an exact analytical Laplace solution.
* **Multi-Format Export:** Outputs 3D interactive surface elevation plots (`matplotlib`), heat flux vector field quivers, and `.vtk` structured geometry files for ParaView visualization.

---

## Performance & Convergence Metrics

The solver was verified against the 2D analytical Laplace heat equation:

$$T_{\text{exact}}(x,y) = T_{\max} \sin\left(\frac{\pi x}{L}\right) \frac{\sinh(\pi y / L)}{\sinh(\pi)}$$

| Node Count ($N$) | Element Count | Matrix Assembly (s) | Linear System Solve (s) | Relative $L_2$ Error (%) |
| :--- | :--- | :--- | :--- | :--- |
| **100** | 162 | ~0.0012s | ~0.0003s | 4.12% |
| **400** | 722 | ~0.0045s | ~0.0011s | 1.85% |
| **1,600** | 3,042 | ~0.0180s | ~0.0048s | 0.82% |
| **4,900** | 9,522 | ~0.0570s | ~0.0165s | 0.31% |
| **10,000** | 19,602 | ~0.1210s | ~0.0380s | **0.15%** |

> **Validation Note:** The solver achieves monotonic $h$-refinement convergence, bringing relative $L_2$ error down to **0.15%** at $N \approx 10,000$ nodes while keeping combined assembly and solve execution times under 0.2 seconds.

---

## Mathematical Formulation

### 1. Primary Field Equation
Steady-state heat conduction governed by the 2D Laplace equation:

$$-\nabla \cdot (k \nabla T) = Q$$

### 2. Element Stiffness Matrix Assembly
For isotropic thermal conductivity ($k_x$, $k_y$), the local element stiffness matrix $K^e$ is integrated as:

$$K^e = \int_{\Omega^e} B^T D B \, d\Omega$$

where $B$ is the strain-displacement matrix containing shape function derivatives ($\frac{\partial N_i}{\partial x}$ and $\frac{\partial N_i}{\partial y}$), and $D$ is the conductivity tensor:

$$D = \begin{bmatrix} k_x & 0 \\ 0 & k_y \end{bmatrix}$$

### 3. Boundary Conditions & Heat Flux Post-Processing
* **Dirichlet BCs:** Prescribed nodal temperatures applied directly via identity row modification in the sparse global system.
* **Heat Flux Vector Field:** Computed via Fourier's Law at element centroids:

$$\vec{q} = -k \nabla T = -\begin{bmatrix} k_x \frac{\partial T}{\partial x} \\ k_y \frac{\partial T}{\partial y} \end{bmatrix}$$

---

## Assumptions & Solver Limitations

### 1. Physical & Material Assumptions
* **Steady-State Thermal Behavior:** Assumes steady-state heat conduction ($\frac{\partial T}{\partial t} = 0$). Thermal mass, heat capacity ($c_p$), density ($\rho$), and transient temperature response are omitted.
* **Isotropic/Orthotropic Material Properties:** Thermal conductivity coefficients ($k_x$, $k_y$) are assumed constant per element and temperature-independent.
* **No Internal Heat Generation ($Q = 0$):** Primary unstructured domain solves pure conduction without volumetric internal heat sources or sinks ($Q(x,y) = 0$).
* **2D Planar Geometry:** Formulated for thin 2D plates operating under plane heat conduction with uniform unit thickness ($t = 1.0$).

### 2. Boundary Condition Support
* **Dirichlet BC Dominance:** Supports fixed nodal temperature boundary conditions ($T = T_0$).
* **Insulated Boundary Default:** Unassigned outer boundaries default to zero heat flux ($\nabla T \cdot \mathbf{n} = 0$).
* **No Convection or Radiation:** Natural/forced surface convection and radiative exchange are not currently modeled.

### 3. Numerical & Mathematical Formulations
* **Linear Shape Functions (T3/Q4):** Constant Strain Triangles (T3) yield piecewise constant temperature gradients ($\nabla T$) and heat flux vectors ($\vec{q}$) within each element, causing step discontinuities across element edges prior to nodal averaging.
* **Flux Accuracy:** Nodal temperature values ($T$) converge at $\mathcal{O}(h^2)$, while post-processed derivative fields ($\vec{q} = -k \nabla T$) converge at $\mathcal{O}(h)$ due to numerical differentiation.

### 4. Computational & Scaling Limits
* **In-Core Sparse Solver:** Direct matrix factorization (`scipy.sparse.linalg.spsolve`) uses $\mathcal{O}(N^{1.5})$ memory in 2D, which is highly efficient for up to $\sim 10^5$ degrees of freedom but requires iterative solvers (e.g., Preconditioned Conjugate Gradient) for $10^6+$ node grids.
* **Single-Threaded Execution:** Matrix assembly and linear system operations execute sequentially without multi-threading (OpenMP/MPI) or GPU acceleration.

---

## Repository Architecture

```text
fea-thermal-solver/
├── src/
│   ├── mesh.py                # Mesh class & meshio wrapper for file loading
│   ├── element.py             # T3 & Q4 element stiffness matrix calculations (Ke)
│   ├── assembly.py            # Global system assembly using scipy.sparse COO
│   ├── boundary_conditions.py # Direct Dirichlet boundary condition application
│   ├── solver.py              # Sparse linear system direct solver (SuperLU)
│   └── postprocess.py         # Heat flux field computation, VTK exporter, 3D plotter
├── main.py                    # Unified entry point (Benchmark execution + 3D FEA solve)
├── plate_with_hole.msh        # Auto-generated Gmsh format mesh file (~10k nodes)
├── solution.vtk               # VTK output file for ParaView inspection
└── README.md                  # Detailed project documentation
