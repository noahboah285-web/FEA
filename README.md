# 2D Unstructured Finite Element Thermal Solver (Python)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FEA Accuracy](https://img.shields.io/badge/L2%20Error-%3C1.5%25-brightgreen.svg)]()

An end-to-end 2D Steady-State Thermal Finite Element Analysis (FEA) solver engineered in Python. The project features automated mesh generation, dual element formulation (T3/Q4), sparse global system assembly, interactive 3D thermal topography visualization, and a quantitative validation suite benchmarked against analytical Laplace solutions.

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
| **1,600** |
