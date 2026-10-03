import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Import your custom modules
from src.mesh import Mesh
from src.element import Q4Element, T3Element
from src.assembly import assemble_global_system
from src.boundary_conditions import apply_dirichlet_bcs
from src.solver import solve_system
from src.postprocess import compute_flux_fields, export_to_vtk

# Import the updated in-memory mesh generator from your main.py
from main import generate_sample_mesh 

# --- Page Configuration ---
st.set_page_config(page_title="FEM Heat Transfer Solver", page_icon="🔥", layout="wide")

# --- UI: Header & Resume Highlights ---
st.title("🔥 2D/3D FEM Heat Transfer Solver")
st.markdown("""
**Project Highlights:**
* Engineered a 1D/2D FEM heat-transfer solver supporting 10,000+ nodes and 9,800+ elements.
* Achieved < 1.5% error versus analytical solutions across validation cases.
* Improved computational efficiency by 12× using sparse matrix storage (CSR).
* Implemented Dirichlet boundary conditions based on spatial coordinates.
""")
st.divider()

# --- UI: Sidebar Parameters ---
st.sidebar.header("⚙️ Simulation Parameters")

st.sidebar.subheader("Mesh Density")
# Tweak the min/max values based on what your solver can handle comfortably in a few seconds
num_r = st.sidebar.slider("Radial Divisions", min_value=10, max_value=60, value=40)
num_theta = st.sidebar.slider("Angular Divisions", min_value=45, max_value=240, value=180)

st.sidebar.subheader("Material Properties")
kx = st.sidebar.number_input("Thermal Conductivity kx (W/m·K)", value=25.0, step=1.0)
ky = st.sidebar.number_input("Thermal Conductivity ky (W/m·K)", value=25.0, step=1.0)

st.sidebar.subheader("Boundary Conditions")
left_bc = st.sidebar.number_input("Left Boundary Temp (°C)", value=20.0, step=5.0)
right_bc = st.sidebar.number_input("Right Boundary Temp (°C)", value=-20.0, step=5.0)

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

# --- Execution Logic ---
if run_button:
    with st.spinner("Generating mesh, assembling matrices, and solving system..."):
        
        start_time = time.perf_counter()
        
        # 1. Generate In-Memory Geometry
        coords, elements, element_types = generate_sample_mesh(num_r=num_r, num_theta=num_theta)
        
        # 2. Create Mesh object directly in memory
        mesh = Mesh.generate_structured(nx=10, ny=10, lx=1.0, ly=1.0)
        mesh.coords = coords
        mesh.elements = elements
        mesh.element_types = element_types
        
        # 3. Compute Element Matrices
        k_elem_dict = {}
        for elem_id in range(mesh.num_elements):
            elem_coords = mesh.coords[mesh.elements[elem_id]]
            e_type = mesh.element_types[elem_id]
            if e_type == "T3":
                k_elem_dict[elem_id] = T3Element.compute_ke(elem_coords, kx, ky)
            elif e_type == "Q4":
                k_elem_dict[elem_id] = Q4Element.compute_ke(elem_coords, kx, ky)
                
        # 4. Assemble Global System
        K_global, F_global = assemble_global_system(mesh, k_elem_dict)
        
        # 5. Apply Boundary Conditions
        left_nodes = np.where(np.isclose(mesh.coords[:, 0], -1.0))[0]
        right_nodes = np.where(np.isclose(mesh.coords[:, 0], 1.0))[0]
        
        dirichlet_bcs = {node: left_bc for node in left_nodes}
        dirichlet_bcs.update({node: right_bc for node in right_nodes})
        
        K_bc, F_bc = apply_dirichlet_bcs(K_global, F_global, dirichlet_bcs)
        
        # 6. Solve
        T = solve_system(K_bc, F_bc)
        
        # 7. Compute Flux
        centroids, grad_T, q = compute_flux_fields(mesh, T, kx, ky)
        q_magnitude = np.linalg.norm(q, axis=1)
        
        # 8. Generate VTK File for Download
        vtk_filename = "solution.vtk"
        export_to_vtk(vtk_filename, mesh, T, q)
        
        end_time = time.perf_counter()

    # --- UI: Display Results ---
    st.success(f"Simulation completed successfully in {end_time - start_time:.3f} seconds!")
    
    # Display top-level metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", f"{mesh.num_nodes:,}")
    col2.metric("Total Elements", f"{mesh.num_elements:,}")
    col3.metric("Max Temperature", f"{T.max():.2f} °C")
    col4.metric("Max Heat Flux", f"{q_magnitude.max():.2f} W/m²")
    
    st.divider()
    st.subheader("Interactive Temperature Distribution")
    st.caption("Drag to rotate, scroll to zoom. Hover over the surface to inspect specific nodal temperatures.")
    
    # --- Interactive 3D Plotting using Plotly ---
    x = mesh.coords[:, 0]
    y = mesh.coords[:, 1]
    z = T.flatten()
    
    # Create the 3D surface using the exact finite element triangulation
    fig = go.Figure(data=[go.Mesh3d(
        x=x, y=y, z=z, 
        intensity=z, 
        colorscale='Inferno',
        colorbar_title="Temp (°C)",
        # Extract the node indices for the triangles (i, j, k)
        i=mesh.elements[:, 0], 
        j=mesh.elements[:, 1], 
        k=mesh.elements[:, 2],
        showscale=True
    )])
    
    # Configure the 3D scene aesthetics
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)', 
            yaxis_title='Y (m)', 
            zaxis_title='Temperature (°C)',
            aspectmode='data' # Ensures the physical dimensions aren't distorted
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )
    
    # Render plot
    st.plotly_chart(fig, use_container_width=True)

    # --- UI: VTK File Download ---
    st.subheader("Raw Data Export")
    st.write("Download the resulting temperature and heat flux fields to analyze locally in ParaView.")
    
    if os.path.exists(vtk_filename):
        with open(vtk_filename, "rb") as file:
            st.download_button(
                label="📥 Download VTK File",
                data=file,
                file_name="fea_results.vtk",
                mime="application/octet-stream"
            )
else:
    st.info("👈 Adjust the parameters in the sidebar and click **Run Simulation** to begin.")