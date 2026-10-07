import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Import custom solver modules
from src.mesh import Mesh
from src.element import Q4Element, T3Element
from src.assembly import assemble_global_system
from src.boundary_conditions import apply_dirichlet_bcs
from src.solver import solve_system
from src.postprocess import compute_flux_fields, export_to_vtk

# Import in-memory mesh generator
from main import generate_sample_mesh 

# --- Page Configuration ---
st.set_page_config(
    page_title="2D/3D FEM Thermal Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme & CSS Injection ---
def inject_custom_css():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        /* Metric Box Styling */
        [data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            padding: 12px 16px;
            border-radius: 8px;
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.8rem;
            font-weight: 500;
        }
        [data-testid="stMetricValue"] {
            color: #38BDF8 !important;
            font-family: 'JetBrains Mono', monospace, sans-serif;
            font-size: 1.4rem;
            font-weight: 700;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        /* Primary Button */
        div.stButton > button {
            background-color: #0284C7;
            color: #FFFFFF;
            border-radius: 6px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
        }
        div.stButton > button:hover {
            background-color: #0369A1;
            color: #FFFFFF;
        }

        /* Secondary Download Button */
        div.stDownloadButton > button {
            background-color: #1E293B;
            color: #F8FAFC;
            border: 1px solid #334155;
            border-radius: 6px;
            font-weight: 600;
        }
        div.stDownloadButton > button:hover {
            background-color: #334155;
            color: #FFFFFF;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Top Header & Collapsible Overview ---
st.title("2D/3D FEM Heat Transfer Solver")

with st.expander("Project Overview & Highlights", expanded=False):
    st.markdown("""
    * **High-Density Domain Support:** Formulated for 10,000+ nodes and 9,800+ hybrid T3/Q4 elements.
    * **Verification:** Validated against analytical Laplace solutions with < 1.5% L2 relative error.
    * **Performance:** Accelerated system assembly and direct solve times using SciPy Compressed Sparse Row (CSR) storage.
    * **Boundary Conditions:** Spatial coordinate-based Dirichlet boundary condition enforcement.
    """)

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Control Panel")

st.sidebar.subheader("Mesh Density")
num_r = st.sidebar.slider("Radial Divisions", min_value=10, max_value=60, value=40)
num_theta = st.sidebar.slider("Angular Divisions", min_value=45, max_value=240, value=180)

st.sidebar.subheader("Material Properties")
kx = st.sidebar.number_input("Thermal Conductivity kx (W/m K)", value=25.0, step=1.0)
ky = st.sidebar.number_input("Thermal Conductivity ky (W/m K)", value=25.0, step=1.0)

st.sidebar.subheader("Boundary Conditions")
left_bc = st.sidebar.number_input("Left Boundary Temp (C)", value=20.0, step=5.0)
right_bc = st.sidebar.number_input("Right Boundary Temp (C)", value=-20.0, step=5.0)

run_button = st.sidebar.button("Run Simulation", use_container_width=True)

# --- Execution Logic ---
if run_button:
    with st.spinner("Executing finite element solver pipeline..."):
        start_time = time.perf_counter()
        
        # 1. Mesh Generation
        coords, elements, element_types = generate_sample_mesh(num_r=num_r, num_theta=num_theta)
        
        mesh = Mesh.generate_structured(nx=10, ny=10, lx=1.0, ly=1.0)
        mesh.coords = coords
        mesh.elements = elements
        mesh.element_types = element_types
        
        # 2. Element Matrices
        k_elem_dict = {}
        for elem_id in range(mesh.num_elements):
            elem_coords = mesh.coords[mesh.elements[elem_id]]
            e_type = mesh.element_types[elem_id]
            if e_type == "T3":
                k_elem_dict[elem_id] = T3Element.compute_ke(elem_coords, kx, ky)
            elif e_type == "Q4":
                k_elem_dict[elem_id] = Q4Element.compute_ke(elem_coords, kx, ky)
                
        # 3. Global Assembly
        K_global, F_global = assemble_global_system(mesh, k_elem_dict)
        
        # 4. Apply Boundary Conditions
        left_nodes = np.where(np.isclose(mesh.coords[:, 0], -1.0))[0]
        right_nodes = np.where(np.isclose(mesh.coords[:, 0], 1.0))[0]
        
        dirichlet_bcs = {node: left_bc for node in left_nodes}
        dirichlet_bcs.update({node: right_bc for node in right_nodes})
        
        K_bc, F_bc = apply_dirichlet_bcs(K_global, F_global, dirichlet_bcs)
        
        # 5. Solve System
        T = solve_system(K_bc, F_bc)
        
        # 6. Post-Processing
        centroids, grad_T, q = compute_flux_fields(mesh, T, kx, ky)
        q_magnitude = np.linalg.norm(q, axis=1)
        
        # 7. File Export
        vtk_filename = "solution.vtk"
        export_to_vtk(vtk_filename, mesh, T, q)
        
        end_time = time.perf_counter()
        solve_duration = end_time - start_time

    # --- KPI Metrics Banner ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nodes", f"{mesh.num_nodes:,}")
    m2.metric("Elements", f"{mesh.num_elements:,}")
    m3.metric("Max Temp", f"{T.max():.2f} C")
    m4.metric("Max Heat Flux", f"{q_magnitude.max():.2f} W/m2")
    m5.metric("Compute Time", f"{solve_duration:.3f} s")

    st.divider()

    # --- Split View Layout (70% Plot / 30% Analytics) ---
    plot_col, analytics_col = st.columns([7, 3])

    with plot_col:
        st.subheader("Interactive Thermal Topography")
        
        x = mesh.coords[:, 0]
        y = mesh.coords[:, 1]
        z = T.flatten()
        
        tri_i, tri_j, tri_k = [], [], []
        for elem, e_type in zip(mesh.elements, mesh.element_types):
            if e_type == "T3":
                tri_i.append(elem[0])
                tri_j.append(elem[1])
                tri_k.append(elem[2])
            elif e_type == "Q4":
                tri_i.extend([elem[0], elem[0]])
                tri_j.extend([elem[1], elem[2]])
                tri_k.extend([elem[2], elem[3]])

        fig = go.Figure(data=[go.Mesh3d(
            x=x, y=y, z=z, 
            intensity=z, 
            colorscale='Inferno',
            colorbar_title="Temp (C)",
            i=tri_i, j=tri_j, k=tri_k,
            showscale=True
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)', 
                yaxis_title='Y (m)', 
                zaxis_title='Temperature (C)',
                aspectmode='manual',
                aspectratio=dict(x=1.4, y=1.4, z=0.7)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=620
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with analytics_col:
        st.subheader("Field Summary")
        
        with st.container(border=True):
            st.markdown("**Thermal Statistics**")
            st.write(f"Minimum Temperature: `{T.min():.2f} C`")
            st.write(f"Mean Temperature: `{T.mean():.2f} C`")
            st.write(f"Maximum Temperature: `{T.max():.2f} C`")
            st.write(f"Mean Heat Flux: `{q_magnitude.mean():.2f} W/m2`")

        with st.container(border=True):
            st.markdown("**Material & Domain**")
            st.write(f"Conductivity Matrix: `kx={kx:.1f}, ky={ky:.1f}`")
            st.write(f"Left BC: `{left_bc:.1f} C`")
            st.write(f"Right BC: `{right_bc:.1f} C`")

        with st.container(border=True):
            st.markdown("**Data Export**")
            st.caption("Export VTK file for post-processing in ParaView.")
            if os.path.exists(vtk_filename):
                with open(vtk_filename, "rb") as file:
                    st.download_button(
                        label="Download VTK Results",
                        data=file,
                        file_name="fea_results.vtk",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
else:
    st.info("Adjust the parameters in the sidebar and click Run Simulation to view results.")