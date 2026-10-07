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

# Import the in-memory mesh generator
from main import generate_sample_mesh 

# --- Page Configuration ---
st.set_page_config(
    page_title="FEM Heat Transfer Solver",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Theme & CSS Injection ---
def inject_custom_css():
    st.markdown("""
    <style>
        /* Hide default Streamlit visual clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        /* Metric Cards Styling */
        [data-testid="stMetric"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            padding: 16px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        [data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-weight: 500;
            font-size: 0.875rem;
        }
        [data-testid="stMetricValue"] {
            color: #38BDF8 !important;
            font-family: 'JetBrains Mono', monospace, sans-serif;
            font-weight: 700;
        }

        /* Sidebar Container Refinement */
        [data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        /* Primary Action Button Styling */
        div.stButton > button {
            background-color: #0284C7;
            color: #FFFFFF;
            border-radius: 6px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease-in-out;
        }
        div.stButton > button:hover {
            background-color: #0369A1;
            border: none;
            color: #FFFFFF;
        }

        /* Secondary / Download Button Styling */
        div.stDownloadButton > button {
            background-color: #1E293B;
            color: #F8FAFC;
            border: 1px solid #334155;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        div.stDownloadButton > button:hover {
            background-color: #334155;
            color: #FFFFFF;
            border-color: #475569;
        }

        /* Tab Navigation Bar */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E293B;
            border-radius: 6px 6px 0px 0px;
            padding: 8px 18px;
            border: 1px solid #334155;
            color: #94A3B8;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0284C7 !important;
            color: #FFFFFF !important;
            border-color: #0284C7 !important;
        }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- UI: Header & Resume Highlights ---
st.title("2D/3D FEM Heat Transfer Solver")
st.markdown("""
**Project Highlights:**
* Engineered a 1D/2D FEM heat-transfer solver supporting 10,000+ nodes and 9,800+ elements.
* Achieved < 1.5% error versus analytical solutions across validation cases.
* Improved computational efficiency by 12x using sparse matrix storage (CSR).
* Implemented Dirichlet boundary conditions based on spatial coordinates.
""")
st.divider()

# --- UI: Sidebar Parameters ---
st.sidebar.header("Simulation Parameters")

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
        
        # 6. Solve Linear System
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
    
    # KPI Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", f"{mesh.num_nodes:,}")
    col2.metric("Total Elements", f"{mesh.num_elements:,}")
    col3.metric("Max Temperature", f"{T.max():.2f} C")
    col4.metric("Max Heat Flux", f"{q_magnitude.max():.2f} W/m2")
    
    st.divider()

    # Organized Output Tabs
    tab_vis, tab_export = st.tabs(["3D Temperature Field", "Data Export & Analysis"])

    with tab_vis:
        st.subheader("Interactive Temperature Distribution")
        st.caption("Drag to rotate, scroll to zoom. Hover over the surface to inspect nodal temperatures.")
        
        # Prepare coordinates for 3D Mesh Plotting
        x = mesh.coords[:, 0]
        y = mesh.coords[:, 1]
        z = T.flatten()
        
        # Build triangle indices supporting T3 and Q4 elements
        tri_i, tri_j, tri_k = [], [], []
        
        for elem, e_type in zip(mesh.elements, mesh.element_types):
            if e_type == "T3":
                tri_i.append(elem[0])
                tri_j.append(elem[1])
                tri_k.append(elem[2])
            elif e_type == "Q4":
                # Split quad into two triangles: (0, 1, 2) and (0, 2, 3)
                tri_i.extend([elem[0], elem[0]])
                tri_j.extend([elem[1], elem[2]])
                tri_k.extend([elem[2], elem[3]])

        # 3D Mesh Surface Plot
        fig = go.Figure(data=[go.Mesh3d(
            x=x, y=y, z=z, 
            intensity=z, 
            colorscale='Inferno',
            colorbar_title="Temp (C)",
            i=tri_i, 
            j=tri_j, 
            k=tri_k,
            showscale=True
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (m)', 
                yaxis_title='Y (m)', 
                zaxis_title='Temperature (C)',
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=1.5, z=0.8)
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab_export:
        st.subheader("Raw Data Export")
        st.write("Download the computed temperature and heat flux fields for local analysis in ParaView.")
        
        if os.path.exists(vtk_filename):
            with open(vtk_filename, "rb") as file:
                st.download_button(
                    label="Download VTK File",
                    data=file,
                    file_name="fea_results.vtk",
                    mime="application/octet-stream"
                )
else:
    st.info("Adjust the parameters in the sidebar and click Run Simulation to begin.")