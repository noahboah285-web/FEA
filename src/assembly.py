# src/assembly.py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def assemble_global_system(mesh, k_elem_dict, f_elem_dict=None):
    """
    Assembles local element stiffness matrices (T3, Q4, etc.) into a global sparse CSR matrix.
    
    Parameters:
        mesh (Mesh): Mesh object containing node coordinates and elements array.
        k_elem_dict (dict): Dictionary mapping elem_id -> local stiffness matrix.
        f_elem_dict (dict, optional): Dictionary mapping elem_id -> local load vector.
        
    Returns:
        K_global (csr_matrix): Assembled global stiffness matrix (num_nodes x num_nodes).
        F_global (ndarray): Assembled global force/thermal load vector (num_nodes,).
    """
    num_nodes = len(mesh.coords)
    num_elements = len(mesh.elements)
    
    rows = []
    cols = []
    data = []
    
    F_global = np.zeros(num_nodes)
    
    for elem_id in range(num_elements):
        nodes = mesh.elements[elem_id]
        k_e = k_elem_dict[elem_id]
        n_nodes = len(nodes)  # Dynamic check: 3 for T3, 4 for Q4
        
        # 1. Accumulate Element Load Vector (if present)
        if f_elem_dict and elem_id in f_elem_dict:
            f_e = np.asarray(f_elem_dict[elem_id]).flatten()
            for i in range(n_nodes):
                F_global[nodes[i]] += f_e[i]
                
        # 2. Accumulate Element Stiffness Matrix Triplet Entry
        rows.extend(np.repeat(nodes, n_nodes))
        cols.extend(np.tile(nodes, n_nodes))
        data.extend(k_e.flatten())
                
    # Create COO matrix and convert to CSR for matrix operations
    K_global = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    
    return K_global, F_global