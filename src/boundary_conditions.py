# src/boundary_conditions.py
import numpy as np
from scipy.sparse import csr_matrix
from src.element import Q4Element


def apply_neumann_bcs(F_global, elements, nodes, flux_bcs):
    """
    Applies Neumann (prescribed heat flux q) boundary conditions.

    Parameters:
        F_global (ndarray): Global load vector of shape (N_nodes,).
        elements (ndarray): Mesh connectivity array of shape (N_elem, 4).
        nodes (ndarray): Node coordinates array of shape (N_nodes, 2).
        flux_bcs (list of dict): Each dict defines:
            {'elem_idx': int, 'edge_idx': int, 'q': float}

    Returns:
        F_global (ndarray): Updated global load vector.
    """
    F_mod = F_global.copy()

    for bc in flux_bcs:
        elem_idx = bc['elem_idx']
        edge_idx = bc['edge_idx']
        q_val = bc['q']

        elem_nodes = elements[elem_idx]
        node_coords = nodes[elem_nodes]

        fe_q = Q4Element.compute_fe_flux(node_coords, edge_idx, q_val)
        F_mod[elem_nodes] += fe_q

    return F_mod


def apply_robin_bcs(K_global, F_global, elements, nodes, convection_bcs):
    """
    Applies Robin (convection h, T_inf) boundary conditions.

    Parameters:
        K_global (csr_matrix): Global conductivity matrix.
        F_global (ndarray): Global load vector.
        elements (ndarray): Mesh connectivity array of shape (N_elem, 4).
        nodes (ndarray): Node coordinates array of shape (N_nodes, 2).
        convection_bcs (list of dict): Each dict defines:
            {'elem_idx': int, 'edge_idx': int, 'h': float, 'T_inf': float}

    Returns:
        K_mod (csr_matrix): Updated global conductivity matrix.
        F_mod (ndarray): Updated global load vector.
    """
    K_mod = K_global.copy().tolil()
    F_mod = F_global.copy()

    for bc in convection_bcs:
        elem_idx = bc['elem_idx']
        edge_idx = bc['edge_idx']
        h_val = bc['h']
        T_inf = bc['T_inf']

        elem_nodes = elements[elem_idx]
        node_coords = nodes[elem_nodes]

        ke_conv, fe_conv = Q4Element.compute_ke_fe_convection(
            node_coords, edge_idx, h_val, T_inf
        )

        # Assemble into global system
        F_mod[elem_nodes] += fe_conv
        for i_local, i_global in enumerate(elem_nodes):
            for j_local, j_global in enumerate(elem_nodes):
                K_mod[i_global, j_global] += ke_conv[i_local, j_local]

    return K_mod.tocsr(), F_mod


def apply_internal_generation(F_global, elements, nodes, Q_dict):
    """
    Applies internal volumetric heat generation Q.

    Parameters:
        F_global (ndarray): Global load vector.
        elements (ndarray): Mesh connectivity array of shape (N_elem, 4).
        nodes (ndarray): Node coordinates array of shape (N_nodes, 2).
        Q_dict (dict or float): Map of {elem_idx: Q_val} or scalar float for uniform Q.

    Returns:
        F_global (ndarray): Updated global load vector.
    """
    F_mod = F_global.copy()

    for elem_idx, elem_nodes in enumerate(elements):
        if isinstance(Q_dict, dict):
            Q_val = Q_dict.get(elem_idx, 0.0)
        else:
            Q_val = float(Q_dict)

        if Q_val == 0.0:
            continue

        node_coords = nodes[elem_nodes]
        fe_Q = Q4Element.compute_fe_generation(node_coords, Q_val)
        F_mod[elem_nodes] += fe_Q

    return F_mod


def apply_dirichlet_bcs(K_global, F_global, dirichlet_bcs):
    """
    Applies Dirichlet boundary conditions using direct matrix elimination.
    """
    K_bc = K_global.copy().tolil()
    F_bc = F_global.copy()

    nodes = list(dirichlet_bcs.keys())
    values = np.array([dirichlet_bcs[n] for n in nodes])

    for node, val in zip(nodes, values):
        F_bc -= K_bc[:, node].toarray().flatten() * val

    F_bc[nodes] = values

    for node in nodes:
        K_bc[node, :] = 0.0
        K_bc[:, node] = 0.0
        K_bc[node, node] = 1.0

    return K_bc.tocsr(), F_bc