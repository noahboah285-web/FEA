# src/solver.py
import numpy as np
from scipy.sparse.linalg import spsolve

def solve_system(K, F):
    """
    Solves the linear system K * T = F using direct sparse linear solver.
    
    Parameters:
        K (csr_matrix): Modified stiffness matrix with BCs applied.
        F (ndarray): Modified force vector with BCs applied.
        
    Returns:
        T (ndarray): Computed nodal temperatures (Nnodes,).
    """
    T = spsolve(K, F)
    return T