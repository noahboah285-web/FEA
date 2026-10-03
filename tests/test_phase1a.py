# tests/test_phase1a.py
import numpy as np
from config import SolverConfig
from src.mesh import generate_mesh
from src.element import Q4Element

def test_partition_of_unity():
    # Test shape functions sum to 1 at arbitrary evaluation point
    xi, eta = 0.35, -0.22
    N = Q4Element.shape_functions(xi, eta)
    assert np.isclose(np.sum(N), 1.0), "Shape functions fail partition of unity!"

def test_unit_square_ke():
    # Unit square element coordinates [0, 1] x [0, 1]
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    kx = ky = 1.0
    ke = Q4Element.compute_ke(coords, kx, ky)
    
    # Property checks for heat conduction element stiffness matrix:
    # 1. Symmetry
    assert np.allclose(ke, ke.T), "Local matrix ke must be symmetric!"
    
    # 2. Zero row-sum property (Constant temperature field produces zero heat fluxes)
    row_sums = np.sum(ke, axis=1)
    assert np.allclose(row_sums, 0.0), "Row sums of ke must equal 0!"
    
    print("All Phase 1a Verification Tests Passed Successfully!")

if __name__ == "__main__":
    test_partition_of_unity()
    test_unit_square_ke()