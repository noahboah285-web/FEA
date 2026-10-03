# src/element.py
import numpy as np


class Q4Element:
    """Isoparametric 4-Node Quadrilateral Element for 2D Heat Conduction."""

    # 2x2 Gauss Quadrature points and weights for 2D body integration
    GAUSS_PTS = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    GAUSS_WTS = [1.0, 1.0]

    # 1D Gauss Quadrature points and weights for 1D edge integration
    EDGE_GAUSS_PTS = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    EDGE_GAUSS_WTS = [1.0, 1.0]

    @staticmethod
    def shape_functions(xi: float, eta: float) -> np.ndarray:
        """Evaluates bilinear shape functions N_i at (xi, eta)."""
        return 0.25 * np.array([
            (1.0 - xi) * (1.0 - eta),  # N1 (Node 0)
            (1.0 + xi) * (1.0 - eta),  # N2 (Node 1)
            (1.0 + xi) * (1.0 + eta),  # N3 (Node 2)
            (1.0 - xi) * (1.0 + eta)   # N4 (Node 3)
        ])

    @staticmethod
    def shape_function_derivatives(xi: float, eta: float) -> np.ndarray:
        """
        Evaluates derivatives of shape functions w.r.t xi and eta.
        Returns:
            dN_dxi (ndarray): Shape (2, 4) where row 0 is dN/d_xi, row 1 is dN/d_eta.
        """
        dN_dxi = 0.25 * np.array([
            [-(1.0 - eta),  (1.0 - eta), (1.0 + eta), -(1.0 + eta)], # dN/dxi
            [-(1.0 - xi),  -(1.0 + xi),  (1.0 + xi),   (1.0 - xi)]  # dN/deta
        ])
        return dN_dxi

    @classmethod
    def compute_ke(cls, node_coords: np.ndarray, kx: float, ky: float) -> np.ndarray:
        """Calculates 4x4 local element conductivity matrix using 2x2 Gauss Quadrature."""
        D = np.array([[kx, 0.0],
                      [0.0, ky]])
        
        ke = np.zeros((4, 4))
        
        for xi, w_xi in zip(cls.GAUSS_PTS, cls.GAUSS_WTS):
            for eta, w_eta in zip(cls.GAUSS_PTS, cls.GAUSS_WTS):
                dN_dnatural = cls.shape_function_derivatives(xi, eta)  # (2, 4)
                J = dN_dnatural @ node_coords  # (2, 2)
                detJ = np.linalg.det(J)
                
                if detJ <= 0.0:
                    raise ValueError(f"Non-positive Jacobian determinant det(J) = {detJ:.4e}. Check node order.")
                
                J_inv = np.linalg.inv(J)
                B = J_inv @ dN_dnatural  # (2, 4)
                
                ke += (B.T @ D @ B) * detJ * w_xi * w_eta
                
        return ke

    @classmethod
    def compute_centroid_flux(cls, node_coords: np.ndarray, T_elem: np.ndarray, kx: float, ky: float):
        """Computes temperature gradient ∇T and heat flux q at element centroid (xi=0, eta=0)."""
        dN_dnatural = cls.shape_function_derivatives(0.0, 0.0)
        J = dN_dnatural @ node_coords
        J_inv = np.linalg.inv(J)
        B = J_inv @ dN_dnatural

        grad_T = B @ T_elem  # [dT/dx, dT/dy]
        qx = -kx * grad_T[0]
        qy = -ky * grad_T[1]

        centroid = np.mean(node_coords, axis=0)
        return centroid, grad_T, np.array([qx, qy])

    # =========================================================================
    # EDGE & VOLUMETRIC INTEGRATION
    # =========================================================================

    @staticmethod
    def _get_edge_natural_coords(edge_idx: int, t: float) -> tuple[float, float]:
        """Maps a 1D edge parameter t in [-1, 1] to 2D natural coordinates (xi, eta)."""
        if edge_idx == 0:   return t, -1.0
        elif edge_idx == 1: return 1.0, t
        elif edge_idx == 2: return -t, 1.0
        elif edge_idx == 3: return -1.0, -t
        else:
            raise ValueError(f"Invalid edge_idx {edge_idx}. Must be 0, 1, 2, or 3.")

    @classmethod
    def compute_edge_jacobian(cls, node_coords: np.ndarray, edge_idx: int, t: float) -> float:
        """Calculates differential edge length J_edge = dGamma / dt at parameter t."""
        xi, eta = cls._get_edge_natural_coords(edge_idx, t)
        dN_dnatural = cls.shape_function_derivatives(xi, eta) # (2, 4)
        
        dx_dxi, dy_dxi = dN_dnatural[0] @ node_coords
        dx_deta, dy_deta = dN_dnatural[1] @ node_coords

        if edge_idx in (0, 2):    # Varying xi
            dx_dt, dy_dt = dx_dxi, dy_dxi
        elif edge_idx in (1, 3):  # Varying eta
            dx_dt, dy_dt = dx_deta, dy_deta

        return float(np.sqrt(dx_dt**2 + dy_dt**2))

    @classmethod
    def compute_fe_flux(cls, node_coords: np.ndarray, edge_idx: int, q: float) -> np.ndarray:
        """Calculates 4x1 element force vector due to prescribed edge heat flux q."""
        fe_q = np.zeros(4)
        for t, w in zip(cls.EDGE_GAUSS_PTS, cls.EDGE_GAUSS_WTS):
            xi, eta = cls._get_edge_natural_coords(edge_idx, t)
            N = cls.shape_functions(xi, eta)
            J_edge = cls.compute_edge_jacobian(node_coords, edge_idx, t)
            fe_q += N * q * J_edge * w
        return fe_q

    @classmethod
    def compute_ke_fe_convection(cls, node_coords: np.ndarray, edge_idx: int, h: float, T_inf: float) -> tuple[np.ndarray, np.ndarray]:
        """Calculates 4x4 edge convection matrix (ke_conv) and 4x1 force vector (fe_conv)."""
        ke_conv = np.zeros((4, 4))
        fe_conv = np.zeros(4)

        for t, w in zip(cls.EDGE_GAUSS_PTS, cls.EDGE_GAUSS_WTS):
            xi, eta = cls._get_edge_natural_coords(edge_idx, t)
            N = cls.shape_functions(xi, eta)
            J_edge = cls.compute_edge_jacobian(node_coords, edge_idx, t)

            ke_conv += np.outer(N, N) * h * J_edge * w
            fe_conv += N * (h * T_inf) * J_edge * w

        return ke_conv, fe_conv

    @classmethod
    def compute_fe_generation(cls, node_coords: np.ndarray, Q: float) -> np.ndarray:
        """Calculates 4x1 element force vector due to internal volumetric heat generation Q."""
        fe_Q = np.zeros(4)
        for xi, w_xi in zip(cls.GAUSS_PTS, cls.GAUSS_WTS):
            for eta, w_eta in zip(cls.GAUSS_PTS, cls.GAUSS_WTS):
                N = cls.shape_functions(xi, eta)
                dN_dnatural = cls.shape_function_derivatives(xi, eta)
                J = dN_dnatural @ node_coords
                detJ = np.linalg.det(J)

                fe_Q += N * Q * detJ * w_xi * w_eta
        return fe_Q


class T3Element:
    """Linear 3-Node Triangular Element (Constant Strain/Gradient Triangle)."""

    @classmethod
    def compute_ke(cls, node_coords: np.ndarray, kx: float, ky: float) -> np.ndarray:
        """Calculates 3x3 local element conductivity matrix analytically."""
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if area <= 0.0:
            raise ValueError(f"Degenerate or inverted triangle with area = {area}")

        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, b2, b3],
            [c1, c2, c3]
        ])

        D = np.array([[kx, 0.0],
                      [0.0, ky]])

        return area * (B.T @ D @ B)

    @classmethod
    def compute_centroid_flux(cls, node_coords: np.ndarray, T_elem: np.ndarray, kx: float, ky: float):
        """Computes constant temperature gradient ∇T and heat flux q across the triangle."""
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

        b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
        c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, b2, b3],
            [c1, c2, c3]
        ])

        grad_T = B @ T_elem
        qx = -kx * grad_T[0]
        qy = -ky * grad_T[1]

        centroid = np.mean(node_coords, axis=0)
        return centroid, grad_T, np.array([qx, qy])

    @classmethod
    def compute_fe_generation(cls, node_coords: np.ndarray, Q: float) -> np.ndarray:
        """Distributes volumetric heat generation equally to the 3 nodes (Area * Q / 3)."""
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        return np.ones(3) * (Q * area / 3.0)