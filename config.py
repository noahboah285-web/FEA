# config.py
from dataclasses import dataclass

@dataclass
class SolverConfig:
    # Geometry (Domain Dimensions in meters)
    Lx: float = 1.0
    Ly: float = 1.0
    
    # Mesh discretization (Number of elements)
    Nx: int = 10
    Ny: int = 10
    
    # Thermal Conductivity (W/m·K)
    kx: float = 50.0  # e.g., Structural Steel
    ky: float = 50.0
    
    # Dirichlet Boundary Conditions (Temperatures in °C or K)
    T_left: float = 100.0
    T_right: float = 0.0
    T_top: float = 0.0
    T_bottom: float = 0.0

if __name__ == "__main__":
    cfg = SolverConfig()
    print(f"Config loaded successfully: Mesh = {cfg.Nx}x{cfg.Ny} elements.")