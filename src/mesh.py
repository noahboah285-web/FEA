# src/mesh.py
import numpy as np
import meshio


class Mesh:
    """
    Unified Mesh class supporting both structured rectangular grids (Q4)
    and unstructured dual-element meshes (T3 & Q4) imported via meshio.
    """
    def __init__(self, coords=None, elements=None, element_types=None, physical_boundaries=None):
        self.coords = np.array(coords, dtype=float) if coords is not None else np.empty((0, 2))
        self.elements = list(elements) if elements is not None else []  # List of lists to allow mixed element topologies (3-node vs 4-node)
        self.element_types = list(element_types) if element_types is not None else []  # e.g., ["T3", "T3", "Q4", ...]
        self.physical_boundaries = physical_boundaries if physical_boundaries is not None else {}

    @property
    def num_elements(self):
        return len(self.elements)

    @property
    def num_nodes(self):
        return len(self.coords)

    # -------------------------------------------------------------------------
    # Factory Method 1: Structured Rectangular Mesh (Phase 1 & 2 Legacy)
    # -------------------------------------------------------------------------
    @classmethod
    def generate_structured(cls, nx, ny, lx, ly):
        """Generates a structured grid of Q4 elements."""
        x = np.linspace(0.0, lx, nx + 1)
        y = np.linspace(0.0, ly, ny + 1)
        
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = np.column_stack([X.ravel(), Y.ravel()])
        
        node_id = lambda i, j: i * (ny + 1) + j
        
        elements = []
        element_types = []
        for i in range(nx):
            for j in range(ny):
                n1 = node_id(i, j)        # Bottom-Left
                n2 = node_id(i + 1, j)    # Bottom-Right
                n3 = node_id(i + 1, j + 1)# Top-Right
                n4 = node_id(i, j + 1)    # Top-Left
                elements.append([n1, n2, n3, n4])
                element_types.append("Q4")
                
        return cls(coords, elements, element_types)

    # -------------------------------------------------------------------------
    # Factory Method 2: Unstructured Gmsh/meshio Reader (Phase 3)
    # -------------------------------------------------------------------------
    @classmethod
    def from_file(cls, filepath):
        """
        Parses .msh, .vtk, or other mesh formats using meshio.
        Extracts 2D elements (T3 / Q4) and physical boundary tags.
        """
        m = meshio.read(filepath)
        coords = m.points[:, :2]  # Extract 2D coordinates (X, Y)

        elements = []
        element_types = []
        physical_boundaries = {}

        # 1. Parse 2D Cell Elements
        for cell_block in m.cells:
            if cell_block.type == "triangle":
                for elem in cell_block.data:
                    elements.append(list(elem))
                    element_types.append("T3")
            elif cell_block.type == "quad":
                for elem in cell_block.data:
                    elements.append(list(elem))
                    element_types.append("Q4")

        # 2. Parse Boundary Physical Markers (Line Elements)
        if "gmsh:physical" in m.cell_data:
            for cell_block, physical_tags in zip(m.cells, m.cell_data["gmsh:physical"]):
                if cell_block.type == "line":
                    for line_nodes, tag in zip(cell_block.data, physical_tags):
                        tag = int(tag)
                        if tag not in physical_boundaries:
                            physical_boundaries[tag] = []
                        physical_boundaries[tag].append(list(line_nodes))

        return cls(coords, elements, element_types, physical_boundaries)

    # -------------------------------------------------------------------------
    # Boundary Node Extraction Methods
    # -------------------------------------------------------------------------
    def get_nodes_on_boundary(self, boundary_name, tol=1e-8):
        """Coordinate-based boundary node lookup for structured grids."""
        boundary_name = boundary_name.lower()
        lx = np.max(self.coords[:, 0])
        ly = np.max(self.coords[:, 1])

        if boundary_name == 'left':
            return np.where(np.abs(self.coords[:, 0] - 0.0) < tol)[0]
        elif boundary_name == 'right':
            return np.where(np.abs(self.coords[:, 0] - lx) < tol)[0]
        elif boundary_name == 'bottom':
            return np.where(np.abs(self.coords[:, 1] - 0.0) < tol)[0]
        elif boundary_name == 'top':
            return np.where(np.abs(self.coords[:, 1] - ly) < tol)[0]
        else:
            raise ValueError(f"Unknown boundary name: {boundary_name}")

    def get_boundary_nodes_by_tag(self, tag):
        """Returns unique node IDs belonging to a Gmsh physical tag."""
        if tag not in self.physical_boundaries:
            return np.array([], dtype=int)
        lines = np.array(self.physical_boundaries[tag])
        return np.unique(lines.flatten())


if __name__ == "__main__":
    # Test 1: Structured Grid (Backward Compatibility)
    mesh_struct = Mesh.generate_structured(nx=2, ny=1, lx=2.0, ly=1.0)
    print(f"--- Structured Mesh ---")
    print(f"Num Nodes: {mesh_struct.num_nodes}")
    print(f"Num Elements: {mesh_struct.num_elements}")
    print(f"Element Types: {set(mesh_struct.element_types)}")

    # Test 2: Unstructured File Import (if .msh file exists)
    try:
        mesh_unstruct = Mesh.from_file("plate_with_hole.msh")
        print(f"\n--- Unstructured Mesh ---")
        print(f"Num Nodes: {mesh_unstruct.num_nodes}")
        print(f"Num Elements: {mesh_unstruct.num_elements}")
        print(f"Element Types Present: {set(mesh_unstruct.element_types)}")
        print(f"Physical Boundary Tags: {list(mesh_unstruct.physical_boundaries.keys())}")
    except FileNotFoundError:
        print("\nNote: 'plate_with_hole.msh' not found. Run main.py to auto-generate a sample mesh.")