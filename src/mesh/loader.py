# src/mesh/loader.py

"""
Mesh loading utilities for 2D and 3D finite element analysis.
"""
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

def load_and_validate_mesh(mesh_size: str, 
                         mesh_dir: str, 
                         test_case: str = 'Test1',
                         prefix: Optional[Tuple[str, str]] = None) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Load mesh files based on test case and validate the mesh.
    
    Args:
        mesh_size: Size identifier for mesh ('25', '50', etc.)
        mesh_dir: Base directory for mesh files
        test_case: Name of test case ('Test1', 'Test2', 'Test3', 'Test3D', or 'SoluteTest')
        prefix: Optional tuple of (points_prefix, elements_prefix) for custom file naming
    
    Returns:
        Tuple containing:
        - points: Node coordinates array
        - elements: Element connectivity array (triangles for 2D, tetrahedra for 3D)
        - mesh_info: Dictionary with mesh metadata
    """
    base_path = Path(mesh_dir)
    
    # Handle different test cases
    if test_case == 'SoluteTest':
        # Load solute transport specific mesh (2D)
        p_file = base_path / 'solute' / f'p_Pinns_{mesh_size}.csv'
        t_file = base_path / 'solute' / f't_Pinns_{mesh_size}.csv'
        is_3d = False
        
    elif test_case == 'Test3D':
        # Load 3D mesh files
        if prefix:
            p_prefix, t_prefix = prefix
        else:
            p_prefix, t_prefix = 'p_3D_132651', 't_3D_132651'
            
        p_file = base_path / 'richards' / f'{p_prefix}.csv'
        t_file = base_path / 'richards' / f'{t_prefix}.csv'
        is_3d = True
        
    else:
        # Load standard 2D Richards equation mesh
        p_file = base_path / 'richards' / f'p{mesh_size}.csv'
        t_file = base_path / 'richards' / f't{mesh_size}.csv'
        is_3d = False
    
    print(f"Loading mesh files:\n  Points: {p_file}\n  Elements: {t_file}")
    
    # Validate mesh files exist
    if not p_file.exists():
        raise FileNotFoundError(f"Points file not found: {p_file}")
    if not t_file.exists():
        raise FileNotFoundError(f"Elements file not found: {t_file}")
    
    # Load mesh data
    points = pd.read_csv(p_file, header=None).values
    elements = pd.read_csv(t_file, header=None).values - 1  # Convert to 0-based indexing
    
    # Convert to JAX arrays
    points = jnp.array(points)
    elements = jnp.array(elements)
    
    # Validate dimensions based on mesh type
    if is_3d:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points array must have 3 columns (x, y, z coordinates) for 3D mesh")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("Elements array must have 4 columns (vertex indices) for tetrahedral mesh")
    else:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points array must have 2 columns (x, y coordinates) for 2D mesh")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("Elements array must have 3 columns (vertex indices) for triangular mesh")
    
    # Check for invalid indices
    if jnp.any(elements < 0) or jnp.any(elements >= len(points)):
        raise ValueError("Element indices out of bounds")
    
    # Compute mesh metadata
    mesh_info = {
        'num_points': len(points),
        'num_elements': len(elements),
        'dimension': 3 if is_3d else 2,
        'element_type': 'tetrahedra' if is_3d else 'triangles',
        'x_range': (float(points[:, 0].min()), float(points[:, 0].max())),
        'y_range': (float(points[:, 1].min()), float(points[:, 1].max())),
        'z_range': (float(points[:, 2].min()), float(points[:, 2].max())) if is_3d else None,
        'mesh_type': 'solute' if test_case == 'SoluteTest' else 'richards',
        'mesh_size': mesh_size if test_case != 'SoluteTest' else 'Pinns'
    }
    
    print(f"Loaded {mesh_info['dimension']}D mesh with {mesh_info['num_points']} points "
          f"and {mesh_info['num_elements']} {mesh_info['element_type']}")
    
    return points, elements, mesh_info