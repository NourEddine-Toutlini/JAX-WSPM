# src/models/boundary.py
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Dict, NamedTuple
from functools import partial
from jax import jit, vmap, lax
from jax.experimental.sparse import BCOO

class BoundaryNodes(NamedTuple):
    """Container for boundary node indices."""
    top: jnp.ndarray
    bottom: jnp.ndarray
    neumann: jnp.ndarray
    freedrainage: jnp.ndarray
    cauchy: jnp.ndarray

@jit
def upper_boundary_condition_test1(x: jnp.ndarray, L: float, alpha0: float, eps0: jnp.ndarray) -> jnp.ndarray:
    """Test case 1: Complex sinusoidal boundary condition."""
    return (1/alpha0) * jnp.log(eps0 + (1 - eps0) * 
                               ((3/4) * jnp.sin(jnp.pi * x / L) - 
                                (1/4) * jnp.sin(3 * jnp.pi * x / L)))

@jit
def upper_boundary_condition_test2(x: jnp.ndarray, L: float, alpha0: float, eps0: jnp.ndarray) -> jnp.ndarray:
    """Test case 2: Simple sinusoidal boundary condition."""
    return (1/alpha0) * jnp.log(eps0 + (1 - eps0) * jnp.sin(jnp.pi * x / L))

@jit
def upper_boundary_condition_test3(x: jnp.ndarray, L: float, alpha0: float, eps0: jnp.ndarray) -> jnp.ndarray:
    """Test case 3: Cosine-based boundary condition."""
    return (1/alpha0) * jnp.log(eps0 + 0.5 * (1 - eps0) * 
                               (1 - jnp.cos(2 * jnp.pi * x / L)))

# def extract_boundary_nodes(points: jnp.ndarray, test_case: str = 'Test1') -> BoundaryNodes:
#     """
#     Extract boundary nodes based on test case.
    
#     Args:
#         points: Array of mesh point coordinates
#         test_case: Name of the test case
        
#     Returns:
#         BoundaryNodes tuple containing boundary node arrays
#     """
#     if test_case == 'Test3':
#         # For Test3: only bottom boundary
#         bottom_nodes = jnp.where(points[:, 1] == 0)[0]
#     else:
#         # For Test1, Test2, and SoluteTest: bottom, left, and right boundaries
#         bottom_nodes = jnp.where(
#             (points[:, 1] == 0) |   # Bottom
#             (points[:, 0] == 0) |   # Left
#             (points[:, 0] == 15.24) # Right
#         )[0]

#     if test_case == 'SoluteTest':
#         # Specific boundaries for solute transport test
#         top_nodes = jnp.where((points[:, 1] == 2.0) & 
#                            (points[:, 0] <= 0.05) & 
#                            (points[:, 0] >= -0.05))[0]
#         neumann_nodes = top_nodes
#         freedrainage_nodes = jnp.where(points[:, 1] == 0)[0]
#         cauchy_nodes = neumann_nodes
#     else:
#         # For original test cases
#         top_nodes = jnp.where(points[:, 1] == 15.24)[0]
#         neumann_nodes = jnp.array([], dtype=jnp.int64)
#         freedrainage_nodes = jnp.array([], dtype=jnp.int64)
#         cauchy_nodes = jnp.array([], dtype=jnp.int64)
    
#     return BoundaryNodes(
#         top=top_nodes,
#         bottom=bottom_nodes,
#         neumann=neumann_nodes,
#         freedrainage=freedrainage_nodes,
#         cauchy=cauchy_nodes
#     )
        
    

# def get_boundary_condition(test_case: str) -> Dict[str, any]:
#     """Get boundary condition functions and parameters for a specific test case."""
#     # Original test cases
#     if test_case in ['Test1', 'Test2', 'Test3']:
#         bc_funcs = {
#             'Test1': upper_boundary_condition_test1,
#             'Test2': upper_boundary_condition_test2,
#             'Test3': upper_boundary_condition_test3
#         }
#         return {
#             'type': 'richards_only',
#             'upper_bc': bc_funcs[test_case]
#         }
    
#     # Solute transport test case
#     elif test_case == 'SoluteTest':
#         return {
#             'type': 'coupled',
#             'water_flux': {
#                 'neumann_flux': -0.1 # Constant water flux at top boundary
#             },
#             'solute': {
#                 'cauchy_flux': -0.1,  # Solute flux for Cauchy BC
#                 'inlet_conc': 1.0   # Inlet concentration
#             }
#         }
    
#     else:
#         raise ValueError(f"Invalid test case: {test_case}")

#@partial(jit, static_argnums=(3,))
@jit
def apply_dirichlet_bcs(matrix: jnp.ndarray,
                       source: jnp.ndarray,
                       bc_nodes: jnp.ndarray,
                       bc_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Dirichlet boundary conditions."""
    def apply_bc(i, carries):
        matrix_dense, source = carries
        idx = bc_nodes[i]
        val = bc_values[i]
        matrix_dense = matrix_dense.at[idx].set(0.0)
        matrix_dense = matrix_dense.at[idx, idx].set(1.0)
        source = source.at[idx].set(val)
        return matrix_dense, source
    
    return lax.fori_loop(0, bc_nodes.shape[0], apply_bc, (matrix, source))  
       

def shape_functions(ksi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute 1D shape functions and their derivatives."""
    N = jnp.stack([(1 - ksi) / 2, (1 + ksi) / 2])
    dN = jnp.array([-1/2, 1/2])
    return N, dN



@partial(jit, static_argnums=(1,))
def apply_neumann_bcs(source: jnp.ndarray,
                     flux: float,
                     boundary_nodes: jnp.ndarray,
                     points: jnp.ndarray,
                     ksi_1d: jnp.ndarray,
                     w_1d: jnp.ndarray) -> jnp.ndarray:
    def apply_neumann(i, source):
        I, J = boundary_nodes[i], boundary_nodes[i+1]
        x1, x2 = points[I, 0], points[J, 0]
        Le = x2 - x1
        Jac = Le / 2
        N, dN = shape_functions(ksi_1d)
        
        def integrate_point(j, local_source):
            return local_source - Jac * w_1d[j] * flux * N[:,j]

        local_source = jax.lax.fori_loop(0, 2, integrate_point, jnp.zeros(2))
        return source.at[jnp.array([I, J])].add(local_source)
    
    return lax.fori_loop(0, boundary_nodes.shape[0]-1, apply_neumann, source)



                    

@partial(jit, static_argnums=()) 
def apply_cauchy_bcs(matrix: jnp.ndarray,
                    source: jnp.ndarray,
                    cauchy_flux: float,
                    inlet_conc: float,
                    boundary_nodes: jnp.ndarray,
                    ksi_1d: jnp.ndarray,
                    w_1d: jnp.ndarray,
                    points: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Cauchy boundary conditions for solute transport."""
    def apply_cauchy(i, carries):
        matrix, source = carries
        I, J = boundary_nodes[i], boundary_nodes[i+1]
        x1, x2 = points[I, 0], points[J, 0]
        Le = x2 - x1
        Jac = Le / 2
        N, dN = shape_functions(ksi_1d)
        
        def integrate_point(j, carry):
            local_matrix, local_source = carry
            matrix_term = (Jac * w_1d[j] * cauchy_flux * inlet_conc *
                         jnp.outer(N[:, j], N[:, j]))
            source_term = (Jac * w_1d[j] * cauchy_flux * inlet_conc * N[:, j])
            return local_matrix + matrix_term, local_source + source_term
        
        local_matrix, local_source = lax.fori_loop(
            0, 2, integrate_point, (jnp.zeros((2, 2)), jnp.zeros(2)))
        
        matrix = matrix.at[jnp.array([I, J])[:, None], 
                         jnp.array([I, J])].add(local_matrix)
        source = source.at[jnp.array([I, J])].add(local_source)
        
        return matrix, source
    
    return lax.fori_loop(
        0, boundary_nodes.shape[0]-1, apply_cauchy, (matrix, source))




@partial(jit, static_argnums=()) 
def apply_cauchy_bcs_sparse(matrix: BCOO,
                           source: jnp.ndarray,
                           cauchy_flux: float,
                           inlet_conc: float,
                           boundary_nodes: jnp.ndarray,
                           ksi_1d: jnp.ndarray,
                           w_1d: jnp.ndarray,
                           points: jnp.ndarray) -> Tuple[BCOO, jnp.ndarray]:
    """Apply Cauchy BCs using sparse matrix format (2-node boundary elements)."""
    
    def process_boundary_element(_, i):
        I, J = boundary_nodes[i], boundary_nodes[i+1]
        x1, x2 = points[I, 0], points[J, 0]
        Le = x2 - x1
        Jac = Le / 2
        N, dN = shape_functions(ksi_1d)
        
        # For 2x2 local matrix, we'll have 4 entries
        indices = jnp.array([[I, I], [I, J], [J, I], [J, J]])
        
        local_matrix = jnp.zeros(4)  # Will hold [M_II, M_IJ, M_JI, M_JJ]
        for j in range(len(w_1d)):
            weight = Jac * w_1d[j] * cauchy_flux
            N_vals = N[:, j]
            # Compute all matrix entries
            local_matrix = local_matrix.at[0].add(weight * N_vals[0] * N_vals[0])  # M_II
            local_matrix = local_matrix.at[1].add(weight * N_vals[0] * N_vals[1])  # M_IJ
            local_matrix = local_matrix.at[2].add(weight * N_vals[1] * N_vals[0])  # M_JI
            local_matrix = local_matrix.at[3].add(weight * N_vals[1] * N_vals[1])  # M_JJ
            
        # Calculate source contributions
        source_I = Jac * inlet_conc * cauchy_flux * (w_1d[0] * N[0, 0] + w_1d[1] * N[0, 1])
        source_J = Jac * inlet_conc * cauchy_flux * (w_1d[0] * N[1, 0] + w_1d[1] * N[1, 1])
        
        return local_matrix, indices, I, J, source_I, source_J
    
    # Process all boundary elements
    results = vmap(process_boundary_element, in_axes=(None, 0))(
        None, jnp.arange(boundary_nodes.shape[0]-1))
    local_matrices, local_indices, Is, Js, sources_I, sources_J = results
    
    # Flatten matrix data
    boundary_data = local_matrices.reshape(-1)
    boundary_indices = local_indices.reshape(-1, 2)
    
    # Create and combine matrices
    boundary_matrix = BCOO((boundary_data, boundary_indices), shape=matrix.shape)
    total_nse = matrix.nse + boundary_matrix.nse
    combined_data = jnp.concatenate([matrix.data, boundary_matrix.data])
    combined_indices = jnp.concatenate([matrix.indices, boundary_matrix.indices])
    
    # Create combined matrix and sum duplicates
    combined_matrix = BCOO((combined_data, combined_indices), shape=matrix.shape)
    matrix_updated = combined_matrix.sum_duplicates(nse=total_nse)
    
    # Update source vector - use direct indexing with I and J
    source = source.at[Is].add(sources_I)
    source = source.at[Js].add(sources_J)
    
    return matrix_updated, source





@jit
def upper_boundary_condition_3d(x: jnp.ndarray, 
                              y: jnp.ndarray, 
                              L: float, 
                              alpha0: float, 
                              eps0: jnp.ndarray) -> jnp.ndarray:
    """
    3D test case: Upper boundary condition for pressure head.
    
    Args:
        x: x-coordinates of boundary points
        y: y-coordinates of boundary points
        L: Domain length
        alpha0: Van Genuchten parameter
        eps0: Initial condition parameter
        
    Returns:
        Pressure head values at boundary points
    """
    return (1/alpha0) * jnp.log(eps0 + (1 - eps0) * jnp.sin(jnp.pi * x / L) * jnp.sin(jnp.pi * y / L))

def extract_boundary_nodes_3d(points: jnp.ndarray) -> BoundaryNodes:
    """
    Extract boundary nodes for 3D domain.
    
    Args:
        points: Array of mesh point coordinates (Nx3)
        
    Returns:
        BoundaryNodes tuple containing boundary node arrays
    """
    L = 15.24  # Domain length
    tolerance = 1e-10  # Tolerance for floating point comparison

    # Top boundary (z = L)
    top_nodes = jnp.where(jnp.abs(points[:, 2] - L) < tolerance)[0]
    
    # Other boundaries (bottom, left, right, front, back)
    bottom_nodes = jnp.where(
        (jnp.abs(points[:, 2]) < tolerance) |        # Bottom (z = 0)
        (jnp.abs(points[:, 0]) < tolerance) |        # Left (x = 0)
        (jnp.abs(points[:, 0] - L) < tolerance) |    # Right (x = L)
        (jnp.abs(points[:, 1]) < tolerance) |        # Front (y = 0)
        (jnp.abs(points[:, 1] - L) < tolerance)      # Back (y = L)
    )[0]
    
    # For 3D we don't use these boundaries, but keep them for consistency
    neumann_nodes = jnp.array([], dtype=jnp.int32)
    freedrainage_nodes = jnp.array([], dtype=jnp.int32)
    cauchy_nodes = jnp.array([], dtype=jnp.int32)
    
    return BoundaryNodes(
        top=top_nodes,
        bottom=bottom_nodes,
        neumann=neumann_nodes,
        freedrainage=freedrainage_nodes,
        cauchy=cauchy_nodes
    )

# Update the existing extract_boundary_nodes function to handle Test3D
def extract_boundary_nodes(points: jnp.ndarray, test_case: str = 'Test1') -> BoundaryNodes:
    """
    Extract boundary nodes based on test case.
    
    Args:
        points: Array of mesh point coordinates
        test_case: Name of the test case
        
    Returns:
        BoundaryNodes tuple containing boundary node arrays
    """
    if test_case == 'Test3D':
        return extract_boundary_nodes_3d(points)
        
    # Rest of the existing function remains the same
    if test_case == 'Test3':
        # For Test3: only bottom boundary
        bottom_nodes = jnp.where(points[:, 1] == 0)[0]
    else:
        # For Test1, Test2, and SoluteTest: bottom, left, and right boundaries
        bottom_nodes = jnp.where(
            (points[:, 1] == 0) |   # Bottom
            (points[:, 0] == 0) |   # Left
            (points[:, 0] == 15.24) # Right
        )[0]

    if test_case == 'SoluteTest':
        # Specific boundaries for solute transport test
        top_nodes = jnp.where((points[:, 1] == 2.0) & 
                           (points[:, 0] <= 0.05) & 
                           (points[:, 0] >= -0.05))[0]
        neumann_nodes = top_nodes
        freedrainage_nodes = jnp.where(points[:, 1] == 0)[0]
        cauchy_nodes = neumann_nodes
    else:
        # For original test cases
        top_nodes = jnp.where(points[:, 1] == 15.24)[0]
        neumann_nodes = jnp.array([], dtype=jnp.int32)
        freedrainage_nodes = jnp.array([], dtype=jnp.int32)
        cauchy_nodes = jnp.array([], dtype=jnp.int32)
    
    return BoundaryNodes(
        top=top_nodes,
        bottom=bottom_nodes,
        neumann=neumann_nodes,
        freedrainage=freedrainage_nodes,
        cauchy=cauchy_nodes
    )

# Update the get_boundary_condition function to include Test3D
def get_boundary_condition(test_case: str) -> Dict[str, any]:
    """Get boundary condition functions and parameters for a specific test case."""
    # Original test cases and 3D test case
    if test_case in ['Test1', 'Test2', 'Test3', 'Test3D']:
        bc_funcs = {
            'Test1': upper_boundary_condition_test1,
            'Test2': upper_boundary_condition_test2,
            'Test3': upper_boundary_condition_test3,
            'Test3D': upper_boundary_condition_3d
        }
        return {
            'type': 'richards_only',
            'upper_bc': bc_funcs[test_case]
        }
    
    # Rest of the function remains the same
    elif test_case == 'SoluteTest':
        return {
            'type': 'coupled',
            'water_flux': {
                'neumann_flux': -0.1
            },
            'solute': {
                'cauchy_flux': -0.1,
                'inlet_conc': 1.0
            }
        }
    else:
        raise ValueError(f"Invalid test case: {test_case}")
    



# @jit
def apply_dirichlet_bcs_sparse(matrix: BCOO,
                              source: jnp.ndarray,
                              bc_nodes: jnp.ndarray,
                              bc_values: jnp.ndarray) -> Tuple[BCOO, jnp.ndarray]:
    """
    Apply Dirichlet boundary conditions to sparse matrix.
    
    Args:
        matrix: Sparse matrix in BCOO format
        source: RHS vector
        bc_nodes: Array of boundary node indices
        bc_values: Array of boundary values
        
    Returns:
        Tuple of (modified sparse matrix, modified source vector)
    """
    # Get matrix properties
    n = matrix.shape[0]
    indices = matrix.indices
    data = matrix.data
    
    # Create mask for rows to zero out
    row_mask = ~jnp.isin(indices[:, 0], bc_nodes)
    
    # Create diagonal entries for boundary nodes
    bc_indices = jnp.stack([bc_nodes, bc_nodes], axis=1)
    bc_data = jnp.ones(bc_nodes.shape[0])
    
    # Filter existing data and indices
    filtered_data = data[row_mask]
    filtered_indices = indices[row_mask]
    
    # Combine with boundary conditions
    new_data = jnp.concatenate([filtered_data, bc_data])
    new_indices = jnp.concatenate([filtered_indices, bc_indices])
    
    # Create new sparse matrix
    new_matrix = BCOO((new_data, new_indices), shape=(n, n))
    
    # Update source vector
    new_source = source.at[bc_nodes].set(bc_values)
    
    # Sum duplicates to ensure proper matrix format
    new_matrix = new_matrix.sum_duplicates()
    
    return new_matrix, new_source
   
# @jit
def apply_all_bcs_sparse(matrix: BCOO,
                        source: jnp.ndarray,
                        boundary_nodes: BoundaryNodes,
                        hupper: jnp.ndarray,
                        L: float = 15.24) -> Tuple[BCOO, jnp.ndarray]:
    """
    Apply all boundary conditions for 3D Richards equation.
    
    Args:
        matrix: Sparse system matrix
        source: RHS vector
        boundary_nodes: BoundaryNodes object containing boundary indices
        hupper: Upper boundary condition values
        L: Domain length (default: 15.24)
        
    Returns:
        Tuple of (modified matrix, modified source)
    """
    # Apply top boundary conditions
    matrix, source = apply_dirichlet_bcs_sparse(
        matrix,
        source,
        boundary_nodes.top,
        hupper
    )
    
    # Apply bottom and other boundary conditions
    bottom_values = jnp.full_like(boundary_nodes.bottom, -L)
    matrix, source = apply_dirichlet_bcs_sparse(
        matrix,
        source,
        boundary_nodes.bottom,
        bottom_values
    )
    
    return matrix, source
