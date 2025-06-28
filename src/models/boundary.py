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




@partial(jit, static_argnums=(2, 3))
def apply_dirichlet_bcs_sparse_2d(matrix: BCOO,
                                  source: jnp.ndarray,
                                  bc_nodes,   # e.g. a tuple or list (static)
                                  bc_values): # e.g. a tuple or list (static)
    # Convert static boundary info to JAX arrays.
    bc_nodes = jnp.array(bc_nodes)
    bc_values = jnp.array(bc_values)
    
    n = matrix.shape[0]
    indices = matrix.indices   # shape (n_entries, 2)
    data = matrix.data         # shape (n_entries,)

    # Create a boolean mask: True if the row index is one of the boundary nodes.
    mask = jnp.isin(indices[:, 0], bc_nodes)
    # Instead of filtering, zero out entries in rows corresponding to boundary nodes.
    new_data = jnp.where(mask, 0.0, data)

    # Now, add the Dirichlet diagonal entries.
    diag_indices = jnp.stack([bc_nodes, bc_nodes], axis=1)
    diag_data = jnp.ones(bc_nodes.shape[0])

    # Concatenate the modified data with the new diagonal entries.
    combined_data = jnp.concatenate([new_data, diag_data])
    combined_indices = jnp.concatenate([indices, diag_indices])

    # Build the new sparse matrix.
    new_matrix = BCOO((combined_data, combined_indices), shape=(n, n))
    # Compute the new number of nonzeros: original nse plus one entry per boundary node.
    new_nse = matrix.nse + bc_nodes.shape[0]
    new_matrix = new_matrix.sum_duplicates(nse=new_nse)

    # Update the source vector at the Dirichlet nodes.
    new_source = source.at[bc_nodes].set(bc_values)

    return new_matrix, new_source






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


# @partial(jit, static_argnums=(0,))
@jit 
def apply_freedrainage_water_bcs(source: jnp.ndarray,
                                hydraulic_conductivity: jnp.ndarray,
                                boundary_nodes: jnp.ndarray,
                                points: jnp.ndarray,
                                ksi_1d: jnp.ndarray,
                                w_1d: jnp.ndarray) -> jnp.ndarray:
    """
    Apply free drainage boundary conditions for water flow (Richards equation).
    
    Args:
        source: Global source vector
        hydraulic_conductivity: Hydraulic conductivity values at nodes
        boundary_nodes: Free drainage boundary node indices
        points: Mesh point coordinates
        ksi_1d: 1D quadrature points
        w_1d: 1D quadrature weights
        
    Returns:
        Modified source vector with free drainage contribution
    """
    def apply_freedrainage(i, Global_source):
        I, J = boundary_nodes[i], boundary_nodes[i+1]
        x1, x2 = points[I, 0], points[J, 0]
        local_Konduc = hydraulic_conductivity[jnp.array([I, J])]
        Le = x2 - x1
        Jac = Le / 2
        N, dN = shape_functions(ksi_1d)
        
        def integrate_point(j, local_source):
            K_avg = jnp.dot(N[:, j], local_Konduc)
            # For free drainage: q = -K (outflow, negative flux)
            # Source term should be NEGATIVE for outflow
            source_term = -Jac * w_1d[j] * K_avg * N[:, j]  # NEGATIVE for outflow!
            return local_source + source_term
        
        local_source = jax.lax.fori_loop(0, 2, integrate_point, jnp.zeros(2))
        Global_source = Global_source.at[jnp.array([I, J])].add(local_source)
        return Global_source
    
    # Apply free drainage to all boundary segments
    return lax.fori_loop(0, boundary_nodes.shape[0] - 1, apply_freedrainage, source)

                    

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
def apply_freedrainage_nitrogen_bcs(Global_matrix: jnp.ndarray,
                                   Global_source: jnp.ndarray, 
                                   hydraulic_conductivity: jnp.ndarray,
                                   water_content: jnp.ndarray,
                                   freedrainage_nodes: jnp.ndarray,
                                   points: jnp.ndarray,
                                   ksi_1d: jnp.ndarray,
                                   w_1d: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply free drainage boundary conditions for nitrogen species transport."""
    
    def apply_single_edge(i, carry):
        Global_matrix_current, Global_source_current = carry
        I, J = freedrainage_nodes[i], freedrainage_nodes[i + 1]
        x1, x2 = points[I, 0], points[J, 0]
        edge_length = x2 - x1
        jacobian_1d = edge_length / 2.0
        
        local_conductivity = hydraulic_conductivity[jnp.array([I, J])]
        local_water_content = water_content[jnp.array([I, J])]
        
        N, dN = shape_functions(ksi_1d)
        
        def integrate_quadrature_point(j, local_carry):
            local_matrix, local_source = local_carry
            K_avg = jnp.dot(N[:, j], local_conductivity)
            theta_avg = jnp.dot(N[:, j], local_water_content)
            water_velocity = -K_avg / theta_avg
            
            matrix_contribution = (jacobian_1d * w_1d[j] * water_velocity * 
                                 jnp.outer(N[:, j], N[:, j]))
            source_contribution = jnp.zeros(2)
            
            return local_matrix + matrix_contribution, local_source + source_contribution
        
        local_matrix_total, local_source_total = lax.fori_loop(
            0, len(w_1d), integrate_quadrature_point, 
            (jnp.zeros((2, 2)), jnp.zeros(2))
        )
        
        node_indices = jnp.array([I, J])
        Global_matrix_new = Global_matrix_current.at[
            node_indices[:, None], node_indices
        ].add(local_matrix_total)
        
        Global_source_new = Global_source_current.at[node_indices].add(local_source_total)
        
        return Global_matrix_new, Global_source_new
    
    final_matrix, final_source = lax.fori_loop(
        0, freedrainage_nodes.shape[0] - 1, 
        apply_single_edge, 
        (Global_matrix, Global_source)
    )
    
    return final_matrix, final_source

@partial(jit, static_argnums=()) 
def apply_freedrainage_nitrogen_bcs_sparse(matrix: BCOO,
                                          source: jnp.ndarray,
                                          hydraulic_conductivity: jnp.ndarray,
                                          water_content: jnp.ndarray,
                                          freedrainage_nodes: jnp.ndarray,
                                          points: jnp.ndarray,
                                          ksi_1d: jnp.ndarray,
                                          w_1d: jnp.ndarray) -> Tuple[BCOO, jnp.ndarray]:
    """Apply free drainage boundary conditions for nitrogen species transport using sparse matrices."""
    
    def process_boundary_element(_, i):
        """Process a single boundary element to compute local contributions."""
        I, J = freedrainage_nodes[i], freedrainage_nodes[i + 1]
        x1, x2 = points[I, 0], points[J, 0]
        edge_length = x2 - x1
        jacobian_1d = edge_length / 2.0
        
        # Get local nodal values for this boundary element
        local_conductivity = hydraulic_conductivity[jnp.array([I, J])]
        local_water_content = water_content[jnp.array([I, J])]
        
        N, dN = shape_functions(ksi_1d)
        
        # For 2x2 local matrix, we'll have 4 entries following the same pattern as Cauchy
        indices = jnp.array([[I, I], [I, J], [J, I], [J, J]])
        
        local_matrix = jnp.zeros(4)  # Will hold [M_II, M_IJ, M_JI, M_JJ]
        local_source = jnp.zeros(2)  # Source contributions for nodes I and J
        
        # Integrate over the boundary element using quadrature
        for j in range(len(w_1d)):
            # Evaluate material properties at quadrature point
            K_avg = jnp.dot(N[:, j], local_conductivity)
            theta_avg = jnp.dot(N[:, j], local_water_content)
            
            # Compute water velocity for free drainage: v = -K/θ (unit gradient assumption)
            water_velocity = -K_avg / theta_avg
            
            # Weight for this quadrature point
            weight = jacobian_1d * w_1d[j] * water_velocity
            
            # Shape function values at this quadrature point
            N_vals = N[:, j]
            
            # Compute matrix entries: weight * N_i * N_j
            # This represents the convective boundary contribution
            local_matrix = local_matrix.at[0].add(weight * N_vals[0] * N_vals[0])  # M_II
            local_matrix = local_matrix.at[1].add(weight * N_vals[0] * N_vals[1])  # M_IJ
            local_matrix = local_matrix.at[2].add(weight * N_vals[1] * N_vals[0])  # M_JI
            local_matrix = local_matrix.at[3].add(weight * N_vals[1] * N_vals[1])  # M_JJ
            
            # For free drainage, typically no external source term
            # The boundary condition is naturally satisfied through the velocity term
        
        # Source contributions are typically zero for free drainage
        source_I = 0.0
        source_J = 0.0
        
        return local_matrix, indices, I, J, source_I, source_J
    
    # Process all boundary elements using vmap (same pattern as Cauchy)
    results = vmap(process_boundary_element, in_axes=(None, 0))(
        None, jnp.arange(freedrainage_nodes.shape[0]-1))
    local_matrices, local_indices, Is, Js, sources_I, sources_J = results
    
    # Flatten matrix data (same pattern as Cauchy)
    boundary_data = local_matrices.reshape(-1)
    boundary_indices = local_indices.reshape(-1, 2)
    
    # Create boundary matrix and combine with existing matrix (same pattern as Cauchy)
    boundary_matrix = BCOO((boundary_data, boundary_indices), shape=matrix.shape)
    total_nse = matrix.nse + boundary_matrix.nse
    combined_data = jnp.concatenate([matrix.data, boundary_matrix.data])
    combined_indices = jnp.concatenate([matrix.indices, boundary_matrix.indices])
    
    # Create combined matrix and sum duplicates (same pattern as Cauchy)
    combined_matrix = BCOO((combined_data, combined_indices), shape=matrix.shape)
    matrix_updated = combined_matrix.sum_duplicates(nse=total_nse)
    
    # Update source vector - typically no contribution for free drainage
    # But we include the framework in case needed
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
    neumann_nodes = jnp.array([], dtype=jnp.int64)
    freedrainage_nodes = jnp.array([], dtype=jnp.int64)
    cauchy_nodes = jnp.array([], dtype=jnp.int64)
    
    return BoundaryNodes(
        top=top_nodes,
        bottom=bottom_nodes,
        neumann=neumann_nodes,
        freedrainage=freedrainage_nodes,
        cauchy=cauchy_nodes
    )

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
        
    elif test_case == 'NitrogenTest':  # ADD THIS NEW CASE HERE
        # Specific boundaries for nitrogen transport test
        # Top boundary (y = 1.0) - Inflow boundary for nitrogen species
        top_nodes = jnp.where(jnp.abs(points[:, 1] - 1.0) < 1e-10)[0]
        
        # Neumann boundary: water infiltration at top
        neumann_nodes = top_nodes
        
        # Free drainage boundary: bottom boundary (y = 0.0) for natural outflow
        freedrainage_nodes = jnp.where(jnp.abs(points[:, 1] - 0.0) < 1e-10)[0]
        
        # Cauchy boundary: same as neumann for nitrogen species inflow
        cauchy_nodes = neumann_nodes
        
        # Update bottom_nodes for nitrogen test (should be the drainage boundary)
        bottom_nodes = freedrainage_nodes
        
    else:
        # For original test cases (Test1, Test2)
        top_nodes = jnp.where(points[:, 1] == 15.24)[0]
        neumann_nodes = jnp.array([], dtype=jnp.int64)
        freedrainage_nodes = jnp.array([], dtype=jnp.int64)
        cauchy_nodes = jnp.array([], dtype=jnp.int64)
    
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
    elif test_case == 'NitrogenTest':
        return {
            'type': 'coupled',
            'water_flux': {
                'neumann_flux': -0.05,  # Infiltration rate (m/day)
                'has_neumann': True,
                'has_freedrainage': True
            },
            'solute_flux': {
                'has_cauchy': True,
                'cauchy_flux': -0.05,   # Same as water flux
                'has_freedrainage': True,
                'inlet_concentrations': {
                    'NH4': 50.0,  # mg/L
                    'NO2': 0.0,   # mg/L  
                    'NO3': 20.0   # mg/L
                }
            }
        }
    else:
        raise ValueError(f"Invalid test case: {test_case}")
    




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
