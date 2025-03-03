"""
Matrix assembly utilities for 3D finite element implementation with time discretization.
"""
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.experimental.sparse import BCOO
from functools import partial
from typing import Tuple
from .gauss_3D import interpolation_3d, basis_ksi_eta_zeta

@jit
def compute_local_matrices_3D_mixed_form(local_coordinates: jnp.ndarray,
                                       local_theta_m: jnp.ndarray,
                                       local_theta_0: jnp.ndarray,
                                       local_pressure_head: jnp.ndarray,
                                       local_conductivity: jnp.ndarray,
                                       local_capacity: jnp.ndarray,
                                       quad_points: jnp.ndarray,
                                       weights: jnp.ndarray,
                                       dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local matrices for the mixed form of 3D Richards equation with time discretization.
    
    Args:
        local_coordinates: Element coordinates (4x3 array for tetrahedron)
        local_theta_m: Water content at previous iteration
        local_theta_0: Initial water content
        local_pressure_head: Pressure head values
        local_conductivity: Hydraulic conductivity values
        local_capacity: Specific moisture capacity values
        quad_points: Quadrature points
        weights: Quadrature weights
        dt: Time step size
        
    Returns:
        Tuple of (local matrix, local source vector)
    """
    def integrate_point(carry, inputs):
        local_stiff, local_mass, local_source = carry
        x, w = inputs
        ksi, eta, zeta = x

        # Get shape functions and derivatives
        ref_shape_func = interpolation_3d(ksi, eta, zeta)
        ref_derivative = basis_ksi_eta_zeta(ksi, eta, zeta)

        # Compute Jacobian and its properties
        Jacob = jnp.dot(ref_derivative, local_coordinates)
        Jacob_inverse = jax.scipy.linalg.inv(Jacob)
        Det_J = jnp.linalg.det(Jacob)
        glob_derivative = jnp.dot(Jacob_inverse, ref_derivative)

        # Compute local matrix contributions
        local_stiff += w * (jnp.dot(ref_shape_func, local_conductivity)) * (jnp.dot(glob_derivative.T, glob_derivative)) * Det_J
        local_mass += w * (jnp.outer(ref_shape_func, ref_shape_func)) * Det_J
        local_source -= w * (jnp.dot(ref_shape_func, local_conductivity)) * glob_derivative.T[:, 2] * Det_J

        return (local_stiff, local_mass, local_source), None

    init_carry = (jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.zeros(4))
    (local_stiff, local_mass, local_source), _ = lax.scan(integrate_point, init_carry, (quad_points, weights))

    # Apply time discretization as in 2D version
    local_capacity = jnp.diag(local_capacity)
    local_mass = jnp.diag(jnp.sum(local_mass, axis=1))
    local_matrix = (1/dt) * (local_mass @ local_capacity) + local_stiff
    local_source = local_source - (1/dt) * (local_mass @ (local_theta_m - local_theta_0 - local_capacity @ local_pressure_head))

    return local_matrix, local_source

@partial(jit, static_argnums=(1,))
def assemble_global_matrices_sparse_3D(tetrahedra: jnp.ndarray,
                                     nnt: int,
                                     points: jnp.ndarray,
                                     theta_m: jnp.ndarray,
                                     theta_n: jnp.ndarray,
                                     pressure_head_m: jnp.ndarray,
                                     conductivity_m: jnp.ndarray,
                                     capacity_m: jnp.ndarray,
                                     quad_points: jnp.ndarray,
                                     weights: jnp.ndarray,
                                     dt: float) -> Tuple[BCOO, jnp.ndarray]:
    """
    Assemble global matrices for 3D using sparse format with time discretization.
    
    Args:
        tetrahedra: Element connectivity array
        nnt: Total number of nodes
        points: Node coordinates
        theta_m: Water content at previous iteration
        theta_n: Water content at previous time step
        pressure_head_m: Pressure head at previous iteration
        conductivity_m: Hydraulic conductivity
        capacity_m: Specific moisture capacity
        quad_points: Quadrature points
        weights: Quadrature weights
        dt: Time step size
        
    Returns:
        Tuple of (global matrix, global source vector)
    """
    def process_element(_, ie):
        nodes = tetrahedra[ie]
        local_coordinates = points[nodes]
        local_theta_m = theta_m[nodes]
        local_theta_0 = theta_n[nodes]
        local_pressure_head = pressure_head_m[nodes]
        local_conductivity = conductivity_m[nodes]
        local_capacity = capacity_m[nodes]
        
        local_matrix, local_source = compute_local_matrices_3D_mixed_form(
            local_coordinates, local_theta_m, local_theta_0, local_pressure_head,
            local_conductivity, local_capacity, quad_points, weights, dt
        )
                
        return local_matrix, local_source, nodes
        
    results = vmap(process_element, in_axes=(None, 0))(None, jnp.arange(tetrahedra.shape[0]))
    local_matrices, local_sources, all_nodes = results
    
    # Calculate total number of nonzero entries
    n_elements = tetrahedra.shape[0]
    entries_per_element = 16  # 4x4 local matrices for tetrahedra
    total_entries = n_elements * entries_per_element
    
    # Pre-allocate arrays for indices and data
    rows = jnp.zeros(total_entries, dtype=jnp.int64)
    cols = jnp.zeros(total_entries, dtype=jnp.int64)
    matrix_data = jnp.zeros(total_entries)
    
    def body_fun(i, carry):
        rows, cols, matrix_data = carry
        nodes = all_nodes[i]
        
        # Generate indices for current element
        local_rows = jnp.repeat(nodes[:, None], 4, axis=1).ravel()
        local_cols = jnp.repeat(nodes[None, :], 4, axis=0).ravel()
        
        # Calculate start index for this element
        start_idx = i * entries_per_element
        
        # Update arrays
        rows = lax.dynamic_update_slice(rows, local_rows, (start_idx,))
        cols = lax.dynamic_update_slice(cols, local_cols, (start_idx,))
        matrix_data = lax.dynamic_update_slice(matrix_data, local_matrices[i].ravel(), (start_idx,))
        
        return rows, cols, matrix_data

    # Update arrays using fori_loop
    rows, cols, matrix_data = lax.fori_loop(
        0, n_elements, body_fun, (rows, cols, matrix_data))
    
    # Create indices array for BCOO format
    indices = jnp.stack([rows, cols], axis=1)
    
    # Create sparse matrix and sum duplicates
    Global_matrix = BCOO((matrix_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)

    
    # Compute Global source
    Global_source = jnp.zeros(nnt)
    def source_body_fun(i, source):
        nodes = all_nodes[i]
        updates = local_sources[i]
        return source.at[nodes].add(updates)

    Global_source = lax.fori_loop(0, n_elements, source_body_fun, Global_source)
    
    return Global_matrix, Global_source