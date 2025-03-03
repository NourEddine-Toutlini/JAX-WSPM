# src/numerics/assembly_solute.py

import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.experimental.sparse import BCOO
from functools import partial
from typing import Tuple
from .gauss import basis_ksi_eta, interpolation

@jit
def compute_local_matrices_solute(local_coordinates: jnp.ndarray,
                                local_theta: jnp.ndarray,
                                local_theta_n: jnp.ndarray,
                                local_water_flux_x: jnp.ndarray,
                                local_water_flux_z: jnp.ndarray,
                                local_dispersion_xx: jnp.ndarray,
                                local_dispersion_xz: jnp.ndarray,
                                local_dispersion_zz: jnp.ndarray,
                                xloc: jnp.ndarray,
                                W: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local matrices for solute transport equation.
    
    Args:
        local_coordinates: Element vertex coordinates
        local_theta: Current water content at nodes
        local_theta_n: Previous time step water content
        local_water_flux_x: x-component of water flux
        local_water_flux_z: z-component of water flux
        local_dispersion_xx: xx component of dispersion tensor
        local_dispersion_xz: xz component of dispersion tensor
        local_dispersion_zz: zz component of dispersion tensor
        xloc: Quadrature points
        W: Quadrature weights
        
    Returns:
        Tuple of (local stiffness matrix, local mass matrix)
    """
    def integrate_point(carry, inputs):
        local_stiff, local_mass = carry
        x, w = inputs
        ksi, eta = x
        
        ref_shape_func = interpolation(ksi, eta)
        ref_derivative = basis_ksi_eta(ksi, eta)
        Jacob = jnp.dot(ref_derivative, local_coordinates)
        Jacob_inverse = jnp.linalg.inv(Jacob)
        Det_J = jnp.linalg.det(Jacob)
        glob_derivative = jnp.dot(Jacob_inverse, ref_derivative)
        
        # Dispersion terms
        stiff_disp = w * (
            -(ref_shape_func @ local_dispersion_xx) * 
            (glob_derivative[0,:].reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_dispersion_xz) * 
            (glob_derivative[0,:].reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_dispersion_xz) * 
            (glob_derivative[1,:].reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_dispersion_zz) * 
            (glob_derivative[1,:].reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T)
        ) * Det_J
        
        # Advection terms
        stiff_adv = w * (
            -(ref_shape_func @ local_water_flux_x) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_water_flux_z) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T)
        ) * Det_J
        
        # Mass term
        mass = w * (-ref_shape_func @ local_theta) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        local_stiff += stiff_disp + stiff_adv
        local_mass += mass
        
        
        
        return (local_stiff, local_mass), None
    
    init_carry = (jnp.zeros((3, 3)), jnp.zeros((3, 3)))
    (local_stiff, local_mass), _ = lax.scan(integrate_point, init_carry, (xloc, W))
    return local_stiff, local_mass

@partial(jit, static_argnums=(1,))
def assemble_global_matrices_solute(triangles: jnp.ndarray,
                                  nnt: int,
                                  points: jnp.ndarray,
                                  theta: jnp.ndarray,
                                  theta_n: jnp.ndarray,
                                  water_flux_x: jnp.ndarray,
                                  water_flux_z: jnp.ndarray,
                                  dispersion_xx: jnp.ndarray,
                                  dispersion_xz: jnp.ndarray,
                                  dispersion_zz: jnp.ndarray,
                                  xloc: jnp.ndarray,
                                  W: jnp.ndarray) -> Tuple[BCOO, BCOO]:
    """
    Assemble global matrices for solute transport using sparse format.
    
    Args:
        triangles: Element connectivity array
        nnt: Total number of nodes
        points: Node coordinates
        theta: Current water content
        theta_n: Previous time step water content
        water_flux_x: x-component of water flux
        water_flux_z: z-component of water flux
        dispersion_xx: xx component of dispersion tensor
        dispersion_xz: xz component of dispersion tensor
        dispersion_zz: zz component of dispersion tensor
        xloc: Quadrature points
        W: Quadrature weights
        
    Returns:
        Tuple of (global stiffness matrix, global mass matrix) in BCOO format
    """
    def process_element(_, ie):
        nodes = triangles[ie, :3]
        local_coordinates = points[nodes]
        local_theta = theta[nodes]
        local_theta_n = theta_n[nodes]
        local_water_flux_x = water_flux_x[nodes]
        local_water_flux_z = water_flux_z[nodes]
        local_dispersion_xx = dispersion_xx[nodes]
        local_dispersion_xz = dispersion_xz[nodes]
        local_dispersion_zz = dispersion_zz[nodes]
        
        local_stiff, local_mass = compute_local_matrices_solute(
            local_coordinates, local_theta, local_theta_n,
            local_water_flux_x, local_water_flux_z,
            local_dispersion_xx, local_dispersion_xz, local_dispersion_zz,
            xloc, W
        )
        
        
        
        return local_stiff, local_mass, nodes
    
    results = vmap(process_element, in_axes=(None, 0))(None, jnp.arange(triangles.shape[0]))
    local_stiffs, local_masses, all_nodes = results
    
    # Calculate total number of nonzero entries
    n_elements = triangles.shape[0]
    entries_per_element = 9  # 3x3 local matrices
    total_entries = n_elements * entries_per_element
    
    # Pre-allocate arrays for indices and data
    rows = jnp.zeros(total_entries, dtype=jnp.int64)
    cols = jnp.zeros(total_entries, dtype=jnp.int64)
    stiff_data = jnp.zeros(total_entries, dtype=local_stiffs.dtype)
    mass_data = jnp.zeros(total_entries, dtype=local_masses.dtype)
    
    def body_fun(i, carry):
        rows, cols, stiff_data, mass_data = carry
        nodes = all_nodes[i]
        
        # Generate row and column indices for current element
        local_rows = jnp.repeat(nodes[:, None], 3, axis=1).ravel()
        local_cols = jnp.repeat(nodes[None, :], 3, axis=0).ravel()
        
        # Calculate start index for this element
        start_idx = i * entries_per_element
        
        # Update arrays
        rows = lax.dynamic_update_slice(rows, local_rows, (start_idx,))
        cols = lax.dynamic_update_slice(cols, local_cols, (start_idx,))
        stiff_data = lax.dynamic_update_slice(stiff_data, local_stiffs[i].ravel(), (start_idx,))
        mass_data = lax.dynamic_update_slice(mass_data, local_masses[i].ravel(), (start_idx,))
        
        return rows, cols, stiff_data, mass_data
    
    # Update arrays using fori_loop
    rows, cols, stiff_data, mass_data = lax.fori_loop(
        0, n_elements, body_fun, (rows, cols, stiff_data, mass_data))
    
    # Create indices array for BCOO format
    indices = jnp.stack([rows, cols], axis=1)
    
    # Create sparse matrices
    Global_stiff = BCOO((stiff_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    Global_mass = BCOO((mass_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    
    return Global_stiff, Global_mass
