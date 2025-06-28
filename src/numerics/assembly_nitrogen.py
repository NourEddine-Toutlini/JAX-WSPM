"""Assembly functions for nitrogen transport equations - SPARSE VERSION."""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.experimental.sparse import BCOO
from functools import partial
from typing import Tuple

@jit
def interpolate_shape_functions(ksi: float, eta: float) -> jnp.ndarray:
    """
    Compute triangular finite element shape functions at local coordinates.
    
    These shape functions are the mathematical foundation that allows us to
    interpolate values smoothly across each triangular element. Think of them
    as blending weights that determine how much each vertex contributes to
    the value at any point inside the triangle.
    """
    return jnp.array([1 - ksi - eta, ksi, eta])

@jit 
def shape_function_derivatives() -> jnp.ndarray:
    """
    Compute derivatives of shape functions in reference coordinates.
    
    These derivatives are essential for computing gradients (like concentration 
    gradients for diffusion) and are constant for linear triangular elements.
    The first row gives d/dξ derivatives, the second row gives d/dη derivatives.
    """
    return jnp.array([[-1, 1, 0], [-1, 0, 1]])

@jit
def compute_local_matrices_nitrogen_species1(local_coordinates: jnp.ndarray,
                                            local_theta: jnp.ndarray,
                                            local_theta_n: jnp.ndarray,  # ← FIXED: Added this parameter
                                            local_water_flux_x: jnp.ndarray,
                                            local_water_flux_z: jnp.ndarray,
                                            local_dispersion_xx: jnp.ndarray,
                                            local_dispersion_xz: jnp.ndarray,
                                            local_dispersion_zz: jnp.ndarray,
                                            nitrogen_params: jnp.ndarray,
                                            xloc: jnp.ndarray,
                                            W: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute local stiffness and mass matrices for NH4+ transport (Species 1).
    
    This function embodies the heart of the finite element method for nitrogen transport.
    We're solving the advection-diffusion-reaction equation:
    
    ∂(θC₁)/∂t + ∇·(qC₁) - ∇·(θD∇C₁) = -μ₁θC₁ - ρKdC₁
    
    Where:
    - C₁ is NH4+ concentration
    - θ is water content  
    - q is water flux vector
    - D is dispersion tensor
    - μ₁ is NH4+ → NO2- reaction rate
    - ρKd represents adsorption
    
    The beauty of finite elements is that we convert this complex PDE into
    a simple matrix equation that computers can solve efficiently.
    """
    # Extract nitrogen parameters - these control the biogeochemical processes
    DL, DT, Dm, rho, Kd, mu1, mu2, mu3 = nitrogen_params
    
    def integrate_point(carry, inputs):
        """
        Integrate contributions at each Gaussian quadrature point.
        
        This is where the mathematical magic happens! We're using Gaussian
        quadrature to accurately integrate over the triangular element.
        Each quadrature point contributes to the final element matrices
        based on the physics at that specific location.
        """
        local_stiff, local_mass = carry
        x, w = inputs  # x contains (ξ, η) coordinates, w is the weight
        ksi, eta = x
        
        # Step 1: Evaluate shape functions and their derivatives
        # This tells us how to interpolate values within the element
        ref_shape_func = interpolate_shape_functions(ksi, eta)
        ref_derivative = shape_function_derivatives()
        
        # Step 2: Transform from reference element to physical element
        # This accounts for element size, shape, and orientation
        Jacob = jnp.dot(ref_derivative, local_coordinates)
        Jacob_inverse = jnp.linalg.inv(Jacob)
        Det_J = jnp.linalg.det(Jacob)
        glob_derivative = jnp.dot(Jacob_inverse, ref_derivative)
        
        # Step 3: Compute dispersion contribution → diffusion physics
        # This represents Fickian diffusion: flux = -D∇C
        # The negative signs come from integration by parts in the weak form
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
        
        # Step 4: Compute advection contribution → water carries nutrients
        # This represents convective transport: flux = qC
        stiff_adv = w * (
            -(ref_shape_func @ local_water_flux_x) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_water_flux_z) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T)
        ) * Det_J
        
        # Step 5: Compute reaction contribution → NH4+ disappears via nitrification
        # This represents the biological transformation: NH4+ → NO2-
        # μ₁ is the first-order reaction rate constant
        stiff_reaction = w * (mu1 * (ref_shape_func @ local_theta)) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Step 6: Compute mass matrix → represents storage and adsorption
        # The -θ term represents storage in the liquid phase
        # The -ρKd term represents adsorption to soil particles
        mass = w * (-(ref_shape_func @ local_theta) - rho * Kd) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Accumulate all contributions
        local_stiff += stiff_disp + stiff_adv + stiff_reaction
        local_mass += mass
        
        return (local_stiff, local_mass), None
    
    # Perform the integration over all quadrature points
    init_carry = (jnp.zeros((3, 3)), jnp.zeros((3, 3)))
    (local_stiff, local_mass), _ = lax.scan(integrate_point, init_carry, (xloc, W))
    return local_stiff, local_mass

@jit
def compute_local_matrices_nitrogen_species2(local_coordinates: jnp.ndarray,
                                            local_theta: jnp.ndarray,
                                            local_theta_n: jnp.ndarray,  # ← FIXED: Added this parameter
                                            local_water_flux_x: jnp.ndarray,
                                            local_water_flux_z: jnp.ndarray,
                                            local_dispersion_xx: jnp.ndarray,
                                            local_dispersion_xz: jnp.ndarray,
                                            local_dispersion_zz: jnp.ndarray,
                                            nitrogen_params: jnp.ndarray,
                                            local_NH4: jnp.ndarray,
                                            xloc: jnp.ndarray,
                                            W: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute local matrices for NO2- transport (Species 2).
    
    NO2- (nitrite) is the intermediate species in the nitrogen cycle.
    It's produced from NH4+ and consumed to produce NO3-:
    
    ∂(θC₂)/∂t + ∇·(qC₂) - ∇·(θD∇C₂) = μ₁θC₁ - μ₂θC₂
    
    This species has both a source term (from NH4+) and a sink term (to NO3-).
    Understanding this coupling is crucial for nitrogen fate and transport modeling.
    """
    DL, DT, Dm, rho, Kd, mu1, mu2, mu3 = nitrogen_params
    
    def integrate_point(carry, inputs):
        local_stiff, local_mass, local_source = carry
        x, w = inputs
        ksi, eta = x
        
        # Shape function evaluation (same process as Species 1)
        ref_shape_func = interpolate_shape_functions(ksi, eta)
        ref_derivative = shape_function_derivatives()
        Jacob = jnp.dot(ref_derivative, local_coordinates)
        Jacob_inverse = jnp.linalg.inv(Jacob)
        Det_J = jnp.linalg.det(Jacob)
        glob_derivative = jnp.dot(Jacob_inverse, ref_derivative)
        
        # Dispersion terms (identical physics to Species 1)
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
        
        # Advection terms (identical physics to Species 1)
        stiff_adv = w * (
            -(ref_shape_func @ local_water_flux_x) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_water_flux_z) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T)
        ) * Det_J
        
        # Reaction sink term: NO2- → NO3- (μ₂ removes NO2-)
        stiff_reaction = w * (mu2 * (ref_shape_func @ local_theta)) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Mass term (NO2- typically doesn't adsorb significantly)
        mass = w * (-(ref_shape_func @ local_theta)) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Source term from NH4+ nitrification: μ₁C₁ → produces NO2-
        # This couples Species 2 to Species 1 concentrations
        NH4_qp = ref_shape_func @ local_NH4  # Interpolate NH4+ at quadrature point
        source = -w * mu1 * (ref_shape_func @ local_theta) * NH4_qp * ref_shape_func * Det_J
        
        local_stiff += stiff_disp + stiff_adv + stiff_reaction
        local_mass += mass
        local_source += source
        
        return (local_stiff, local_mass, local_source), None
    
    init_carry = (jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros(3))
    (local_stiff, local_mass, local_source), _ = lax.scan(integrate_point, init_carry, (xloc, W))
    return local_stiff, local_mass, local_source

@jit
def compute_local_matrices_nitrogen_species3(local_coordinates: jnp.ndarray,
                                            local_theta: jnp.ndarray,
                                            local_theta_n: jnp.ndarray,  # ← FIXED: Added this parameter
                                            local_water_flux_x: jnp.ndarray,
                                            local_water_flux_z: jnp.ndarray,
                                            local_dispersion_xx: jnp.ndarray,
                                            local_dispersion_xz: jnp.ndarray,
                                            local_dispersion_zz: jnp.ndarray,
                                            nitrogen_params: jnp.ndarray,
                                            local_NO2: jnp.ndarray,
                                            xloc: jnp.ndarray,
                                            W: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute local matrices for NO3- transport (Species 3).
    
    NO3- (nitrate) is the final product of nitrification:
    
    ∂(θC₃)/∂t + ∇·(qC₃) - ∇·(θD∇C₃) = μ₂θC₂
    
    NO3- typically only has a source term (from NO2-) and no significant
    reaction sinks in aerobic conditions. This makes it very mobile in
    groundwater systems and a concern for water quality.
    """
    DL, DT, Dm, rho, Kd, mu1, mu2, mu3 = nitrogen_params
    
    def integrate_point(carry, inputs):
        local_stiff, local_mass, local_source = carry
        x, w = inputs
        ksi, eta = x
        
        # Standard finite element machinery
        ref_shape_func = interpolate_shape_functions(ksi, eta)
        ref_derivative = shape_function_derivatives()
        Jacob = jnp.dot(ref_derivative, local_coordinates)
        Jacob_inverse = jnp.linalg.inv(Jacob)
        Det_J = jnp.linalg.det(Jacob)
        glob_derivative = jnp.dot(Jacob_inverse, ref_derivative)
        
        # Dispersion terms (same physics as other species)
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
        
        # Advection terms (same physics as other species)
        stiff_adv = w * (
            -(ref_shape_func @ local_water_flux_x) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[0,:].reshape((3, 1)).T) -
            (ref_shape_func @ local_water_flux_z) * 
            (ref_shape_func.reshape((3, 1)) @ glob_derivative[1,:].reshape((3, 1)).T)
        ) * Det_J
        
        # Optional denitrification term (μ₃ could represent NO3- → N2 under anaerobic conditions)
        # In most aerobic systems, this would be zero
        stiff_reaction = w * (mu3 * (ref_shape_func @ local_theta)) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Mass term (NO3- also typically doesn't adsorb significantly)
        mass = w * (-(ref_shape_func @ local_theta)) * (
            ref_shape_func.reshape((3, 1)) @ ref_shape_func.reshape((3, 1)).T
        ) * Det_J
        
        # Source term from NO2- nitrification: μ₂C₂ → produces NO3-
        NO2_qp = ref_shape_func @ local_NO2  # Interpolate NO2- at quadrature point
        source = -w * mu2 * (ref_shape_func @ local_theta) * NO2_qp * ref_shape_func * Det_J
        
        local_stiff += stiff_disp + stiff_adv + stiff_reaction
        local_mass += mass
        local_source += source
        
        return (local_stiff, local_mass, local_source), None
    
    init_carry = (jnp.zeros((3, 3)), jnp.zeros((3, 3)), jnp.zeros(3))
    (local_stiff, local_mass, local_source), _ = lax.scan(integrate_point, init_carry, (xloc, W))
    return local_stiff, local_mass, local_source

# Now the main assembly functions that orchestrate the element-by-element computation

@partial(jit, static_argnums=(1,))
def assemble_nitrogen_species1_sparse(triangles: jnp.ndarray,
                                     nnt: int,
                                     points: jnp.ndarray,
                                     theta: jnp.ndarray,
                                     water_flux_x: jnp.ndarray,
                                     water_flux_z: jnp.ndarray,
                                     dispersion_xx: jnp.ndarray,
                                     dispersion_xz: jnp.ndarray,
                                     dispersion_zz: jnp.ndarray,
                                     nitrogen_params: jnp.ndarray,
                                     xloc: jnp.ndarray,
                                     W: jnp.ndarray) -> Tuple[BCOO, BCOO, jnp.ndarray]:
    """
    Assemble global sparse matrices for NH4+ transport (Species 1).
    
    This function demonstrates the power of the finite element method:
    we compute local contributions for each triangle, then intelligently
    combine them into global matrices that represent the entire domain.
    
    The sparse format is crucial for computational efficiency - instead of
    storing n×n dense matrices (which could be millions of entries), we
    only store the non-zero entries (typically much smaller).
    """
    
    def process_element(_, ie):
        """
        Process a single triangular element.
        
        This function extracts all the local data for one triangle and
        computes its contribution to the global system. The beauty of
        finite elements is that we can process each element independently!
        """
        nodes = triangles[ie, :3]  # Get the 3 vertex nodes for this triangle
        
        # Extract all local nodal values for this element
        local_coordinates = points[nodes]
        local_theta = theta[nodes]
        local_theta_n = theta[nodes]  # For this example, current and previous are the same
        local_water_flux_x = water_flux_x[nodes]
        local_water_flux_z = water_flux_z[nodes]
        local_dispersion_xx = dispersion_xx[nodes]
        local_dispersion_xz = dispersion_xz[nodes]
        local_dispersion_zz = dispersion_zz[nodes]
        
        # Compute the element matrices
        local_stiff, local_mass = compute_local_matrices_nitrogen_species1(
            local_coordinates, local_theta, local_theta_n,  # ← FIXED: Now passing correct number of arguments
            local_water_flux_x, local_water_flux_z,
            local_dispersion_xx, local_dispersion_xz, local_dispersion_zz,
            nitrogen_params, xloc, W
        )
        
        return local_stiff, local_mass, nodes
    
    # Process all elements simultaneously using JAX's vectorization magic
    # This is much more efficient than looping over elements one by one
    results = vmap(process_element, in_axes=(None, 0))(None, jnp.arange(triangles.shape[0]))
    local_stiffs, local_masses, all_nodes = results
    
    # Now we need to assemble these local contributions into global sparse matrices
    # This is the "assembly" phase of the finite element method
    
    n_elements = triangles.shape[0]
    entries_per_element = 9  # 3×3 local matrices give 9 entries each
    total_entries = n_elements * entries_per_element
    
    # Pre-allocate arrays for the sparse matrix storage
    # This approach avoids dynamic memory allocation, which is crucial for JAX performance
    rows = jnp.zeros(total_entries, dtype=jnp.int64)
    cols = jnp.zeros(total_entries, dtype=jnp.int64)
    stiff_data = jnp.zeros(total_entries, dtype=local_stiffs.dtype)
    mass_data = jnp.zeros(total_entries, dtype=local_masses.dtype)
    
    def body_fun(i, carry):
        """
        Add one element's contributions to the global sparse matrix storage.
        
        This function takes a 3×3 local matrix and figures out where each
        entry should go in the global n×n matrix based on the element's
        node connectivity.
        """
        rows, cols, stiff_data, mass_data = carry
        nodes = all_nodes[i]
        
        # Generate global row and column indices for this element's contributions
        # This is the "connectivity" that tells us how local degrees of freedom
        # map to global degrees of freedom
        local_rows = jnp.repeat(nodes[:, None], 3, axis=1).ravel()  # [node1,node1,node1,node2,node2,node2,node3,node3,node3]
        local_cols = jnp.repeat(nodes[None, :], 3, axis=0).ravel()  # [node1,node2,node3,node1,node2,node3,node1,node2,node3]
        
        # Calculate where to store this element's data in our pre-allocated arrays
        start_idx = i * entries_per_element
        
        # Store the contributions efficiently using JAX's dynamic update operations
        rows = lax.dynamic_update_slice(rows, local_rows, (start_idx,))
        cols = lax.dynamic_update_slice(cols, local_cols, (start_idx,))
        stiff_data = lax.dynamic_update_slice(stiff_data, local_stiffs[i].ravel(), (start_idx,))
        mass_data = lax.dynamic_update_slice(mass_data, local_masses[i].ravel(), (start_idx,))
        
        return rows, cols, stiff_data, mass_data
    
    # Process all elements' contributions efficiently
    rows, cols, stiff_data, mass_data = lax.fori_loop(
        0, n_elements, body_fun, (rows, cols, stiff_data, mass_data))
    
    # Create the final sparse matrices
    indices = jnp.stack([rows, cols], axis=1)
    
    # BCOO format stores (data, indices) pairs efficiently
    # sum_duplicates() handles overlapping contributions (when multiple elements share nodes)
    Global_stiff = BCOO((stiff_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    Global_mass = BCOO((mass_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    
    # For NH4+, no external source term (only reaction and boundary sources)
    Global_source = jnp.zeros(nnt)
    
    return Global_stiff, Global_mass, Global_source

@partial(jit, static_argnums=(1,))
def assemble_nitrogen_species2_sparse(triangles: jnp.ndarray,
                                     nnt: int,
                                     points: jnp.ndarray,
                                     theta: jnp.ndarray,
                                     water_flux_x: jnp.ndarray,
                                     water_flux_z: jnp.ndarray,
                                     dispersion_xx: jnp.ndarray,
                                     dispersion_xz: jnp.ndarray,
                                     dispersion_zz: jnp.ndarray,
                                     nitrogen_params: jnp.ndarray,
                                     NH4_concentration: jnp.ndarray,
                                     xloc: jnp.ndarray,
                                     W: jnp.ndarray) -> Tuple[BCOO, BCOO, jnp.ndarray]:
    """
    Assemble global sparse matrices for NO2- transport (Species 2).
    
    This function is nearly identical to Species 1, but includes the coupling
    to NH4+ concentrations through the source term. This demonstrates how
    reactive transport creates mathematical coupling between species.
    """
    
    def process_element(_, ie):
        nodes = triangles[ie, :3]
        local_coordinates = points[nodes]
        local_theta = theta[nodes]
        local_theta_n = theta[nodes]
        local_water_flux_x = water_flux_x[nodes]
        local_water_flux_z = water_flux_z[nodes]
        local_dispersion_xx = dispersion_xx[nodes]
        local_dispersion_xz = dispersion_xz[nodes]
        local_dispersion_zz = dispersion_zz[nodes]
        local_NH4 = NH4_concentration[nodes]  # ← This provides the coupling to Species 1
        
        local_stiff, local_mass, local_source = compute_local_matrices_nitrogen_species2(
            local_coordinates, local_theta, local_theta_n,  # ← FIXED: Correct arguments
            local_water_flux_x, local_water_flux_z,
            local_dispersion_xx, local_dispersion_xz, local_dispersion_zz,
            nitrogen_params, local_NH4, xloc, W
        )
        
        return local_stiff, local_mass, local_source, nodes
    
    # Same assembly pattern as Species 1
    results = vmap(process_element, in_axes=(None, 0))(None, jnp.arange(triangles.shape[0]))
    local_stiffs, local_masses, local_sources, all_nodes = results
    
    n_elements = triangles.shape[0]
    entries_per_element = 9
    total_entries = n_elements * entries_per_element
    
    rows = jnp.zeros(total_entries, dtype=jnp.int64)
    cols = jnp.zeros(total_entries, dtype=jnp.int64)
    stiff_data = jnp.zeros(total_entries, dtype=local_stiffs.dtype)
    mass_data = jnp.zeros(total_entries, dtype=local_masses.dtype)
    
    def body_fun(i, carry):
        rows, cols, stiff_data, mass_data = carry
        nodes = all_nodes[i]
        
        local_rows = jnp.repeat(nodes[:, None], 3, axis=1).ravel()
        local_cols = jnp.repeat(nodes[None, :], 3, axis=0).ravel()
        start_idx = i * entries_per_element
        
        rows = lax.dynamic_update_slice(rows, local_rows, (start_idx,))
        cols = lax.dynamic_update_slice(cols, local_cols, (start_idx,))
        stiff_data = lax.dynamic_update_slice(stiff_data, local_stiffs[i].ravel(), (start_idx,))
        mass_data = lax.dynamic_update_slice(mass_data, local_masses[i].ravel(), (start_idx,))
        
        return rows, cols, stiff_data, mass_data
    
    rows, cols, stiff_data, mass_data = lax.fori_loop(
        0, n_elements, body_fun, (rows, cols, stiff_data, mass_data))
    
    indices = jnp.stack([rows, cols], axis=1)
    
    Global_stiff = BCOO((stiff_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    Global_mass = BCOO((mass_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    
    # Assemble the source vector from element contributions
    Global_source = jnp.zeros(nnt)
    def source_body_fun(i, source):
        nodes = all_nodes[i]
        updates = local_sources[i]
        return source.at[nodes].add(updates)

    Global_source = lax.fori_loop(0, n_elements, source_body_fun, Global_source)
    
    return Global_stiff, Global_mass, Global_source

@partial(jit, static_argnums=(1,))
def assemble_nitrogen_species3_sparse(triangles: jnp.ndarray,
                                     nnt: int,
                                     points: jnp.ndarray,
                                     theta: jnp.ndarray,
                                     water_flux_x: jnp.ndarray,
                                     water_flux_z: jnp.ndarray,
                                     dispersion_xx: jnp.ndarray,
                                     dispersion_xz: jnp.ndarray,
                                     dispersion_zz: jnp.ndarray,
                                     nitrogen_params: jnp.ndarray,
                                     NO2_concentration: jnp.ndarray,
                                     xloc: jnp.ndarray,
                                     W: jnp.ndarray) -> Tuple[BCOO, BCOO, jnp.ndarray]:
    """
    Assemble global sparse matrices for NO3- transport (Species 3).
    
    This completes our nitrogen cycle model. NO3- receives input from NO2-
    and typically has no significant sinks in aerobic groundwater systems.
    This makes NO3- the most persistent and mobile nitrogen species.
    """
    
    def process_element(_, ie):
        nodes = triangles[ie, :3]
        local_coordinates = points[nodes]
        local_theta = theta[nodes]
        local_theta_n = theta[nodes]
        local_water_flux_x = water_flux_x[nodes]
        local_water_flux_z = water_flux_z[nodes]
        local_dispersion_xx = dispersion_xx[nodes]
        local_dispersion_xz = dispersion_xz[nodes]
        local_dispersion_zz = dispersion_zz[nodes]
        local_NO2 = NO2_concentration[nodes]  # ← Coupling to Species 2
        
        local_stiff, local_mass, local_source = compute_local_matrices_nitrogen_species3(
            local_coordinates, local_theta, local_theta_n,  # ← FIXED: Correct arguments
            local_water_flux_x, local_water_flux_z,
            local_dispersion_xx, local_dispersion_xz, local_dispersion_zz,
            nitrogen_params, local_NO2, xloc, W
        )
        
        return local_stiff, local_mass, local_source, nodes
    
    # Identical assembly pattern to the other species
    results = vmap(process_element, in_axes=(None, 0))(None, jnp.arange(triangles.shape[0]))
    local_stiffs, local_masses, local_sources, all_nodes = results
    
    n_elements = triangles.shape[0]
    entries_per_element = 9
    total_entries = n_elements * entries_per_element
    
    rows = jnp.zeros(total_entries, dtype=jnp.int64)
    cols = jnp.zeros(total_entries, dtype=jnp.int64)
    stiff_data = jnp.zeros(total_entries, dtype=local_stiffs.dtype)
    mass_data = jnp.zeros(total_entries, dtype=local_masses.dtype)
    
    def body_fun(i, carry):
        rows, cols, stiff_data, mass_data = carry
        nodes = all_nodes[i]
        
        local_rows = jnp.repeat(nodes[:, None], 3, axis=1).ravel()
        local_cols = jnp.repeat(nodes[None, :], 3, axis=0).ravel()
        start_idx = i * entries_per_element
        
        rows = lax.dynamic_update_slice(rows, local_rows, (start_idx,))
        cols = lax.dynamic_update_slice(cols, local_cols, (start_idx,))
        stiff_data = lax.dynamic_update_slice(stiff_data, local_stiffs[i].ravel(), (start_idx,))
        mass_data = lax.dynamic_update_slice(mass_data, local_masses[i].ravel(), (start_idx,))
        
        return rows, cols, stiff_data, mass_data
    
    rows, cols, stiff_data, mass_data = lax.fori_loop(
        0, n_elements, body_fun, (rows, cols, stiff_data, mass_data))
    
    indices = jnp.stack([rows, cols], axis=1)
    
    Global_stiff = BCOO((stiff_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    Global_mass = BCOO((mass_data, indices), shape=(nnt, nnt)).sum_duplicates(nse=total_entries)
    
    # Assemble source vector
    Global_source = jnp.zeros(nnt)
    def source_body_fun(i, source):
        nodes = all_nodes[i]
        updates = local_sources[i]
        return source.at[nodes].add(updates)

    Global_source = lax.fori_loop(0, n_elements, source_body_fun, Global_source)
    
    return Global_stiff, Global_mass, Global_source