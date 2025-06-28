"""Nitrogen transport solver for multi-species reactive transport modeling - SPARSE VERSION."""

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO
from jax import debug


from ..models.van_genuchten import van_genuchten_model
from ..models.boundary import (
    extract_boundary_nodes,          
    get_boundary_condition, 
    apply_neumann_bcs, 
    apply_freedrainage_water_bcs,
    apply_cauchy_bcs_sparse,  # ← Now using sparse version
    apply_freedrainage_nitrogen_bcs_sparse  # ← Now using sparse version
)
from ..numerics.gauss import gauss_triangle
from ..numerics.assembly_re_v2 import assemble_global_matrices_sparse_re
# Import the NEW sparse nitrogen assembly functions
from ..numerics.assembly_nitrogen import (  
    assemble_nitrogen_species1_sparse,
    assemble_nitrogen_species2_sparse, 
    assemble_nitrogen_species3_sparse
)
from ..models.transport_utilities import calculate_fluxes_and_dispersion_FE
from ..mesh.loader import load_and_validate_mesh
from .solver_types import solve_system

# At the top of your nitrogen_solver.py file, outside the NitrogenSolver class
@jit
def lump_sparse_mass_matrix(mass_matrix: BCOO) -> BCOO:
    """
    Convert a consistent sparse mass matrix to a lumped (diagonal) mass matrix.
    
    This performs row-sum lumping: M_lumped[i,i] = sum(M_consistent[i,:])
    This is a pure function that doesn't need class instance data.
    
    Args:
        mass_matrix: Consistent mass matrix in BCOO sparse format
        
    Returns:
        Lumped mass matrix in BCOO sparse format (diagonal matrix)
    """
    # Extract sparse matrix components
    indices = mass_matrix.indices  # Shape: (nnz, 2) 
    values = mass_matrix.data      # Shape: (nnz,)
    n_nodes = mass_matrix.shape[0] # Number of nodes
    
    # Get row indices for each non-zero entry
    row_indices = indices[:, 0]
    
    # Perform row-wise summation using JAX's scatter-add
    row_sums = jnp.zeros(n_nodes, dtype=values.dtype)
    row_sums = row_sums.at[row_indices].add(values)
    
    # Create diagonal indices for lumped matrix
    diagonal_indices = jnp.stack([jnp.arange(n_nodes), jnp.arange(n_nodes)], axis=1)
    
    # Return lumped mass matrix as diagonal sparse matrix
    return BCOO((row_sums, diagonal_indices), shape=mass_matrix.shape)

class NitrogenSolver:
    """Solver for coupled water flow and nitrogen transport equations using sparse matrices."""
    
    def __init__(self, config: 'SimulationConfig'):
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        self.points, self.triangles, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            self.config.test_case
        )
        
        self.boundary_nodes = extract_boundary_nodes(self.points, self.config.test_case)
        
        self.quad_points, self.weights = gauss_triangle(3)
        self.ksi_1d = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
        self.w_1d = jnp.array([1.0, 1.0])
        
        self.bc_info = get_boundary_condition(self.config.test_case)
        
        self.nnt = len(self.points)
        self.initialize_problem()
    
    def initialize_problem(self):
        # Initial conditions remain the same
        self.pressure_head = -1.0 * jnp.ones(self.nnt)
        self.NH4_concentration = jnp.ones(self.nnt) * self.config.nitrogen.c_init_NH4
        self.NO2_concentration = jnp.ones(self.nnt) * self.config.nitrogen.c_init_NO2
        self.NO3_concentration = jnp.ones(self.nnt) * self.config.nitrogen.c_init_NO3

    @partial(jit, static_argnums=(0,))
    def solve_richards_step(self, pressure_head_n: jnp.ndarray, 
                           theta_n: jnp.ndarray, 
                           dt: float) -> Tuple[jnp.ndarray, float, int]:
        # Richards equation solver remains unchanged since it already uses sparse matrices
        def body_fun(carry):
            pressure_head_m, err, iter_count = carry

            Capacity_m, Konduc_m, theta_m = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_m, self.config.van_genuchten.to_array())

            Global_matrix, Global_source = assemble_global_matrices_sparse_re(
                self.triangles, self.nnt, self.points, theta_m, theta_n,
                pressure_head_m, Konduc_m, Capacity_m, self.quad_points, self.weights, dt
            )

            if self.bc_info['type'] == 'coupled':
                neumann_value = self.bc_info['water_flux']['neumann_flux']
                Global_source = apply_neumann_bcs(
                    Global_source, neumann_value, self.boundary_nodes.neumann,
                    self.points, self.ksi_1d, self.w_1d
                )
                # ADD THIS: Apply free drainage BC (water outflow at bottom)
                if self.boundary_nodes.freedrainage.size > 0:  # Check if freedrainage nodes exist
                    Global_source = apply_freedrainage_water_bcs(
                        Global_source, Konduc_m, self.boundary_nodes.freedrainage,
                        self.points, self.ksi_1d, self.w_1d
                    )

            
            pressure_head_new, convergence = solve_system(
                matrix=Global_matrix, rhs=Global_source, x0=pressure_head_m,
                solver_config=self.config.solver
            )

            err = jnp.linalg.norm(pressure_head_new - pressure_head_m)
            return pressure_head_new, err, iter_count + 1

        def cond_fun(carry):
            _, err, iter_count = carry
            return jnp.logical_and(err >= 1e-6, iter_count < 100)

        initial_carry = (pressure_head_n, jnp.inf, 0)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        pressure_head_new, err, iter_count = final_carry
        
        # debug.print("Iteration: {}, Error: {}, dt: {}", iter_count, err, dt)


        return pressure_head_new, err, iter_count

    @partial(jit, static_argnums=(0,))
    
    
    def solve_nitrogen_transport_step(self, pressure_head: jnp.ndarray, theta: jnp.ndarray,
                                    water_flux_x: jnp.ndarray, water_flux_z: jnp.ndarray,
                                    D_xx: jnp.ndarray, D_xz: jnp.ndarray, D_zz: jnp.ndarray,
                                    NH4_n: jnp.ndarray, NO2_n: jnp.ndarray, NO3_n: jnp.ndarray,
                                    dt: float, Konduc: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        nitrogen_params = self.config.nitrogen.to_array()
        
        
        # NH4+ transport - NOW USING SPARSE ASSEMBLY
        Global_stiff_NH4, Global_mass_NH4, Global_source_NH4 = assemble_nitrogen_species1_sparse(
            self.triangles, self.nnt, self.points, theta, water_flux_x, water_flux_z,
            D_xx, D_xz, D_zz, nitrogen_params, self.quad_points, self.weights
        )
        
        Global_mass_NH4 = lump_sparse_mass_matrix(Global_mass_NH4)

        
        # Time integration: (M/dt + K){C^{n+1}} = (M/dt){C^n} + F
        # For sparse matrices, we need to handle the mass matrix multiplication differently
        
        Global_source_NH4 = self.sparse_mass_multiply(Global_mass_NH4, NH4_n, dt) + Global_source_NH4
        
        # jax.clear_caches()
        # Global_source_NH4 = (1/dt) * (Global_mass_NH4 @ NH4_n)
        

        
        Global_matrix_NH4, Global_source_NH4 = self.apply_nitrogen_boundary_conditions_sparse(
            Global_stiff_NH4, Global_source_NH4, Global_mass_NH4,
            self.config.nitrogen.c_inlet_NH4, dt, Konduc, theta
        )
        
        NH4_new, _ = solve_system(
            matrix=Global_matrix_NH4, rhs=Global_source_NH4, x0=NH4_n,
            solver_config=self.config.solver
        )
        
        # NO2- transport - NOW USING SPARSE ASSEMBLY
        Global_stiff_NO2, Global_mass_NO2, Global_source_NO2 = assemble_nitrogen_species2_sparse(
            self.triangles, self.nnt, self.points, theta, water_flux_x, water_flux_z,
            D_xx, D_xz, D_zz, nitrogen_params, NH4_new, self.quad_points, self.weights
        )
        
        Global_mass_NO2 = lump_sparse_mass_matrix(Global_mass_NO2)
        
        Global_source_NO2 = self.sparse_mass_multiply(Global_mass_NO2, NO2_n, dt) + Global_source_NO2
        
        Global_matrix_NO2, Global_source_NO2 = self.apply_nitrogen_boundary_conditions_sparse(
            Global_stiff_NO2, Global_source_NO2, Global_mass_NO2,
            self.config.nitrogen.c_inlet_NO2, dt, Konduc, theta
        )
        
        
        NO2_new, _ = solve_system(
            matrix=Global_matrix_NO2, rhs=Global_source_NO2, x0=NO2_n,
            solver_config=self.config.solver
        )
        
        
        # NO3- transport - NOW USING SPARSE ASSEMBLY
        Global_stiff_NO3, Global_mass_NO3, Global_source_NO3 = assemble_nitrogen_species3_sparse(
            self.triangles, self.nnt, self.points, theta, water_flux_x, water_flux_z,
            D_xx, D_xz, D_zz, nitrogen_params, NO2_new, self.quad_points, self.weights
        )
        
        Global_mass_NO3 = lump_sparse_mass_matrix(Global_mass_NO3)
        
        Global_source_NO3 = self.sparse_mass_multiply(Global_mass_NO3, NO3_n, dt) + Global_source_NO3
        
        Global_matrix_NO3, Global_source_NO3 = self.apply_nitrogen_boundary_conditions_sparse(
            Global_stiff_NO3, Global_source_NO3, Global_mass_NO3,
            self.config.nitrogen.c_inlet_NO3, dt, Konduc, theta
        )

        NO3_new, _ = solve_system(
            matrix=Global_matrix_NO3, rhs=Global_source_NO3, x0=NO3_n,
            solver_config=self.config.solver
        )

        
        return NH4_new, NO2_new, NO3_new
    
    
    
    
    @partial(jit, static_argnums=(0,))
    def sparse_mass_multiply(self, mass_matrix: BCOO, concentration: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Efficiently compute (1/dt) * M @ C for sparse mass matrix.
        This replaces the dense operation (1/dt) * (Global_mass @ concentration).
        """
        # Use JAX's sparse matrix-vector multiplication
        mass_times_conc = mass_matrix @ concentration
        return (1.0 / dt) * mass_times_conc

    @partial(jit, static_argnums=(0,))
    def apply_nitrogen_boundary_conditions_sparse(self, Global_stiff: BCOO, Global_source: jnp.ndarray,
                                                 Global_mass: BCOO, inlet_concentration: float,
                                                 dt: float, Konduc: jnp.ndarray, 
                                                 theta: jnp.ndarray) -> Tuple[BCOO, jnp.ndarray]:
        """
        Apply boundary conditions to sparse matrices.
        This is the key function that needed to be updated for sparse compatibility.
        """
        
        # Apply Cauchy boundary conditions if present (inlet boundary)
        if self.boundary_nodes.cauchy.size > 0:
            Global_stiff, Global_source = apply_cauchy_bcs_sparse(
                Global_stiff, Global_source, self.bc_info['solute_flux']['cauchy_flux'],
                inlet_concentration, self.boundary_nodes.cauchy,
                self.ksi_1d, self.w_1d, self.points
            )
        
        # Apply free drainage boundary conditions if present (outlet boundary)
        if self.boundary_nodes.freedrainage.size > 0:
            Global_stiff, Global_source = apply_freedrainage_nitrogen_bcs_sparse(
                Global_stiff, Global_source, Konduc, theta,
                self.boundary_nodes.freedrainage, self.points, self.ksi_1d, self.w_1d
            )
        
        # Combine mass and stiffness matrices: Global_matrix = (1/dt) * M + K
        # For sparse matrices, we need to use sparse arithmetic
        Global_matrix = self.sparse_matrix_add(Global_mass, Global_stiff, dt)
        
        return Global_matrix, Global_source
    
    
    
    
    
    @partial(jit, static_argnums=(0,))
    def sparse_matrix_add(self, mass_matrix: BCOO, stiff_matrix: BCOO, dt: float) -> BCOO:
        """
        Efficiently compute (1/dt) * M + K for sparse matrices.
        This replaces the dense operation Global_matrix = (1/dt) * Global_mass + Global_stiff.
        """
        # Scale the mass matrix by 1/dt
        scaled_mass_data = (1.0 / dt) * mass_matrix.data
        scaled_mass = BCOO((scaled_mass_data, mass_matrix.indices), shape=mass_matrix.shape)
        
        # Combine the scaled mass matrix with the stiffness matrix
        # Concatenate data and indices, then sum duplicates
        combined_data = jnp.concatenate([scaled_mass.data, stiff_matrix.data])
        combined_indices = jnp.concatenate([scaled_mass.indices, stiff_matrix.indices])
        
        # Create combined matrix and handle overlapping entries
        combined_matrix = BCOO((combined_data, combined_indices), shape=mass_matrix.shape)
        total_nse = scaled_mass.nse + stiff_matrix.nse
        
        # Sum duplicate entries to get the final system matrix
        return combined_matrix.sum_duplicates(nse=total_nse)

    # The rest of the solver methods remain unchanged
    def solve(self) -> Dict[str, Any]:
        simulation_start = time.perf_counter()
        
        all_pressure = []
        all_theta = []
        all_NH4 = []
        all_NO2 = []
        all_NO3 = []
        all_times = []
        all_iterations = []
        all_errors = []
        all_dt = []
        
        pressure_head_n = self.pressure_head
        NH4_n = self.NH4_concentration
        NO2_n = self.NO2_concentration
        NO3_n = self.NO3_concentration
        current_time = 0.0
        dt = self.config.time.dt_init
        
        pbar = tqdm(total=float(self.config.time.Tmax), desc='Nitrogen Simulation Progress', unit='time units')
        
        while current_time < self.config.time.Tmax:
            _, _, theta_n = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_n, self.config.van_genuchten.to_array())
            
            pressure_head, error, iter_count = self.solve_richards_step(pressure_head_n, theta_n, dt)
            
            fluxes_disp = calculate_fluxes_and_dispersion_FE(
                self.points, self.triangles, pressure_head, self.config.van_genuchten.to_array(),
                self.config.nitrogen.DL, self.config.nitrogen.DT, self.config.nitrogen.Dm
            )
             
            water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, theta = fluxes_disp
            
            
            _, Konduc, _ = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head, self.config.van_genuchten.to_array())
            

            NH4_new, NO2_new, NO3_new = self.solve_nitrogen_transport_step(
                pressure_head, theta, water_flux_x, water_flux_z,
                D_xx, D_xz, D_zz, NH4_n, NO2_n, NO3_n, dt, Konduc)

            dt_new = self.adapt_time_step(dt, iter_count)
            
            all_pressure.append(pressure_head)
            all_theta.append(theta)
            all_NH4.append(NH4_new)
            all_NO2.append(NO2_new)
            all_NO3.append(NO3_new)
            all_times.append(current_time)
            all_iterations.append(iter_count)
            all_errors.append(error)
            all_dt.append(dt)
            
            pressure_head_n = pressure_head
            NH4_n = NH4_new
            NO2_n = NO2_new
            NO3_n = NO3_new
            current_time += dt
            dt = dt_new
            
            pbar.update(float(dt)) 
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {float(dt):.2e}, '  # Also convert dt here
                
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        
        

        return {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta), 
            'NH4_concentration': jnp.array(all_NH4),
            'NO2_concentration': jnp.array(all_NO2),
            'NO3_concentration': jnp.array(all_NO3),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'points': self.points,           # ← ADD THIS
            'triangles': self.triangles,     # ← ADD THIS
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config            # ← ADD THIS TOO (often expected)
        }

    @partial(jit, static_argnums=(0,))
    def adapt_time_step(self, dt: float, iter_count: int) -> float:
        return lax.cond(
            iter_count <= 5,
            lambda _: jnp.minimum(1.2 * dt, 0.01),
            lambda _: lax.cond(
                iter_count <= 15,
                lambda _: dt,
                lambda _: jnp.maximum(0.5 * dt, 1e-6),
                None
            ),
            None
        )
    
    