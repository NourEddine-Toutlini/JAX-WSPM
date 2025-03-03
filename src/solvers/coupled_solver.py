# src/solvers/coupled_solver.py
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time, psutil
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO
# from jax.scipy.sparse.linalg import gmres, bicgstab, cg
from .solver_types import solve_system


from ..models.van_genuchten import van_genuchten_model

from ..models.boundary import get_boundary_condition, extract_boundary_nodes, apply_neumann_bcs, apply_cauchy_bcs,                                         shape_functions, apply_cauchy_bcs_sparse

from ..models.transport_utilities import calculate_fluxes_and_dispersion, calculate_peclet_courant
from ..numerics.gauss import gauss_triangle
from ..numerics.assembly_re_v2 import assemble_global_matrices_sparse_re
from ..numerics.assembly_solute import assemble_global_matrices_solute
from ..mesh.loader import load_and_validate_mesh
from .jax_linear_solver import JAXSparseSolver


def track_memory(location):
    process = psutil.Process()
    print(f"Memory usage at {location}: {process.memory_info().rss / 1024 / 1024} MB")
    
    

class CoupledSolver:
    """Solver for coupled water flow and solute transport equations."""
    
    def __init__(self, config: 'SimulationConfig'):
        """Initialize the solver with configuration."""
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        self.points, self.triangles, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            self.config.test_case  # Add this parameter
        )
        
        # Get boundary nodes
        self.boundary_nodes = extract_boundary_nodes(self.points, self.config.test_case)  
        # Initialize numerical integration
        self.quad_points, self.weights = gauss_triangle(3)
        self.ksi_1d = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
        self.w_1d = jnp.array([1.0, 1.0])
        
        # Get boundary condition function
        self.bc_info = get_boundary_condition(self.config.test_case)
                
        # Initialize solver parameters
        self.nnt = len(self.points)
        self.initialize_problem()
    
    def initialize_problem(self):
        """Initialize the problem variables."""
        # Initial pressure head
        self.pressure_head = -1.3 * jnp.ones(self.nnt)
        
        # Initial solute concentration
        self.solute = jnp.ones(self.nnt) * self.config.solute.c_init
        

    @partial(jit, static_argnums=(0,))
    def solve_richards_step(self, pressure_head_n: jnp.ndarray,
                          dt: float) -> Tuple[jnp.ndarray, float, int]:
        """Solve one time step of Richards equation."""
        def body_fun(carry):
            pressure_head_m, pressure_head_n, err, iter_count = carry
            
            # Get soil properties
            Capacity_m, Konduc_m, thetan_m = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_m, self.config.van_genuchten.to_array())
                
            _, _, thetan_0 = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_n, self.config.van_genuchten.to_array())
            
            # Assemble matrices
            # track_memory("Before Richards assembly")
            Global_stiff, Global_mass, Global_source = assemble_global_matrices_sparse_re(
                self.triangles, self.nnt, self.points, thetan_m, thetan_m,
                pressure_head_m, Konduc_m, Capacity_m, self.quad_points, self.weights
            )
            
            # track_memory("After Richards assembly")
            
            # Calculate matrix sum with explicit nse
            total_entries = self.triangles.shape[0] * 9  # 9 entries per element
            
           
            nnt = self.points.shape[0]
            mass_diag = jnp.zeros(nnt)
            mass_diag = mass_diag.at[Global_mass.indices[:, 0]].add(Global_mass.data)
            
            
            mass_indices = jnp.stack([jnp.arange(nnt), jnp.arange(nnt)], axis=1)
            Global_mass = BCOO((mass_diag, mass_indices), shape=(nnt, nnt))

            # Form system matrix for Mixed form
            Capacity_m = BCOO((Capacity_m, mass_indices), shape=(nnt, nnt))
            
            # track_memory("Befor Richards multi: Global_source = (1/dt) * (Global_mass @ (Capacity_m @ pressure_head_m)) - \
                          # (1/dt) * (Global_mass @ (thetan_m - thetan_0)) + Global_source")
            # Compute system matrix and source
            Global_source = (1/dt) * (Global_mass @ (Capacity_m @ pressure_head_m)) - \
                          (1/dt) * (Global_mass @ (thetan_m - thetan_0)) + Global_source
            
            # track_memory("after Richards multi: Global_source = (1/dt) * (Global_mass @ (Capacity_m @ pressure_head_m)) - \
                          # (1/dt) * (Global_mass @ (thetan_m - thetan_0)) + Global_source")
            jax.clear_caches()
            
            # track_memory("befor Richards Global_matrix = ((1/dt) * (Global_mass @ Capacity_m) + Global_stiff).sum_duplicates(nse=total_entries)")
            Global_matrix = ((1/dt) * (Global_mass @ Capacity_m) + Global_stiff).sum_duplicates(nse=total_entries)
            # track_memory("After Richards Global_matrix = ((1/dt) * (Global_mass @ Capacity_m) + Global_stiff).sum_duplicates(nse=total_entries)")
            jax.clear_caches()
            
            
            # for Psi form
            # Global_mass = jnp.diag(jnp.sum(Global_mass.todense(), axis=1))
            # Global_matrix = (1 / dt) * Global_mass + Global_stiff.todense()
            # Global_source = (1 / dt) * jnp.dot(Global_mass, pressure_head_n) + Global_source
            # track_memory("before richards apply BCs")
            # Apply BCs based on test case
            if self.bc_info['type'] == 'coupled':
                # Apply Neumann BC at top (specified flux)
                neumann_value = self.bc_info['water_flux']['neumann_flux']
                Global_source = apply_neumann_bcs(
                    Global_source,
                    neumann_value,
                    self.boundary_nodes.neumann,
                    self.points,
                    self.ksi_1d, 
                    self.w_1d
                )
            # track_memory("After Richards apply BCs")    
                
            # track_memory("Before Richards linear system")
            # In solve Ax = b:
            pressure_head, convergence = solve_system(
                matrix=Global_matrix,
                rhs=Global_source,
                x0=pressure_head_m,
                solver_config=self.config.solver
            )
            # track_memory("After Richards linear system")
  
            err = jnp.linalg.norm(pressure_head - pressure_head_m) / jnp.linalg.norm(pressure_head)
            
            return pressure_head, pressure_head_n, err, iter_count + 1
        #input("press enter to pass") 

        def cond_fun(carry):
            _, _, err, iter_count = carry
            return jnp.logical_and(err >= 1e-5, iter_count < 100)

        initial_carry = (pressure_head_n, pressure_head_n, jnp.inf, 0)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        pressure_head, _, err, iter_count = final_carry
        jax.clear_caches()  # Clear JIT caches
        
        return pressure_head, err, iter_count
    
    jax.clear_caches()  # Clear JIT caches

    
    
    
    
    
    
    
    @partial(jit, static_argnums=(0,))
    def solve_transport_step(self, 
                           solute_n: jnp.ndarray,
                           pressure_head: jnp.ndarray,
                           theta: jnp.ndarray,
                           theta_n: jnp.ndarray,
                           water_flux_x: jnp.ndarray,
                           water_flux_z: jnp.ndarray,
                           abs_q: float,
                           D_xx: jnp.ndarray,
                           D_xz: jnp.ndarray,
                           D_zz: jnp.ndarray,
                           dt: float) -> jnp.ndarray:
        """Solve one time step of solute transport equation."""
        # Calculate fluxes and dispersion
        
        fluxes_disp = calculate_fluxes_and_dispersion(
            self.points, self.triangles, pressure_head,
            self.config.van_genuchten.to_array(),
            self.config.solute.DL,
            self.config.solute.DT,
            self.config.solute.Dm
        )
        

        
        #water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, theta = fluxes_disp
        
        # Assemble matrices
        # track_memory("After Richards assembly")
        Global_stiff, Global_mass = assemble_global_matrices_solute(
            self.triangles, self.nnt, self.points,
            theta, theta_n, water_flux_x, water_flux_z,
            D_xx, D_xz, D_zz,
            self.quad_points, self.weights
        )
        # track_memory("After Richards assembly")
        jax.clear_caches()
        # track_memory("Befor solute multi: Global_source = (1/dt) * (Global_mass @ solute_n) ")
        Global_source = (1/dt) * (Global_mass @ solute_n)
        # track_memory("After solute multi: Global_source = (1/dt) * (Global_mass @ solute_n)")
        
        # track_memory("before solute apply BCs")
        # Apply BCs using sparse format
        Global_stiff, Global_source = apply_cauchy_bcs_sparse(
            Global_stiff, Global_source,
            self.bc_info['solute']['cauchy_flux'],
            self.bc_info['solute']['inlet_conc'],
            self.boundary_nodes.cauchy,
            self.ksi_1d, self.w_1d, self.points
        )
        # track_memory("After solute apply BCs")
        jax.clear_caches()
        
        # track_memory("Befor Solute sum: Global_matrix = ((1/dt) * Global_mass + Global_stiff)")
        # Form system matrix while keeping sparse format
        Global_matrix = ((1/dt) * Global_mass + Global_stiff)
        # track_memory("After Solute sum: Global_matrix = ((1/dt) * Global_mass + Global_stiff)")
        
        
        # track_memory("Befor  Solute linear system")
        # In solve_transport_step:
        solute, convergence = solve_system(
            matrix=Global_matrix,
            rhs=Global_source,
            x0=solute_n,
            solver_config=self.config.solver
        )
        # track_memory("After Solute linear system")
        jax.clear_caches()
        
        return solute
    
    @partial(jit, static_argnums=(0,))
    def adapt_time_step(self,
                       dt: float,
                       iter_count: int,
                       water_flux_x: jnp.ndarray,
                       water_flux_z: jnp.ndarray,
                       theta: jnp.ndarray,
                       D_xx: jnp.ndarray,
                       D_zz: jnp.ndarray) -> Tuple[float, float, float]:
        """Adapt time step based on iterations and stability criteria."""
        # Get Peclet and Courant numbers
        Pe, Cr = calculate_peclet_courant(
            self.points, self.triangles,
            water_flux_x, water_flux_z, theta,
            D_xx, D_zz, dt
        )
        
        # Adjust time step based on iteration count
        dt_richards = lax.cond(
            iter_count <= self.config.time.m_it,
            lambda _: self.config.time.lambda_amp * dt,
            lambda _: lax.cond(
                iter_count <= self.config.time.M_it,
                lambda _: dt,
                lambda _: self.config.time.lambda_red * dt,
                None
            ),
            None
        )
        
        # Adjust for stability
        #pe_cr = Pe * Cr
        #dt_stability = jnp.where(pe_cr > 2, dt * 2 / pe_cr, dt)
        #dt_stability = jnp.minimum(dt_stability, 1.5 * dt)
        
        # Choose smaller time step
        #dt_new = jnp.minimum(dt_richards, dt_stability)
        #dt_new = jnp.maximum(dt_new, 1e-6)
        dt_new = dt_richards
        return dt_new, Pe, Cr
    
    def solve(self) -> Dict[str, Any]:
        """Solve the coupled system for the full simulation time."""
        simulation_start = time.perf_counter()
        
        # Initialize storage
        all_pressure = []
        all_theta = []
        all_solute = []
        all_times = []
        all_iterations = []
        all_errors = []
        all_dt = []
        all_Pe = []
        all_Cr = []
        
        # Initialize simulation variables
        pressure_head_n = self.pressure_head
        solute_n = self.solute
        current_time = 0.0
        dt = self.config.time.dt_init
        
        # Progress bar
        pbar = tqdm(total=float(self.config.time.Tmax),
                   desc='Simulation Progress',
                   unit='time units')
        
        while current_time < self.config.time.Tmax:
            # Solve Richards equation
            pressure_head, error, iter_count = self.solve_richards_step(
                pressure_head_n, dt)
            
            # Calculate water content and fluxes
            fluxes_disp = calculate_fluxes_and_dispersion(
                self.points, self.triangles, pressure_head,
                self.config.van_genuchten.to_array(),
                self.config.solute.DL,
                self.config.solute.DT,
                self.config.solute.Dm
            )
             
            water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, theta = fluxes_disp
            
            # Get water content at previous time
            _, _, theta_n = vmap(van_genuchten_model, in_axes=(0, None))(
                pressure_head_n, self.config.van_genuchten.to_array())
            
            # Solve transport equation
            solute = self.solve_transport_step(
                solute_n, pressure_head, theta, theta_n, water_flux_x, water_flux_z, abs_q, D_xx, D_xz, D_zz, dt)
            
            # Adapt time step
            dt_new, Pe, Cr = self.adapt_time_step(
                dt, iter_count, water_flux_x, water_flux_z,
                theta, D_xx, D_zz)
            
            # Store results
            all_pressure.append(pressure_head)
            all_theta.append(theta)
            all_solute.append(solute)
            all_times.append(current_time)
            all_iterations.append(iter_count)
            all_errors.append(error)
            all_dt.append(dt_new)
            all_Pe.append(Pe)
            all_Cr.append(Cr)
            
            # Update for next time step
            pressure_head_n = pressure_head
            solute_n = solute
            current_time += float(dt)
            dt = dt_new
            
            # Update progress bar
            pbar.update(float(dt))  # Convert dt to float before updating
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {float(dt):.2e}, '  # Also convert dt here
                f'Pe: {float(Pe):.2e}, Cr: {float(Cr):.2e}'
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        
        # Prepare results
        results = {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta),
            'solute': jnp.array(all_solute),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'Pe_values': jnp.array(all_Pe),
            'Cr_values': jnp.array(all_Cr),
            'points': self.points,
            'triangles': self.triangles,
            'final_pressure': pressure_head,
            'final_theta': theta,
            'final_solute': solute,
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config
        }
        
        return results
