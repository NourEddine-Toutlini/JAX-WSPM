# src/solvers/richards_solver.py
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
import time
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm
from functools import partial
from jax.experimental.sparse import BCOO
from jax import debug

from ..models.exponential import exponential_model
from ..models.boundary import (
    extract_boundary_nodes, 
    get_boundary_condition, 
    apply_dirichlet_bcs
)
from ..numerics.gauss import gauss_triangle
from ..numerics.assembly_re import assemble_global_matrices_sparse_re
from ..mesh.loader import load_and_validate_mesh
from ..utils.exact_solutions import exact_solution
from .solver_types import solve_system

def integrate_2d_domain(f: jnp.ndarray, points: jnp.ndarray, triangles: jnp.ndarray) -> float:
    """
    Integrate a function over a 2D domain using Gaussian quadrature for triangles.
    """
    # Gauss points and weights for triangles (3-point rule)
    gauss_points = jnp.array([
        [1/6, 1/6],
        [2/3, 1/6],
        [1/6, 2/3]
    ])
    gauss_weights = jnp.array([1/3, 1/3, 1/3])
    
    total_mass = 0.0
    
    for triangle in triangles:
        # Get triangle vertices
        vertices = points[triangle]
        
        # Calculate Jacobian
        J = jnp.array([
            [vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]],
            [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]]
        ])
        det_J = jnp.abs(jnp.linalg.det(J))
        
        # Integration over reference triangle
        tri_sum = 0.0
        for i, weight in enumerate(gauss_weights):
            # Map reference point to physical triangle
            x = vertices[0, 0] + J[0, 0]*gauss_points[i, 0] + J[0, 1]*gauss_points[i, 1]
            y = vertices[0, 1] + J[1, 0]*gauss_points[i, 0] + J[1, 1]*gauss_points[i, 1]
            
            # Interpolate function value at this point
            shape_funcs = jnp.array([
                1 - gauss_points[i, 0] - gauss_points[i, 1],
                gauss_points[i, 0],
                gauss_points[i, 1]
            ])
            f_val = jnp.sum(f[triangle] * shape_funcs)
            
            tri_sum += weight * f_val
            
        total_mass += tri_sum * det_J
        
    return total_mass

# def integrate_fem_trapz(theta, points, triangles):
#     integral = 0.0
#     for tri in triangles:
#         nodes = points[tri]  
#         theta_values = theta[tri]
#         A = 0.5 * jnp.abs(jnp.linalg.det(jnp.array([
#             [nodes[0, 0], nodes[0, 1], 1],
#             [nodes[1, 0], nodes[1, 1], 1],
#             [nodes[2, 0], nodes[2, 1], 1]
#         ])))
#         integral += A * jnp.mean(theta_values)
#     return integral

class RichardsSolver_Mass:
    """Solver for the Richards equation using mixed form finite elements."""
    
    def __init__(self, config: 'SimulationConfig'):
        self.config = config
        self.setup_solver()
    
    def setup_solver(self):
        # Load and validate mesh
        self.points, self.triangles, self.mesh_info = load_and_validate_mesh(
            self.config.mesh.mesh_size,
            self.config.mesh.mesh_dir,
            self.config.test_case
        )
        
        # Get boundary nodes
        self.boundary_nodes = extract_boundary_nodes(self.points, self.config.test_case)
        
        # Initialize numerical integration
        self.quad_points, self.weights = gauss_triangle(3)
        
        # Get boundary condition function
        self.bc_info = get_boundary_condition(self.config.test_case)

        # Initialize solver parameters
        self.nnt = len(self.points)
        self.initialize_problem()
    
    def initialize_problem(self):
        """Initialize the problem variables."""
        # Initial pressure head and domain length
        self.pressure_head = -15.24 * jnp.ones(self.nnt)
        self.L = 15.24
        
        # Calculate initial boundary conditions
        if self.bc_info['type'] == 'richards_only':
            x_top = self.points[self.boundary_nodes.top][:, 0]
            eps0_top = jnp.exp(self.config.exponential.alpha0 * 
                             self.pressure_head[self.boundary_nodes.top])
            self.hupper = self.bc_info['upper_bc'](x_top, self.L, 
                                                 self.config.exponential.alpha0,
                                                 eps0_top)
    
    @partial(jit, static_argnums=(0,))
    def adapt_time_step(self, dt: float, iter_count: int) -> float:
        """Adapt time step based on iteration count."""
        time_params = self.config.time
        return lax.cond(
            iter_count <= time_params.m_it,
            lambda _: time_params.lambda_amp * dt,
            lambda _: lax.cond(
                iter_count <= time_params.M_it,
                lambda _: dt,
                lambda _: time_params.lambda_red * dt,
                None
            ),
            None
        )
    
    @partial(jit, static_argnums=(0,))
    def solve_timestep(self, pressure_head_n: jnp.ndarray,
                      dt: float) -> Tuple[jnp.ndarray, float, int]:
        """Solve one time step of Richards equation."""
        def body_fun(carry):
            pressure_head_m, pressure_head_n, err, iter_count = carry
            
            # Calculate soil properties
            Capacity_m, Konduc_m, thetan_m = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_m, self.config.exponential.to_array())
            
            _, _, thetan_0 = vmap(exponential_model, in_axes=(0, None))(
                pressure_head_n, self.config.exponential.to_array())
            
            # Assemble matrices
            Global_stiff, Global_mass, Global_source = assemble_global_matrices_sparse_re(
                self.triangles, self.nnt, self.points,
                thetan_m, thetan_m, pressure_head_m,
                Konduc_m, Capacity_m, self.quad_points, self.weights
            )
            
            # Mass matrix lumping
            nnt = self.points.shape[0]
            mass_diag = jnp.zeros(nnt)
            mass_diag = mass_diag.at[Global_mass.indices[:, 0]].add(Global_mass.data)
            mass_indices = jnp.stack([jnp.arange(nnt), jnp.arange(nnt)], axis=1)
            Global_mass = BCOO((mass_diag, mass_indices), shape=(nnt, nnt))
            
            # Form system matrix
            total_entries = self.triangles.shape[0] * 9
            Capacity_m = BCOO((Capacity_m, mass_indices), shape=(nnt, nnt))
            Global_source = (1/dt) * (Global_mass @ (Capacity_m @ pressure_head_m)) - \
                          (1/dt) * (Global_mass @ (thetan_m - thetan_0)) + Global_source
            Global_matrix = ((1/dt) * (Global_mass @ Capacity_m) + Global_stiff).sum_duplicates(nse=total_entries)
            
            # Apply boundary conditions
            matrix_dense = Global_matrix.todense()
            if self.bc_info['type'] == 'richards_only':
                matrix_dense, Global_source = apply_dirichlet_bcs(
                    matrix_dense, Global_source, self.boundary_nodes.top, self.hupper)
                matrix_dense, Global_source = apply_dirichlet_bcs(
                    matrix_dense, Global_source, self.boundary_nodes.bottom,
                    jnp.full_like(self.boundary_nodes.bottom, -15.24))
            
            Global_matrix = BCOO.fromdense(matrix_dense, nse=total_entries)
            
            # Solve system
            pressure_head, convergence = solve_system(
                matrix=Global_matrix,
                rhs=Global_source,
                x0=pressure_head_m,
                solver_config=self.config.solver
            )

            err = jnp.linalg.norm(pressure_head - pressure_head_m) / jnp.linalg.norm(pressure_head)
            return pressure_head, pressure_head_n, err, iter_count + 1

        def cond_fun(carry):
            _, _, err, iter_count = carry
            return jnp.logical_and(err >= 1e-6, iter_count < 100)

        initial_carry = (pressure_head_n, pressure_head_n, jnp.inf, 0)
        final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
        pressure_head, _, err, iter_count = final_carry
        
        return pressure_head, err, iter_count
        
    def solve(self) -> Dict[str, Any]:
        """Solve the Richards equation with mass balance calculation."""
        simulation_start = time.perf_counter()
        
        # Initialize storage
        all_pressure = []
        all_theta = []
        all_times = []
        all_iterations = []
        all_errors = []
        all_dt = []
        all_mb_num = []
        all_mb_ex = []
        
        # Get model parameters
        alpha0 = self.config.exponential.alpha0
        thetas = self.config.exponential.thetas
        thetar = self.config.exponential.thetar
        Ks = self.config.exponential.Ks
        test_number = int(self.config.test_case[-1])
        
        # Calculate parameters for mass balance
        hd = -15.24
        eps0 = jnp.exp(alpha0 * hd)
        d = alpha0 * (thetas - thetar) / Ks
        
        # Initialize variables
        pressure_head_n = self.pressure_head
        current_time = 0.0
        dt = float(self.config.time.dt_init)
        pressure_head_0 = -15.24 * jnp.ones_like(pressure_head_n)
        # Calculate initial condition
        _, _, theta_initial = vmap(exponential_model, in_axes=(0, None))(
            pressure_head_0, self.config.exponential.to_array())
        
        
        # Get initial mass
        initial_mass = integrate_2d_domain(theta_initial, self.points, self.triangles)
        # initial_mass = integrate_fem_trapz(theta_initial, self.points, self.triangles)
        print(f"initial_mass {initial_mass}")
        
        # Progress bar
        pbar = tqdm(total=float(self.config.time.Tmax),
                   desc='Simulation Progress',
                   unit='time units')
        
        # Time stepping loop
        while current_time < self.config.time.Tmax:
            # Solve current time step
            pressure_head, error, iter_count = self.solve_timestep(
                pressure_head_n, dt)
            
            # Calculate water content and mass balance
            _, _, theta = vmap(exponential_model, in_axes=(0, None))(
                pressure_head, self.config.exponential.to_array())
            
            # Calculate numerical mass change
            current_mass_num = integrate_2d_domain(theta, self.points, self.triangles)
            # current_mass_num = integrate_fem_trapz(theta_initial, self.points, self.triangles)
            mb_num = current_mass_num - initial_mass
            print(f"mb_num {current_mass_num}")
            # Calculate exact solution and mass change
            _, S_exact = exact_solution(self.points[:, 0], self.points[:, 1], 
                                      current_time, test_number, self.L,
                                      alpha0, eps0, d)

            theta_exact = (thetas - thetar) * S_exact + thetar
            print(f"error:{jnp.max(theta - theta_exact)}")
            current_mass_ex = integrate_2d_domain(theta_exact, self.points, self.triangles)
            # current_mass_ex = integrate_fem_trapz(theta_initial, self.points, self.triangles)
            mb_ex = current_mass_ex - initial_mass
            print(f"mb_ex {current_mass_ex}")
            
            # Store results
            all_pressure.append(pressure_head)
            all_theta.append(theta)
            all_times.append(current_time)
            all_iterations.append(int(iter_count))
            all_errors.append(float(error))
            all_dt.append(dt)
            all_mb_num.append(float(current_mass_num))
            all_mb_ex.append(float(current_mass_ex))
            
            # Update for next time step
            pressure_head_n = pressure_head
            current_time += dt
            dt = float(self.adapt_time_step(dt, iter_count))
            
            # Update progress bar
            pbar.update(dt)
            pbar.set_description(
                f'Time: {float(current_time):.2f}, Error: {float(error):.2e}, '
                f'Iterations: {int(iter_count)}, dt: {dt:.2e}'
            )
        
        pbar.close()
        simulation_time = time.perf_counter() - simulation_start
        
        # Convert lists to arrays
        mb_num_array = jnp.array(all_mb_num)
        mb_ex_array = jnp.array(all_mb_ex)
        
        # Calculate mass balance error
        mbe = jnp.abs(1 - mb_num_array/mb_ex_array) * 100
        print(mbe)
        
        # Prepare results
        results = {
            'pressure_head': jnp.array(all_pressure),
            'theta': jnp.array(all_theta),
            'times': jnp.array(all_times),
            'iterations': jnp.array(all_iterations),
            'errors': jnp.array(all_errors),
            'dt_values': jnp.array(all_dt),
            'points': self.points,
            'triangles': self.triangles,
            'final_theta': jnp.array(all_theta[-1]),
            'simulation_time': simulation_time,
            'mesh_info': self.mesh_info,
            'config': self.config,
            'mass_balance_numerical': mb_num_array,
            'mass_balance_exact': mb_ex_array,
            'mass_balance_error': mbe
        }
        
        return results