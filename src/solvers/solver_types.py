import jax.numpy as jnp
from typing import Tuple, Optional
from jax.experimental.sparse import BCOO
from .jax_linear_solver import JAXSparseSolver
from config.settings import SolverParameters

def solve_system(
    matrix: BCOO,
    rhs: jnp.ndarray,
    x0: jnp.ndarray,
    solver_config: SolverParameters
) -> Tuple[jnp.ndarray, bool]:
    """
    Solve a linear system using the configured solver and preconditioner.
    
    Args:
        matrix: System matrix in BCOO sparse format
        rhs: Right-hand side vector
        x0: Initial guess for the solution
        solver_config: Solver configuration from settings.SolverParameters
    
    Returns:
        Tuple of (solution_vector, convergence_flag)
    """
    # Initialize solver
    solver = JAXSparseSolver(matrix)
    
    # Select solver based on configuration
    if solver_config.solver_type == 'gmres':
        solution, convergence = solver.solve_gmres(
            rhs,
            x0=x0,
            tol=solver_config.tol,
            restart=solver_config.restart,
            maxiter=solver_config.maxiter,
            precond_type=solver_config.precond_type if solver_config.precond_type != 'none' else None,
            **solver_config.precond_params
        )
    
    elif solver_config.solver_type == 'bicgstab':
        solution, convergence = solver.solve_bicgstab(
            rhs,
            x0=x0,
            tol=solver_config.tol,
            maxiter=solver_config.maxiter,
            precond_type=solver_config.precond_type if solver_config.precond_type != 'none' else None,
            **solver_config.precond_params
        )
    
    elif solver_config.solver_type == 'cg':
        solution, convergence = solver.solve_cg(
            rhs,
            x0=x0,
            tol=solver_config.tol,
            maxiter=solver_config.maxiter,
            precond_type=solver_config.precond_type if solver_config.precond_type != 'none' else None,
            **solver_config.precond_params
        )
    
    elif solver_config.solver_type == 'direct':
        matrix_dense = matrix.todense()
        solution = jnp.linalg.solve(matrix_dense, rhs)
        convergence = True
    
    else:
        raise ValueError(f"Unsupported solver type: {solver_config.solver_type}")
    
    return solution, convergence