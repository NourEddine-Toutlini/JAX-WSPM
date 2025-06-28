"""
Solver implementations for the Richards equation.
"""

from .richards_solver_v2 import RichardsSolver
from .jax_linear_solver import JAXSparseSolver

__all__ = ['RichardsSolver', 'JAXSparseSolver']