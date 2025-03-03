"""
Solver implementations for the Richards equation.
"""

from .richards_solver import RichardsSolver
from .jax_linear_solver import JAXSparseSolver

__all__ = ['RichardsSolver', 'JAXSparseSolver']