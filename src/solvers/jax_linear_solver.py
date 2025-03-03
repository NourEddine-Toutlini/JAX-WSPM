import jax.numpy as jnp
import jax.scipy.sparse.linalg as spla
from jax import jit, lax, vmap
from typing import Tuple, Optional, Callable
from functools import partial
from jax.experimental.sparse import BCOO

class Preconditioner:
    """Collection of preconditioners for sparse linear systems working with BCOO format."""
    
    @staticmethod
    def create_jacobi(A: BCOO) -> Callable:
        """Create a Jacobi preconditioner using JIT-compatible operations."""
        diagonals = []
        nrows = A.shape[0]

        # Extract diagonal using lax.scan
        def scan_fn(carry, ind_data):
            i, j = ind_data[0]  # indices
            val = ind_data[1]   # value
            diag = carry.at[i].add(jnp.where(i == j, val, 0.0))
            return diag, None

        diagonals = lax.scan(
            scan_fn, 
            init=jnp.zeros(nrows), 
            xs=(A.indices, A.data)
        )[0]

        diag_inv = 1.0 / jnp.where(diagonals != 0, diagonals, 1.0)

        @jit
        def preconditioner(x):
            return diag_inv * x

        return preconditioner

    @staticmethod
    def create_block_jacobi(A: BCOO, block_size: int = 2) -> Callable:
        """Create Block Jacobi using static shapes for JIT compatibility."""
        n = A.shape[0]
        n_blocks = n // block_size + (1 if n % block_size else 0)
        block_starts = jnp.arange(0, n, block_size)

        blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = min(start + block_size, n)
            size = end - start

            # Initialize block matrix
            block = jnp.zeros((block_size, block_size))

            # Fill block using mask operations
            mask = ((A.indices[:, 0] >= start) & (A.indices[:, 0] < end) & 
                   (A.indices[:, 1] >= start) & (A.indices[:, 1] < end))

            local_rows = jnp.where(mask, A.indices[:, 0] - start, 0)
            local_cols = jnp.where(mask, A.indices[:, 1] - start, 0)

            for idx in range(A.data.shape[0]):
                block = block.at[local_rows[idx], local_cols[idx]].add(
                    jnp.where(mask[idx], A.data[idx], 0.0)
                )

            # Add small diagonal term for stability
            block = block + jnp.eye(block_size) * 1e-12
            # Compute inverse of this block
            if size < block_size:
                block = block[:size, :size]
            blocks.append(jnp.linalg.inv(block))

        @jit
        def preconditioner(x):
            result = jnp.zeros_like(x)
            for i in range(n_blocks):
                start = i * block_size
                end = min(start + block_size, n)
                block = blocks[i]
                x_block = x[start:end]
                if block.shape[0] < block_size:
                    result = result.at[start:end].set(block @ x_block)
                else:
                    result = result.at[start:end].set(block @ x_block[:block.shape[0]])
            return result

        return preconditioner

    @staticmethod
    def create_ssor(A: BCOO, omega: float = 1.0) -> Callable:
        """Create a Symmetric Successive Over-Relaxation (SSOR) preconditioner for BCOO matrix."""
        n = A.shape[0]

        # Pre-compute diagonal and off-diagonal parts using JAX-friendly operations
        def process_entry(carry, index_val):
            diag, L_data, L_idx, U_data, U_idx = carry
            idx, val = index_val
            row, col = idx[0], idx[1]

            # Use where instead of direct boolean comparisons
            is_diag = jnp.where(row == col, True, False)
            is_lower = jnp.where(row > col, True, False)

            # Update diagonal
            diag = diag.at[row].add(jnp.where(is_diag, val, 0.0))

            # Update L and U parts
            L_data = L_data.at[index_val[0]].set(jnp.where(is_lower, val, 0.0))
            L_idx = L_idx.at[index_val[0]].set(jnp.where(is_lower, 1.0, 0.0))
            U_data = U_data.at[index_val[0]].set(jnp.where(~is_lower & ~is_diag, val, 0.0))
            U_idx = U_idx.at[index_val[0]].set(jnp.where(~is_lower & ~is_diag, 1.0, 0.0))

            return (diag, L_data, L_idx, U_data, U_idx), None

        # Initialize carriers
        initial_diag = jnp.zeros(n)
        initial_L_data = jnp.zeros(A.nse)
        initial_L_idx = jnp.zeros(A.nse)
        initial_U_data = jnp.zeros(A.nse)
        initial_U_idx = jnp.zeros(A.nse)

        # Process all entries
        (diag_vals, L_data, L_idx, U_data, U_idx), _ = lax.scan(
            process_entry,
            (initial_diag, initial_L_data, initial_L_idx, initial_U_data, initial_U_idx),
            (A.indices, A.data)
        )

        # Create D/omega
        D_omega = diag_vals / omega

        @jit
        def preconditioner(x):
            # Forward sweep
            y = jnp.zeros_like(x)
            def forward_sweep(i, val):
                y, _ = val
                y_i = x[i]

                # Apply L using vectorized operations
                mask = L_idx == 1.0
                contrib = jnp.where(mask & (A.indices[:, 0] == i),
                                  L_data * y[A.indices[:, 1]],
                                  0.0)
                y_i = y_i - jnp.sum(contrib)

                y = y.at[i].set(y_i / D_omega[i])
                return (y, None)

            y, _ = lax.fori_loop(0, n, forward_sweep, (y, None))

            # Backward sweep
            z = jnp.zeros_like(x)
            def backward_sweep(i, val):
                z, _ = val
                i = n - 1 - i  # Reverse index
                z_i = diag_vals[i] * y[i]

                # Apply U using vectorized operations
                mask = U_idx == 1.0
                contrib = jnp.where(mask & (A.indices[:, 1] == i),
                                  U_data * z[A.indices[:, 0]],
                                  0.0)
                z_i = z_i - jnp.sum(contrib)

                z = z.at[i].set(z_i / D_omega[i])
                return (z, None)

            z, _ = lax.fori_loop(0, n, backward_sweep, (z, None))
            return z

        return preconditioner

    @staticmethod
    def create_ilu0(A: BCOO) -> Callable:
        """Create an ILU(0) preconditioner for BCOO matrix using JAX-friendly operations."""
        n = A.shape[0]

        # Sort indices and data
        sorted_idx = jnp.argsort(A.indices[:, 0] * n + A.indices[:, 1])
        sorted_data = A.data[sorted_idx]
        sorted_indices = A.indices[sorted_idx]

        # Initialize matrices using JAX-friendly operations
        def process_entry(carry, index_val):
            L_data, U_data, diag = carry
            idx, val = index_val
            row, col = idx[0], idx[1]

            is_diag = jnp.where(row == col, True, False)
            is_lower = jnp.where(row > col, True, False)

            # Update matrices using where operations
            diag = diag.at[row].add(jnp.where(is_diag, val, 0.0))
            L_data = L_data.at[index_val[0]].set(
                jnp.where(is_lower, val / jnp.where(diag[col] == 0, 1.0, diag[col]), 0.0)
            )
            U_data = U_data.at[index_val[0]].set(
                jnp.where(~is_lower & ~is_diag, val, 0.0)
            )

            return (L_data, U_data, diag), None

        # Initialize carriers
        initial_L_data = jnp.zeros(A.nse)
        initial_U_data = jnp.zeros(A.nse)
        initial_diag = jnp.zeros(n)

        # Process entries
        (L_data, U_data, diag_data), _ = lax.scan(
            process_entry,
            (initial_L_data, initial_U_data, initial_diag),
            (sorted_indices, sorted_data)
        )

        @jit
        def preconditioner(x):
            # Forward substitution
            y = jnp.zeros_like(x)
            def forward_sweep(i, val):
                y, _ = val
                y_i = x[i]

                # Vectorized L application
                mask = (sorted_indices[:, 0] == i) & (sorted_indices[:, 1] < i)
                contrib = jnp.where(mask, L_data * y[sorted_indices[:, 1]], 0.0)
                y_i = y_i - jnp.sum(contrib)

                y = y.at[i].set(y_i)
                return (y, None)

            y, _ = lax.fori_loop(0, n, forward_sweep, (y, None))

            # Backward substitution
            z = jnp.zeros_like(y)
            def backward_sweep(i, val):
                z, _ = val
                i = n - 1 - i  # Reverse index
                z_i = y[i]

                # Vectorized U application
                mask = (sorted_indices[:, 0] == i) & (sorted_indices[:, 1] > i)
                contrib = jnp.where(mask, U_data * z[sorted_indices[:, 1]], 0.0)
                z_i = z_i - jnp.sum(contrib)

                z = z.at[i].set(z_i / jnp.where(diag_data[i] == 0, 1.0, diag_data[i]))
                return (z, None)

            z, _ = lax.fori_loop(0, n, backward_sweep, (z, None))
            return z

        return preconditioner


class JAXSparseSolver:
    """Sparse linear solvers using JAX's implementations with preconditioner support."""
    
    def __init__(self, A: BCOO):
        """Initialize solver with system matrix."""
        self.A = A
        
    def solve_cg(self, b: jnp.ndarray,
                x0: Optional[jnp.ndarray] = None,
                tol: float = 1e-5,
                atol: float = 0.0,
                maxiter: Optional[int] = None,
                precond_type: Optional[str] = None,
                **precond_kwargs) -> Tuple[jnp.ndarray, bool]:
        """Solve using Conjugate Gradient method."""
        M = None
        if precond_type == 'jacobi':
            M = Preconditioner.create_jacobi(self.A)
        elif precond_type == 'block_jacobi':
            block_size = precond_kwargs.get('block_size', 2)
            M = Preconditioner.create_block_jacobi(self.A, block_size)
        elif precond_type == 'ssor':
            omega = precond_kwargs.get('omega', 1.0)
            M = Preconditioner.create_ssor(self.A, omega)
        elif precond_type == 'ilu':
            M = Preconditioner.create_ilu0(self.A)
        
        return spla.cg(self.A, b, x0=x0, tol=tol, atol=atol,
                      maxiter=maxiter, M=M)
    
    def solve_bicgstab(self, b: jnp.ndarray,
                      x0: Optional[jnp.ndarray] = None,
                      tol: float = 1e-5,
                      atol: float = 0.0,
                      maxiter: Optional[int] = None,
                      precond_type: Optional[str] = None,
                      **precond_kwargs) -> Tuple[jnp.ndarray, bool]:
        """Solve using BiCGSTAB method."""
        M = None
        if precond_type == 'jacobi':
            M = Preconditioner.create_jacobi(self.A)
        elif precond_type == 'block_jacobi':
            block_size = precond_kwargs.get('block_size', 2)
            M = Preconditioner.create_block_jacobi(self.A, block_size)
        elif precond_type == 'ssor':
            omega = precond_kwargs.get('omega', 1.0)
            M = Preconditioner.create_ssor(self.A, omega)
        elif precond_type == 'ilu':
            M = Preconditioner.create_ilu0(self.A)
        
        return spla.bicgstab(self.A, b, x0=x0, tol=tol, atol=atol,
                           maxiter=maxiter, M=M)
    
    def solve_gmres(self, b: jnp.ndarray,
                   x0: Optional[jnp.ndarray] = None,
                   tol: float = 1e-5,
                   atol: float = 0.0,
                   restart: int = 20,
                   maxiter: Optional[int] = None,
                   precond_type: Optional[str] = None,
                   **precond_kwargs) -> Tuple[jnp.ndarray, bool]:
        """Solve using GMRES method."""
        M = None
        if precond_type == 'jacobi':
            M = Preconditioner.create_jacobi(self.A)
        elif precond_type == 'block_jacobi':
            block_size = precond_kwargs.get('block_size', 2)
            M = Preconditioner.create_block_jacobi(self.A, block_size)
        elif precond_type == 'ssor':
            omega = precond_kwargs.get('omega', 1.0)
            M = Preconditioner.create_ssor(self.A, omega)
        elif precond_type == 'ilu':
            M = Preconditioner.create_ilu0(self.A)
        
        return spla.gmres(self.A, b, x0=x0, tol=tol, atol=atol,
                         restart=restart, maxiter=maxiter, M=M)