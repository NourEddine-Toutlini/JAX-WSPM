import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Dict, Tuple

from src.utils.exact_solutions import exact_solution


def calculate_final_errors(numerical_results: Dict, config) -> Dict:
    """Calculate L2 and L∞ errors at final time step."""
    points = numerical_results['points']
    pressure_head = numerical_results['pressure_head'][-1]  # Final time step only
    theta = numerical_results['theta'][-1]  # Final time step only
    final_time = numerical_results['times'][-1]
    
    # Get parameters
    L = 15.24
    alpha0 = config.exponential.alpha0
    thetas = config.exponential.thetas
    thetar = config.exponential.thetar
    Ks = config.exponential.Ks
    hd = -15.24
    eps0 = jnp.exp(alpha0 * hd)
    d = alpha0 * (thetas - thetar) / Ks
    
    # Convert test case string to number
    test_map = {'Test1': 1, 'Test2': 2, 'Test3': 3}
    test_number = test_map[config.test_case]
    
    # Calculate exact solution at final time
    h_exact, Se_exact = exact_solution(
        points[:, 0], points[:, 1], final_time, 
        test_number=test_number,
        L=L, alpha0=alpha0, eps0=eps0, d=d
    )
    
    # Calculate saturation
    Se_num = (theta - thetar) / (thetas - thetar)
    
    # Calculate errors
    l2_error_h = jnp.sqrt(jnp.mean((pressure_head - h_exact)**2))
    l2_error_s = jnp.sqrt(jnp.mean((Se_num - Se_exact)**2))
    linf_error_h = jnp.max(jnp.abs(pressure_head - h_exact))
    linf_error_s = jnp.max(jnp.abs(Se_num - Se_exact))
    
    # Relative L2 errors
    l2_relative_h = jnp.sqrt(jnp.sum((pressure_head - h_exact)**2)) / jnp.sqrt(jnp.sum(h_exact**2))
    l2_relative_s = jnp.sqrt(jnp.sum((Se_num - Se_exact)**2)) / jnp.sqrt(jnp.sum(Se_exact**2))
    
    return {
        'l2_pressure': l2_error_h,
        'linf_pressure': linf_error_h,
        'l2_saturation': l2_error_s,
        'linf_saturation': linf_error_s,
        'l2_relative_pressure': l2_relative_h,
        'l2_relative_saturation': l2_relative_s
    }