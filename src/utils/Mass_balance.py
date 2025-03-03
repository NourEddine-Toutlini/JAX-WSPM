# src/utils/Mass_balance.py
import jax.numpy as jnp
from jax import vmap
from typing import Dict, Tuple
from ..utils.exact_solutions import exact_solution

def integrate_over_domain(theta: jnp.ndarray, points: jnp.ndarray,
                         triangles: jnp.ndarray) -> float:
    """
    Integrate moisture content over the domain using piece-wise constant approximation.
    """
    total_mass = 0.0
    
    for triangle in triangles:
        # Calculate element area
        x = points[triangle][:, 0]
        y = points[triangle][:, 1]
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - 
                        (x[2] - x[0]) * (y[1] - y[0]))
        
        # Average theta in element (piece-wise constant approximation)
        theta_avg = jnp.mean(theta[triangle])
        
        # Add contribution to total mass
        total_mass += area * theta_avg
    
    return total_mass

def calculate_total_mass(theta: jnp.ndarray, points: jnp.ndarray,
                        triangles: jnp.ndarray) -> float:
    """
    Calculate total mass for a given moisture content distribution.
    """
    return integrate_over_domain(theta, points, triangles)

def calculate_mass_balance(results: Dict, config: Dict) -> Dict:
    """
    Calculate mass balance evolution and relative error.
    """
    points = results['points']
    triangles = results['triangles']
    times = results['times']
    theta_numerical = results['theta']
    
    # Get parameters
    L = 15.24
    alpha0 = config.exponential.alpha0
    thetas = config.exponential.thetas
    thetar = config.exponential.thetar
    Ks = config.exponential.Ks
    hd = -15.24
    eps0 = jnp.exp(alpha0 * hd)
    d = alpha0 * (thetas - thetar) / Ks
    test_map = {'Test1': 1, 'Test2': 2, 'Test3': 3}
    test_number = test_map[config.test_case]
    print(test_number)
    
    # Initialize arrays for mass balance
    mb_num = jnp.zeros_like(times)
    mb_ex = jnp.zeros_like(times)
    mbe = jnp.zeros_like(times)
    
    # Calculate initial masses (t = 0)
    theta_initial = (thetas - thetar) * jnp.exp(alpha0 * hd) + thetar
    theta_initial = jnp.ones_like(points[:, 0]) * theta_initial
    
    # Calculate initial total mass
    initial_mass_num = calculate_total_mass(theta_initial, points, triangles)
    
    # Calculate mass balance evolution
    for i, t in enumerate(times):
        print(jnp.mean(theta_numerical[i]))
        # Calculate total mass for numerical solution at current time
        current_mass_num = calculate_total_mass(theta_numerical[i], points, triangles)
        mb_num_t = current_mass_num - initial_mass_num
        
        # Calculate exact solution at current time
        _, S_exact = exact_solution(points[:, 0], points[:, 1], i,
                                  test_number, L, alpha0, eps0, d)
        theta_exact = (thetas - thetar) * S_exact + thetar
        
        # Calculate total mass for exact solution at current time
        current_mass_ex = calculate_total_mass(theta_exact, points, triangles)
        mb_ex_t = current_mass_ex - initial_mass_num
        
        # Store results
        mb_num = mb_num.at[i].set(mb_num_t)
        mb_ex = mb_ex.at[i].set(mb_ex_t)
        
        # Calculate relative mass balance error (MBE)
        mbe = mbe.at[i].set(abs(1 - mb_num_t/mb_ex_t) * 100)
    
    return {
        'mass_balance_numerical': mb_num,
        'mass_balance_exact': mb_ex,
        'mass_balance_error': mbe
    }