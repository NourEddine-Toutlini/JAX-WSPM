"""
Gaussian quadrature and shape function implementations for 3D tetrahedral finite elements.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple

@partial(jit, static_argnums=(0,))
def gauss_tetrahedron(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get Gauss quadrature points and weights for tetrahedral elements.
    
    Args:
        n: Number of quadrature points (1, 4, 5, 11, or 15 supported)
        
    Returns:
        Tuple of (quadrature points, weights)
        
    Raises:
        ValueError: If n is not supported
    """
    if n == 1:
        # 1-point quadrature (degree of precision 1)
        Q = jnp.array([[0.25, 0.25, 0.25]])
        W = jnp.array([1.0])
        
    elif n == 4:
        # 4-point quadrature (degree of precision 2)
        a = 0.585410196624969
        b = 0.138196601125011
        Q = jnp.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ])
        W = jnp.array([0.25, 0.25, 0.25, 0.25])
        
    elif n == 5:
        # 5-point quadrature (degree of precision 3)
        a = 0.25
        b = 0.5
        c = 1/6
        Q = jnp.array([
            [a, a, a],
            [b, c, c],
            [c, b, c],
            [c, c, b],
            [c, c, c]
        ])
        W = jnp.array([-0.8, 0.45, 0.45, 0.45, 0.45])
        
    elif n == 11:
        # 11-point quadrature (degree of precision 4)
        a = 0.25
        b = 0.785714285714286
        c = 0.071428571428571
        d = 0.399403576166799
        e = 0.100596423833201
        Q = jnp.array([
            [a, a, a],
            [b, c, c],
            [c, b, c],
            [c, c, b],
            [c, c, c],
            [d, d, e],
            [d, e, d],
            [e, d, d],
            [d, e, e],
            [e, d, e],
            [e, e, d]
        ])
        W = jnp.array([-0.013155555555556,
                       0.007622222222222,
                       0.007622222222222,
                       0.007622222222222,
                       0.007622222222222,
                       0.024888888888889,
                       0.024888888888889,
                       0.024888888888889,
                       0.024888888888889,
                       0.024888888888889,
                       0.024888888888889])
        
    elif n == 15:
        # 15-point quadrature (degree of precision 5)
        a = 0.25
        b = 0.0
        c = 1/3
        d = 0.727272727272727
        e = 0.090909090909091
        f = 0.066550153573664
        g = 0.433449846426336
        Q = jnp.array([
            [a, a, a],
            [b, c, c],
            [c, b, c],
            [c, c, b],
            [c, c, c],
            [d, e, e],
            [e, d, e],
            [e, e, d],
            [e, e, e],
            [f, f, g],
            [f, g, f],
            [g, f, f],
            [f, g, g],
            [g, f, g],
            [g, g, f]
        ])
        W = jnp.array([0.030283678097089,
                       0.006026785714286,
                       0.006026785714286,
                       0.006026785714286,
                       0.006026785714286,
                       0.011645249086029,
                       0.011645249086029,
                       0.011645249086029,
                       0.011645249086029,
                       0.010949141561386,
                       0.010949141561386,
                       0.010949141561386,
                       0.010949141561386,
                       0.010949141561386,
                       0.010949141561386])
    else:
        raise ValueError('n must be 1, 4, 5, 11, or 15')
    
    return Q, W

@jit
def shape_functions_3d(ksi: float, eta: float, zeta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute shape functions and their derivatives for tetrahedral elements.
    
    Args:
        ksi: First local coordinate
        eta: Second local coordinate
        zeta: Third local coordinate
        
    Returns:
        Tuple of (shape functions, derivatives)
    """
    # Shape functions for tetrahedron
    N = jnp.array([
        1.0 - ksi - eta - zeta,  # N1
        ksi,                     # N2
        eta,                     # N3
        zeta                     # N4
    ])
    
    # Derivatives of shape functions with respect to ksi, eta, zeta
    dN = jnp.array([
        [-1.0, 1.0, 0.0, 0.0],  # d/dksi
        [-1.0, 0.0, 1.0, 0.0],  # d/deta
        [-1.0, 0.0, 0.0, 1.0]   # d/dzeta
    ])
    
    return N, dN

@jit
def basis_ksi_eta_zeta(ksi: float, eta: float, zeta: float) -> jnp.ndarray:
    """
    Compute basis function derivatives with respect to ksi, eta, and zeta.
    
    Args:
        ksi: First local coordinate
        eta: Second local coordinate
        zeta: Third local coordinate
        
    Returns:
        Array of basis function derivatives
    """
    return jnp.array([
        [-1.0, 1.0, 0.0, 0.0],  # d/dksi
        [-1.0, 0.0, 1.0, 0.0],  # d/deta
        [-1.0, 0.0, 0.0, 1.0]   # d/dzeta
    ])

@jit
def interpolation_3d(ksi: float, eta: float, zeta: float) -> jnp.ndarray:
    """
    Compute interpolation functions for tetrahedral elements.
    
    Args:
        ksi: First local coordinate
        eta: Second local coordinate
        zeta: Third local coordinate
        
    Returns:
        Array of interpolation functions
    """
    return jnp.array([
        1.0 - ksi - eta - zeta,  # N1
        ksi,                     # N2
        eta,                     # N3
        zeta                     # N4
    ])