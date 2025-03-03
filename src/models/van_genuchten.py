# src/models/van_genuchten.py
"""
Implementation of the Van Genuchten model for soil water retention and hydraulic conductivity.
"""
import jax.numpy as jnp
from jax import jit
from typing import Tuple

@jit
def van_genuchten_model(h: float, phi: jnp.ndarray) -> Tuple[float, float, float]:
    """
    Compute soil hydraulic properties using the Van Genuchten model.
    
    Args:
        h: Pressure head
        phi: Array of model parameters [alpha, theta_S, theta_R, n, m, Ksat]
    
    Returns:
        Tuple of (C, K, theta) where:
            C: Specific moisture capacity
            K: Hydraulic conductivity
            theta: Water content
    """
    alpha, theta_S, theta_R, n, m, Ksat = phi
    
    # Compute water content
    theta = jnp.where(h < 0,
                     (theta_S - theta_R) * ((1 + (alpha * jnp.abs(h))**n)**(-m)) + theta_R,
                     theta_S)
    
    # Compute relative saturation
    Se = jnp.where(h < 0,
                   (theta - theta_R) / (theta_S - theta_R),
                   1.0)
    
    # Compute hydraulic conductivity
    K = jnp.where(h < 0,
                  Ksat * ((Se**0.5) * (1 - (1 - Se**(1/m))**m)**2),
                  Ksat)
    
    # Compute specific moisture capacity
    C = jnp.where(h < 0,
                  (theta_S - theta_R) * alpha * m * n * (alpha * jnp.abs(h))**(n-1) *
                  (1 + (alpha * jnp.abs(h))**n)**(-m-1),
                  0.0)
    
    return C, K, theta

def validate_parameters(phi: jnp.ndarray) -> bool:
    """
    Validate Van Genuchten parameters to ensure they are physically meaningful.
    
    Args:
        phi: Array of parameters [alpha, theta_S, theta_R, n, m, Ksat]
        
    Returns:
        True if parameters are valid, raises ValueError otherwise
    """
    alpha, theta_S, theta_R, n, m, Ksat = phi
    
    if theta_S <= theta_R:
        raise ValueError("Saturated water content must be greater than residual water content")
    
    if alpha <= 0:
        raise ValueError("Alpha parameter must be positive")
        
    if n <= 1:
        raise ValueError("n parameter must be greater than 1")
        
    if m <= 0 or m >= 1:
        raise ValueError("m parameter must be between 0 and 1")
        
    if Ksat <= 0:
        raise ValueError("Saturated hydraulic conductivity must be positive")
        
    if theta_R < 0 or theta_S > 1:
        raise ValueError("Water content values must be between 0 and 1")
        
    return True
