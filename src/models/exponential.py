# src/models/exponential.py
"""
Implementation of the Gardner exponential model for soil water retention.
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple

@jit
def exponential_model(h: jnp.ndarray, phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute soil hydraulic properties using the Gardner model.
    
    Args:
        h: Pressure head
        phi: Array of model parameters [alpha0, tetas, tetar, ks]
    
    Returns:
        Tuple of (C, K, theta) where:
            C: Specific moisture capacity
            K: Hydraulic conductivity
            theta: Water content
    """
    alpha0, tetas, tetar, ks = phi
    
    # Compute water content
    theta = jnp.where(h <= 0,
                      (tetas - tetar) * jnp.exp(alpha0 * h) + tetar,
                      tetas)
    
    # Compute hydraulic conductivity
    K = jnp.where(h <= 0,
                  ks * jnp.exp(alpha0 * h),
                  ks)
    
    # Compute specific moisture capacity
    C = jnp.where(h <= 0,
                  alpha0 * (tetas - tetar) * jnp.exp(alpha0 * h),
                  0.0)
    
    return C, K, theta

@jit
def compute_soil_properties(pressure_head: jnp.ndarray,
                          soil_params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Vectorized computation of soil hydraulic properties.
    
    Args:
        pressure_head: Array of pressure head values
        soil_params: Array of soil parameters [alpha0, tetas, tetar, ks]
    
    Returns:
        Tuple of (Capacity, Conductivity, WaterContent) arrays
    """
    # Apply exponential model to each pressure head value
    C, K, theta = jax.vmap(exponential_model, in_axes=(0, None))(pressure_head, soil_params)
    return C, K, theta

def validate_parameters(soil_params: jnp.ndarray) -> bool:
    """
    Validate soil parameters to ensure they are physically meaningful.
    
    Args:
        soil_params: Array of soil parameters [alpha0, tetas, tetar, ks]
        
    Returns:
        True if parameters are valid, raises ValueError otherwise
    """
    alpha0, tetas, tetar, ks = soil_params
    
    if tetas <= tetar:
        raise ValueError("Saturated water content must be greater than residual water content")
    
    if alpha0 <= 0:
        raise ValueError("Alpha parameter must be positive")
        
    if ks <= 0:
        raise ValueError("Saturated hydraulic conductivity must be positive")
        
    if tetar < 0 or tetas > 1:
        raise ValueError("Water content values must be between 0 and 1")
        
    return True