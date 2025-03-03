# src/models/transport_utilities.py
"""
Utilities for calculating water fluxes and dispersion tensors in solute transport.
"""
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Tuple, Callable
from functools import partial

@jit
def interpolate_pressure_head(x: float, y: float, points: jnp.ndarray,
                            triangles: jnp.ndarray, pressure_head: jnp.ndarray) -> float:
    """
    Interpolate pressure head at arbitrary point using shape functions.
    """
    def find_triangle(x: float, y: float) -> jnp.ndarray:
        def point_in_triangle(triangle: jnp.ndarray) -> bool:
            x1, y1 = points[triangle[0]]
            x2, y2 = points[triangle[1]]
            x3, y3 = points[triangle[2]]
            
            def sign(p1x, p1y, p2x, p2y, p3x, p3y):
                return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)
            
            d1 = sign(x, y, x1, y1, x2, y2)
            d2 = sign(x, y, x2, y2, x3, y3)
            d3 = sign(x, y, x3, y3, x1, y1)
            
            has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
            has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
            
            return ~(has_neg & has_pos)
        
        triangle_index = jnp.argmax(vmap(point_in_triangle)(triangles))
        return triangles[triangle_index]
    
    triangle = find_triangle(x, y)
    x1, y1 = points[triangle[0]]
    x2, y2 = points[triangle[1]]
    x3, y3 = points[triangle[2]]
    
    det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    l1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
    l2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
    l3 = 1 - l1 - l2
    
    return (l1*pressure_head[triangle[0]] +
            l2*pressure_head[triangle[1]] +
            l3*pressure_head[triangle[2]])

@jit
def calculate_fluxes_and_dispersion(points: jnp.ndarray,
                                  triangles: jnp.ndarray,
                                  pressure_head: jnp.ndarray,
                                  phi: jnp.ndarray,
                                  DL: float,
                                  DT: float,
                                  Dm: float) -> Tuple[jnp.ndarray, ...]:
    """
    Calculate water fluxes and dispersion tensor components for solute transport.
    
    Args:
        points: Mesh points coordinates
        triangles: Element connectivity
        pressure_head: Current pressure head solution
        phi: Van Genuchten parameters
        DL: Longitudinal dispersivity
        DT: Transverse dispersivity
        Dm: Molecular diffusion coefficient
        
    Returns:
        Tuple of arrays for water fluxes and dispersion components
    """
    def pressure_head_function(x, y):
        return interpolate_pressure_head(x, y, points, triangles, pressure_head)
    
    grad_h = grad(pressure_head_function, argnums=(0, 1))
    
    from .van_genuchten import van_genuchten_model
    
    def process_node(node):
        x, y = points[node]
        dh_dx, dh_dy = grad_h(x, y)
        
        _, K, theta = van_genuchten_model(pressure_head[node], phi)
        
        # Calculate water fluxes
        water_flux_x = -K * dh_dx
        water_flux_z = -K * (dh_dy + 1)  # +1 for gravity
        
        abs_q = jnp.sqrt(water_flux_x**2 + water_flux_z**2)
        
        # Calculate tortuosity
        theta_s = phi[1]
        tau = (theta**(7/3)) / theta_s**2
        
        # Calculate dispersion tensor components
        Dispersion_xx = ((DL * water_flux_x**2 + DT * water_flux_z**2) / abs_q +
                        theta * Dm * tau)
        Dispersion_zz = ((DL * water_flux_z**2 + DT * water_flux_x**2) / abs_q +
                        theta * Dm * tau)
        Dispersion_xz = (DL - DT) * water_flux_x * water_flux_z / abs_q
        
        return (water_flux_x, water_flux_z, abs_q,
                Dispersion_xx, Dispersion_xz, Dispersion_zz, theta)
    
    return vmap(process_node)(jnp.arange(points.shape[0]))

@jit
def calculate_element_size(points: jnp.ndarray, triangles: jnp.ndarray) -> jnp.ndarray:
    """Calculate characteristic size for each element."""
    def element_size(triangle):
        vertices = points[triangle]
        edges = vertices - jnp.roll(vertices, 1, axis=0)
        return jnp.min(jnp.sqrt(jnp.sum(edges**2, axis=1)))
    
    return vmap(element_size)(triangles)

@jit
def calculate_peclet_courant(points: jnp.ndarray,
                           triangles: jnp.ndarray,
                           water_flux_x: jnp.ndarray,
                           water_flux_z: jnp.ndarray,
                           theta: jnp.ndarray,
                           Dispersion_xx: jnp.ndarray,
                           Dispersion_zz: jnp.ndarray,
                           dt: float) -> Tuple[float, float]:
    """
    Calculate Peclet and Courant numbers for stability analysis.
    """
    element_sizes = calculate_element_size(points, triangles)
    
    def calculate_for_element(element, h):
        nodes = triangles[element]
        v_x = water_flux_x[nodes] / theta[nodes]
        v_z = water_flux_z[nodes] / theta[nodes]
        D_xx = Dispersion_xx[nodes]
        D_zz = Dispersion_zz[nodes]
        
        Pe_x = (jnp.abs(v_x) * h) / (2 * D_xx)
        Pe_z = (jnp.abs(v_z) * h) / (2 * D_zz)
        Pe = jnp.maximum(jnp.max(Pe_x), jnp.max(Pe_z))
        
        Cr_x = (jnp.abs(v_x) * dt) / h
        Cr_z = (jnp.abs(v_z) * dt) / h
        Cr = jnp.maximum(jnp.max(Cr_x), jnp.max(Cr_z))
        
        return Pe, Cr
    
    Pe_Cr = vmap(calculate_for_element)(jnp.arange(triangles.shape[0]),
                                      element_sizes)
    Pe_values = jnp.array([pc[0] for pc in Pe_Cr])
    Cr_values = jnp.array([pc[1] for pc in Pe_Cr])
    
    return jnp.max(Pe_values), jnp.max(Cr_values)
