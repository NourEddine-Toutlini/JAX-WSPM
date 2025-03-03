"""
Utilities for handling simulation results saving and loading.
Optimized for cluster computing without visualization dependencies.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any, Optional

def save_simulation_results(results: Dict[str, Any], 
                          output_dir: str,
                          test_case: str,
                          save_frequency: int = 5) -> None:
    """
    Save simulation results to NPZ files.
    
    Args:
        results: Dictionary containing simulation results
        output_dir: Directory to save results
        test_case: Name of test case (Test1, Test2, Test3)
        save_frequency: Frequency of saving timesteps
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main time series results
    indices = np.arange(0, len(results['times']), save_frequency)
    main_results_file = output_path / f'{test_case}_time_series.npz'
    np.savez(
        main_results_file,
        pressure_head=np.array(results['pressure_head'][indices]),
        theta=np.array(results['theta'][indices]),
        times=np.array(results['times'][indices]),
        iterations=np.array(results['iterations'][indices]),
        errors=np.array(results['errors'][indices]),
        dt_values=np.array(results['dt_values'][indices])
    )
    
    # Save mesh and final state data
    mesh_results_file = output_path / f'{test_case}_mesh_and_final.npz'
    np.savez(
        mesh_results_file,
        points=np.array(results['points']),
        triangles=np.array(results['triangles']),
        final_pressure_head=np.array(results['pressure_head'][-1]),
        final_theta=np.array(results['theta'][-1])
    )
    
    # Save performance data
    performance_file = output_path / f'{test_case}_performance.npz'
    np.savez(
        performance_file,
        simulation_time=np.array(results['simulation_time']),
        total_iterations=np.array(np.sum(results['iterations'])),
        average_iterations=np.array(np.mean(results['iterations'])),
        final_error=np.array(results['errors'][-1]),
        average_error=np.array(np.mean(results['errors'])),
        total_timesteps=np.array(len(results['times'])),
        final_time=np.array(results['times'][-1])
    )

def load_simulation_results(output_dir: str, test_case: str) -> Dict[str, Any]:
    """
    Load simulation results from NPZ files.
    
    Args:
        output_dir: Directory containing results
        test_case: Name of test case (Test1, Test2, Test3)
        
    Returns:
        Dictionary containing loaded results
    """
    output_path = Path(output_dir)
    
    # Load time series data
    time_series = np.load(output_path / f'{test_case}_time_series.npz')
    
    # Load mesh and final state data
    mesh_data = np.load(output_path / f'{test_case}_mesh_and_final.npz')
    
    # Load performance data
    performance = np.load(output_path / f'{test_case}_performance.npz')
    
    # Combine all results
    results = {
        # Time series data
        'pressure_head': jnp.array(time_series['pressure_head']),
        'theta': jnp.array(time_series['theta']),
        'times': jnp.array(time_series['times']),
        'iterations': jnp.array(time_series['iterations']),
        'errors': jnp.array(time_series['errors']),
        'dt_values': jnp.array(time_series['dt_values']),
        
        # Mesh and final state data
        'points': jnp.array(mesh_data['points']),
        'triangles': jnp.array(mesh_data['triangles']),
        'final_pressure_head': jnp.array(mesh_data['final_pressure_head']),
        'final_theta': jnp.array(mesh_data['final_theta']),
        
        # Performance data
        'simulation_time': float(performance['simulation_time']),
        'total_iterations': int(performance['total_iterations']),
        'average_iterations': float(performance['average_iterations']),
        'final_error': float(performance['final_error']),
        'average_error': float(performance['average_error']),
        'total_timesteps': int(performance['total_timesteps']),
        'final_time': float(performance['final_time'])
    }
    
    return results

def verify_results(results: Dict[str, Any]) -> bool:
    """
    Verify that all required fields are present in results.
    
    Args:
        results: Dictionary of simulation results
        
    Returns:
        True if all required fields are present
        
    Raises:
        ValueError if any required fields are missing
    """
    required_fields = [
        'pressure_head', 'theta', 'times', 'iterations', 'errors', 'dt_values',
        'points', 'triangles', 'final_theta', 'simulation_time'
    ]
    
    missing_fields = [field for field in required_fields if field not in results]
    
    if missing_fields:
        raise ValueError(f"Missing required fields in results: {missing_fields}")
    
    return True