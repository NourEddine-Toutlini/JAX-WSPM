# src/cli.py

import jax
# print(jax.__version__)
import os
import jax.numpy as jnp
from jax import jit, vmap, lax
import argparse
import json
from pathlib import Path
import time
from colorama import Fore, Style, init
import psutil

from config.settings import (
    SimulationConfig, 
    TimeSteppingParameters,
    SolverParameters,
    ExponentialParameters,
    VanGenuchtenParameters,
    SoluteParameters
)
from src.solvers.richards_solver import RichardsSolver
from src.solvers.coupled_solver_v2 import CoupledSolver
from src.utils.plotting import ResultsVisualizer 
from src.utils.performance import print_simulation_summary
from src.utils.results_handler import save_simulation_results
from src.utils.error_calculation import calculate_final_errors 
from src.utils.exact_solutions import exact_solution

from config.welcome_banner import (
    print_welcome_banner, 
    print_fancy_banner
)

# At the top of your file, after imports
#jax.config.update('jax_debug_nans', True)
jax.config.update('jax_enable_x64', True)
#jax.config.update('jax_debug_infs', True)

from jax import profiler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Richards Equation and Solute Transport Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mesh options
    parser.add_argument('--mesh-size', choices=['25', '50', '100', '200', '1024', '2048', '4096', '8192', '16384', '65536', '267039', '1014454'],
                       default='25', help='Mesh resolution')
    
    # Test case selection
    parser.add_argument('--test-case', 
                       choices=['Test1', 'Test2', 'Test3', 'SoluteTest'],
                       default='Test1', 
                       help='Test case to simulate')
    
    # Solver options
    parser.add_argument('--solver', choices=['gmres', 'cg', 'bicgstab', 'direct'],
                       default='gmres', help='Linear solver type')
    parser.add_argument('--preconditioner',
                       choices=['none', 'jacobi', 'ilu', 'block_jacobi', 'ssor'],
                       default='none', help='Preconditioner type')
    
    # Time stepping options
    parser.add_argument('--dt', type=float, default=1e-6,
                       help='Initial time step size')
    parser.add_argument('--tmax', type=float, default=1.0,
                       help='Maximum simulation time')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    parser.add_argument('--save-frequency', type=int, default=5,
                       help='Frequency of saving results')
    
    return parser.parse_args()



def main():
    init()  # Initialize colorama
    #print_welcome_banner()
    print_fancy_banner()
    
    # jax.print_environment_info()
    # devices = jax.devices()
    # num_devices = jax.device_count()

    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create time stepping parameters
    time_params = TimeSteppingParameters(
        dt_init=args.dt,
        Tmax=args.tmax
    )
    
    # Create solver parameters
    solver_params = SolverParameters(
        solver_type=args.solver,
        precond_type=args.preconditioner
    )
    
    # Create main configuration
    config = SimulationConfig(
        exponential=ExponentialParameters(),
        van_genuchten=VanGenuchtenParameters(),
        solute=SoluteParameters(),
        time=time_params,
        solver=solver_params,
        test_case=args.test_case,
        save_frequency=args.save_frequency
    )
    
    config.mesh.mesh_size = args.mesh_size
    # Save configuration
    config_file = output_dir / 'simulation_config.json'
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    
    print(f"\n{Fore.GREEN}Starting simulation with configuration:{Style.RESET_ALL}")
    print(f"Test case: {args.test_case}")
    print(f"Solver: {args.solver} with {args.preconditioner} preconditioner")
    
    try:
        # Choose appropriate solver based on test case
        if args.test_case == 'SoluteTest':
            solver = CoupledSolver(config)
            
        else:
            # solver = RichardsSolver(config)
            solver = RichardsSolver(config)
        
        # Run simulation
        
        results = solver.solve()
        
        # Calculate errors for Test1, Test2, Test3
        if args.test_case != 'SoluteTest':
            error_metrics = calculate_final_errors(results, config)
            results.update(error_metrics)
        
        
        # Save numerical results for the RE
        results_file = output_dir / f'numerical_results_{args.test_case}.npz'  #_{args.solver}_{args.preconditioner}_mesh{args.mesh_size}
        
        jnp.savez(results_file,
             pressure_head=results['pressure_head'],
             theta=results.get('theta', None),
             solute=results.get('solute', None),
             times=results['times'],
             points=results['points'],
             triangles=results['triangles'],
             iterations=results['iterations'],
             errors=results['errors'],
             dt_values=results['dt_values'],
             l2_pressure=float(results.get('l2_pressure', 0)),
             linf_pressure=float(results.get('linf_pressure', 0)),
             l2_saturation=float(results.get('l2_saturation', 0)),
             linf_saturation=float(results.get('linf_saturation', 0)),
             l2_relative_pressure=float(results.get('l2_relative_pressure', 0)),
             l2_relative_saturation=float(results.get('l2_relative_saturation', 0)),
             simulation_time=results['simulation_time'],
             memory_usage=float(psutil.Process().memory_info().rss / (1024 * 1024)))  # Memory in MB

        # Create and save visualizations 
        
        if args.test_case == 'SoluteTest':
            visualizer = ResultsVisualizer(output_dir)
            visualizer.plot_final_state(
                results['points'],  # ? Pass mesh points
                results['triangles'],  # ? Pass triangle indices
                results['final_theta'],  
                results['final_solute'],  
                # results['final_pressure'], 
                output_dir / 'final_state.png'
            )

        else:
            visualizer = ResultsVisualizer(output_dir)
            visualizer.plot_all_results(results)
        
        print(f"\n{Fore.GREEN}Results saved successfully:{Style.RESET_ALL}")
        print(f"  - Configuration: {config_file}")
        print(f"  - Numerical results: {results_file}")
        print(f"  - Plots: {output_dir}/final_state.png")
        
        # Print performance summary
        print_simulation_summary(results)
        
    except Exception as e:
        print(f"\n{Fore.RED}Error during simulation:{Style.RESET_ALL}")
        print(f"{str(e)}")
        raise

        
if __name__ == "__main__":
    main()
    