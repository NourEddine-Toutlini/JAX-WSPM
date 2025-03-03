"""
Performance monitoring and reporting utilities.
"""

import os
import psutil
import time
from typing import Dict, Any
import numpy as np
from colorama import Fore, Style, init

# Initialize colorama for colored output
init()

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

def get_system_info() -> Dict[str, Any]:
    """
    Gather system information including CPU, memory, and GPU if available.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        'cpu': {
            'percent': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count(),
            'freq': psutil.cpu_freq() if hasattr(psutil, 'cpu_freq') else None
        },
        'memory': {
            'total': psutil.virtual_memory().total,
            'available': psutil.virtual_memory().available,
            'percent': psutil.virtual_memory().percent
        },
        'process': {
            'memory': psutil.Process(os.getpid()).memory_info().rss
        }
    }
    
    # Add GPU information if available
    if NVIDIA_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info['gpu'] = {
                'name': pynvml.nvmlDeviceGetName(handle),
                'memory': pynvml.nvmlDeviceGetMemoryInfo(handle),
                'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle)
            }
            pynvml.nvmlShutdown()
        except Exception as e:
            info['gpu'] = {'error': str(e)}
            
    return info

def format_memory(bytes: int) -> str:
    """Format memory size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} PB"

def print_simulation_summary(results: Dict[str, Any]) -> None:
    """
    Print detailed simulation performance summary.
    
    Args:
        results: Dictionary containing simulation results and metrics
    """
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*70)
    print(f"{Fore.YELLOW}{Style.BRIGHT}Richards Equation Solver - Simulation Summary".center(70))
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*70)
    
    # Timing Statistics
    print(f"\n{Fore.GREEN}{Style.BRIGHT}⏱️  Timing Statistics:")
    print(f"{Fore.WHITE}   • Total Simulation Time: {Fore.YELLOW}{results['simulation_time']:.2f} seconds")
    if 'times' in results:
        print(f"{Fore.WHITE}   • Physical Time Simulated: {Fore.YELLOW}{results['times'][-1]:.2f} time units")
        print(f"{Fore.WHITE}   • Average Time per Step: {Fore.YELLOW}{results['simulation_time']/len(results['times']):.4f} seconds")
    
    # Convergence Statistics
    if 'iterations' in results and 'errors' in results:
        iterations = np.array(results['iterations'])
        errors = np.array(results['errors'])
        print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 Convergence Statistics:")
        print(f"{Fore.WHITE}   • Total Iterations: {Fore.YELLOW}{np.sum(iterations)}")
        print(f"{Fore.WHITE}   • Average Iterations per Step: {Fore.YELLOW}{np.mean(iterations):.2f}")
        print(f"{Fore.WHITE}   • Maximum Iterations: {Fore.YELLOW}{np.max(iterations)}")
        print(f"{Fore.WHITE}   • Final Error: {Fore.YELLOW}{errors[-1]:.2e}")
        print(f"{Fore.WHITE}   • Average Error: {Fore.YELLOW}{np.mean(errors):.2e}")
    
    # Mesh Statistics
    if 'mesh_info' in results:
        mesh_info = results['mesh_info']
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🔍 Mesh Statistics:")
        print(f"{Fore.WHITE}   • Number of Nodes: {Fore.YELLOW}{mesh_info['num_points']}")
        print(f"{Fore.WHITE}   • Number of Elements: {Fore.YELLOW}{mesh_info['num_elements']}")
        print(f"{Fore.WHITE}   • Domain Size: {Fore.YELLOW}{mesh_info['x_range'][1]-mesh_info['x_range'][0]:.2f} × "
              f"{mesh_info['y_range'][1]-mesh_info['y_range'][0]:.2f}")
        
    # Error Metrics (New Section)

    if any(key in results for key in ['l2_pressure', 'linf_pressure', 'l2_saturation', 'linf_saturation']):
        print(f"\n{Fore.GREEN}{Style.BRIGHT}📈 Error Metrics:")
        if 'l2_pressure' in results:
            print(f"{Fore.WHITE}   • L2 Error (Pressure): {Fore.YELLOW}{float(results['l2_pressure']):.2e}")
            print(f"{Fore.WHITE}   • L∞ Error (Pressure): {Fore.YELLOW}{float(results['linf_pressure']):.2e}")
        if 'l2_saturation' in results:
            print(f"{Fore.WHITE}   • L2 Error (Saturation): {Fore.YELLOW}{float(results['l2_saturation']):.2e}")
            print(f"{Fore.WHITE}   • L∞ Error (Saturation): {Fore.YELLOW}{float(results['linf_saturation']):.2e}")
        if 'l2_relative_pressure' in results:
            print(f"{Fore.WHITE}   • Relative L2 Error (Pressure): {Fore.YELLOW}{float(results['l2_relative_pressure']):.2e}")
            print(f"{Fore.WHITE}   • Relative L2 Error (Saturation): {Fore.YELLOW}{float(results['l2_relative_saturation']):.2e}")
        
    
    # System Resource Usage
    system_info = get_system_info()
    print(f"\n{Fore.GREEN}{Style.BRIGHT}💻 System Resource Usage:")
    print(f"{Fore.WHITE}   • CPU Usage: {Fore.YELLOW}{system_info['cpu']['percent']}%")
    print(f"{Fore.WHITE}   • Memory Usage: {Fore.YELLOW}{system_info['memory']['percent']}%")
    print(f"{Fore.WHITE}   • Process Memory: {Fore.YELLOW}{format_memory(system_info['process']['memory'])}")
    
    if 'gpu' in system_info:
        gpu_info = system_info['gpu']
        if 'error' not in gpu_info:
            print(f"{Fore.WHITE}   • GPU Usage: {Fore.YELLOW}{gpu_info['utilization'].gpu}%")
            print(f"{Fore.WHITE}   • GPU Memory: {Fore.YELLOW}{format_memory(gpu_info['memory'].used)}")
    
    # Performance Insights
    print(f"\n{Fore.GREEN}{Style.BRIGHT}💡 Performance Insights:")
    
    # Memory efficiency
    mem_per_node = system_info['process']['memory'] / mesh_info['num_points']
    print(f"{Fore.WHITE}   • Memory per Node: {Fore.YELLOW}{format_memory(mem_per_node)}")
    
    if 'iterations' in results:
        if np.max(iterations) > 2 * np.mean(iterations):
            print(f"{Fore.RED}   ⚠️  High iteration variance detected - consider adjusting time step parameters")
        
        if np.mean(iterations) > 10:
            print(f"{Fore.YELLOW}   ⚠️  High average iteration count - consider using a stronger preconditioner")
    
    if system_info['memory']['percent'] > 80:
        print(f"{Fore.RED}   ⚠️  High memory usage - consider using a coarser mesh")
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*70 + f"{Style.RESET_ALL}")

def log_performance_metrics(output_dir: str, results: Dict[str, Any]) -> None:
    """
    Save performance metrics to a log file.
    
    Args:
        output_dir: Directory to save the log file
        results: Dictionary containing simulation results and metrics
    """
    log_file = os.path.join(output_dir, 'performance_log.txt')
    system_info = get_system_info()
    
    with open(log_file, 'w') as f:
        f.write("Performance Log\n")
        f.write("===============\n\n")
        
        # Write timing information
        f.write("Timing Information:\n")
        f.write(f"Total Simulation Time: {results['simulation_time']:.2f} seconds\n")
        f.write(f"Physical Time Simulated: {results['times'][-1]:.2f} time units\n\n")
        
        # Write convergence information
        f.write("Convergence Information:\n")
        f.write(f"Total Iterations: {np.sum(results['iterations'])}\n")
        f.write(f"Average Iterations: {np.mean(results['iterations']):.2f}\n")
        f.write(f"Final Error: {results['errors'][-1]:.2e}\n\n")
        
        # Write system information
        f.write("System Information:\n")
        f.write(f"CPU Usage: {system_info['cpu']['percent']}%\n")
        f.write(f"Memory Usage: {system_info['memory']['percent']}%\n")
        f.write(f"Process Memory: {format_memory(system_info['process']['memory'])}\n")