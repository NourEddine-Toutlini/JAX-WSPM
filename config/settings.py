# config/settings.py

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Tuple
import jax.numpy as jnp
from pathlib import Path
from functools import partial

@dataclass
class VanGenuchtenParameters:
    """Van Genuchten soil hydraulic parameters."""
    Ks: float = 0.25       # Saturated hydraulic conductivity
    thetas: float = 0.43   # Saturated water content
    thetar: float = 0.078  # Residual water content
    alpha: float = 3.6     # Van Genuchten a parameter
    n: float = 1.56        # Van Genuchten n parameter
    m: float = 1 - 1/n       # Van Genuchten m parameter (typically 1-1/n)
    
    def to_array(self) -> jnp.ndarray:
        """Convert parameters to JAX array format."""
        return jnp.array([self.alpha, self.thetas, self.thetar, 
                         self.n, self.m, self.Ks])
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SoluteParameters:
    """Parameters for solute transport."""
    DL: float = 0.5        # Longitudinal dispersivity
    DT: float = 0.1        # Transverse dispersivity
    Dm: float = 0.0        # Molecular diffusion coefficient
    c_init: float = 0.1    # Initial concentration
    c_inlet: float = 1.0   # Inlet concentration for Cauchy BC

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ExponentialParameters:
    """Original exponential model parameters."""
    Ks: float = 0.1        # Saturated hydraulic conductivity
    thetas: float = 0.45   # Saturated water content
    thetar: float = 0.15   # Residual water content
    alpha0: float = 0.164  # Exponential model parameter
    
    def to_array(self) -> jnp.ndarray:
        return jnp.array([self.alpha0, self.thetas, self.thetar, self.Ks])
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TimeSteppingParameters:
    """Parameters for adaptive time-stepping."""
    dt_init: float = 1e-5     # Initial time step
    Tmax: float = 1.0         # Maximum simulation time
    m_it: int = 5             # Min iterations for time step increase
    M_it: int = 7             # Max iterations before reduction
    W_it: int = 10            # Warning iterations threshold
    lambda_amp: float = 2.0   # Time step amplification factor
    lambda_red: float = 0.5   # Time step reduction factor
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class SolverParameters:
    """Linear solver configuration."""
    solver_type: str = 'gmres'    
    precond_type: str = 'ilu'  
    tol: float = 1e-10
    maxiter: int = 100
    restart: int = 30
    precond_params: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 1.0,          
        'omega': 1.0,         
        'block_size': 2        
    })
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class MeshParameters:
    """Mesh configuration."""
    mesh_size: str = '25'    
    mesh_dir: str = 'data/mesh'
    test_case: str = 'Test1'  # Reference to test case to determine mesh type
    
    @property
    def mesh_files(self) -> Tuple[Path, Path]:
        """Get appropriate mesh files based on test case."""
        base_path = Path(self.mesh_dir)
        
        if self.test_case == 'SoluteTest':
            return (
                base_path / 'solute' / f'p_Pinns.csv',
                base_path / 'solute' / f't_Pinns.csv'
            )
        elif self.test_case == 'Test3D':
            return (
                base_path / 'richards' / 'p_3D.csv',
                base_path / 'richards' / 't_3D.csv'
            )
        else:
            return (
                base_path / 'richards' / f'p{self.mesh_size}.csv',
                base_path / 'richards' / f't{self.mesh_size}.csv'
            )
    
    def validate_mesh_files(self) -> bool:
        """Validate that required mesh files exist."""
        points_file, triangles_file = self.mesh_files
        if not points_file.exists():
            raise FileNotFoundError(
                f"Points file not found: {points_file}\n"
                f"Make sure the mesh files are in the correct directory structure:\n"
                f"- For Richards tests: {self.mesh_dir}/richards/\n"
                f"- For Solute test: {self.mesh_dir}/solute/"
            )
        if not triangles_file.exists():
            raise FileNotFoundError(
                f"Triangles file not found: {triangles_file}"
            )
        return True
    
    def to_dict(self) -> Dict:
        return {
            'mesh_size': self.mesh_size,
            'mesh_dir': str(self.mesh_dir),
            'test_case': self.test_case,
            'points_file': str(self.mesh_files[0]),
            'triangles_file': str(self.mesh_files[1])
        }

@dataclass
class SimulationConfig:
    """Main configuration combining all parameters."""
    exponential: ExponentialParameters = field(default_factory=ExponentialParameters)
    van_genuchten: VanGenuchtenParameters = field(default_factory=VanGenuchtenParameters)
    solute: SoluteParameters = field(default_factory=SoluteParameters)
    time: TimeSteppingParameters = field(default_factory=TimeSteppingParameters)
    solver: SolverParameters = field(default_factory=SolverParameters)
    test_case: str = 'Test1'  # Options: 'Test1', 'Test2', 'Test3', 'SoluteTest'
    save_frequency: int = 1
    
    def __post_init__(self):
        """Initialize mesh parameters after other attributes are set."""
        self.mesh = MeshParameters(test_case=self.test_case)
    
    def validate(self) -> bool:
        """Validate the configuration."""
        valid_tests = {'Test1', 'Test2', 'Test3', 'Test3D', 'SoluteTest'}  # Added Test3D
        valid_mesh_sizes = {'25', '50', '100', '4096'}
        valid_solvers = {'gmres', 'cg', 'bicgstab'}
        
        if self.test_case not in valid_tests:
            raise ValueError(f"Invalid test case. Must be one of {valid_tests}")
        
        if self.test_case not in ['SoluteTest', 'Test3D'] and self.mesh.mesh_size not in valid_mesh_sizes:
            raise ValueError(f"Invalid mesh size for Richards test. Must be one of {valid_mesh_sizes}")
            
        if self.solver.solver_type not in valid_solvers:
            raise ValueError(f"Invalid solver type. Must be one of {valid_solvers}")
        
        # Validate mesh files exist
        self.mesh.validate_mesh_files()
        
        return True
    
    def to_dict(self) -> Dict:
        return {
            'exponential': self.exponential.to_dict(),
            'van_genuchten': self.van_genuchten.to_dict(),
            'solute': self.solute.to_dict(),
            'time': self.time.to_dict(),
            'solver': self.solver.to_dict(),
            'mesh': self.mesh.to_dict(),
            'test_case': self.test_case,
            'save_frequency': self.save_frequency
        }

# Helper function to create configuration from dictionary
def create_config_from_dict(config_dict: Dict[str, Any]) -> SimulationConfig:
    """Create a SimulationConfig instance from a dictionary."""
    return SimulationConfig(
        exponential=ExponentialParameters(**config_dict.get('exponential', {})),
        van_genuchten=VanGenuchtenParameters(**config_dict.get('van_genuchten', {})),
        solute=SoluteParameters(**config_dict.get('solute', {})),
        time=TimeSteppingParameters(**config_dict.get('time', {})),
        solver=SolverParameters(**config_dict.get('solver', {})),
        test_case=config_dict.get('test_case', 'Test1'),
        save_frequency=config_dict.get('save_frequency', 1)
    )

# Default configuration instance
default_config = SimulationConfig()