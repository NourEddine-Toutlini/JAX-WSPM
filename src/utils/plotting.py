# src/utils/plotting.py
"""
Visualization utilities for the Richards equation solver.
"""

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

class ResultsVisualizer:
    """Handles visualization of simulation results."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('classic')
        self.default_figsize = (10, 6)
        self.default_dpi = 300
        
    def plot_water_content(self, 
                           points: jnp.ndarray,
                           triangles: jnp.ndarray,
                           theta: jnp.ndarray,
                           title: str = "Water Content Distribution",
                           filename: Optional[str] = None) -> None:
        """Plot water content distribution."""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        tri = Triangulation(points[:, 0], points[:, 1], triangles[:, 0:3])
        contour = ax.tricontourf(tri, np.array(theta), cmap='jet', levels=10)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Water Content (θ)', fontsize=6)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(title, fontsize=8, pad=10)
        ax.grid(False)
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.default_dpi,
                        bbox_inches='tight', facecolor='white')
            plt.close()

    def plot_convergence_history(self,
                                iterations: jnp.ndarray,
                                errors: jnp.ndarray,
                                filename: Optional[str] = None) -> None:
        """Plot convergence history."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.default_figsize)
        ax1.plot(iterations, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Iterations', fontsize=10)
        ax1.set_title('Iteration History', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax2.semilogy(errors, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel('Error', fontsize=10)
        ax2.set_title('Error History', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.default_dpi,
                        bbox_inches='tight', facecolor='white')
            plt.close()

    def plot_time_stepping(self,
                           times: jnp.ndarray,
                           dt_values: jnp.ndarray,
                           filename: Optional[str] = None) -> None:
        """Plot time stepping history."""
        fig, ax = plt.subplots(figsize=self.default_figsize)
        ax.semilogy(times, dt_values, 'g-', linewidth=1.5)
        ax.set_xlabel('Simulation Time', fontsize=10)
        ax.set_ylabel('Time Step Size (dt)', fontsize=10)
        ax.set_title('Adaptive Time Stepping History', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.default_dpi,
                        bbox_inches='tight', facecolor='white')
            plt.close()

    def plot_all_results(self, results: Dict[str, Any]) -> None:
        """Create all plots for simulation results."""
        self.plot_water_content(
            points=results['points'],
            triangles=results['triangles'],
            theta=results['final_theta'],
            filename='final_water_content.png'
        )
        self.plot_convergence_history(
            iterations=results['iterations'],
            errors=results['errors'],
            filename='convergence_history.png'
        )
        self.plot_time_stepping(
            times=results['times'],
            dt_values=results['dt_values'],
            filename='time_stepping.png'
        )

    def plot_final_state(self, points, triangles, final_theta, final_solute, filename):
        """Plot final water content and solute concentration distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
        tri = Triangulation(points[:, 0], points[:, 1], triangles)
        tricontour1 = ax1.tricontourf(tri, final_theta, cmap='jet', levels=8)
        ax1.set_xlabel('X', fontsize=5)
        ax1.set_ylabel('Y', fontsize=5)
        ax1.set_title('Final Water Content', fontsize=5)
        cbar1 = plt.colorbar(tricontour1, ax=ax1)
        cbar1.set_label('Water Content (θ)', fontsize=4)
        tricontour2 = ax2.tricontourf(tri, final_solute, cmap='jet', levels=8)
        ax2.set_xlabel('X', fontsize=5)
        ax2.set_ylabel('Y', fontsize=5)
        ax2.set_title('Final Solute Concentration', fontsize=5)
        cbar2 = plt.colorbar(tricontour2, ax=ax2)
        cbar2.set_label('Solute Concentration', fontsize=4)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=self.default_dpi,
                    bbox_inches='tight')
        plt.close()

    def plot_3d_results(self, results: Dict[str, Any]) -> None:
        """Create all plots for 3D simulation results."""
        self.plot_3d_water_content(
            points=results['points'],
            tetrahedra=results['tetrahedra'],
            theta=results['final_theta'],
            filename='final_water_content_3d.png'
        )
        self.plot_convergence_history(
            iterations=results['iterations'],
            errors=results['errors'],
            filename='convergence_history_3d.png'
        )
        self.plot_time_stepping(
            times=results['times'],
            dt_values=results['dt_values'],
            filename='time_stepping_3d.png'
        )

    def plot_3d_water_content(self,
                              points: jnp.ndarray,
                              tetrahedra: jnp.ndarray,
                              theta: jnp.ndarray,
                              title: str = "3D Water Content Distribution",
                              filename: Optional[str] = None) -> None:
        """Plot 3D water content distribution using tetrahedral mesh."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(15, 10))
        views = [(0,90), (0,0), (90,0), (45,45)]
        for idx, (az, el) in enumerate(views, 1):
            ax = fig.add_subplot(221+idx-1, projection='3d')
            self._plot_3d_view(ax, points, tetrahedra, theta, azim=az,
                               elev=el, title=f"{title} View {idx}")
        plt.tight_layout()
        if filename:
            plt.savefig(self.output_dir / filename, dpi=self.default_dpi,
                        bbox_inches='tight', facecolor='white')
            plt.close()

    def _plot_3d_view(self,
                      ax: plt.Axes,
                      points: jnp.ndarray,
                      tetrahedra: jnp.ndarray,
                      theta: jnp.ndarray,
                      azim: float = 45,
                      elev: float = 45,
                      title: str = "") -> None:
        """Helper for single 3D view."""
        points_np = np.array(points)
        tetra_np = np.array(tetrahedra)
        theta_np = np.array(theta)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for tet in tetra_np:
            verts = [points_np[tet[i]] for i in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]]
            avg = theta_np[tet].mean()
            pc = Poly3DCollection(verts, alpha=0.3)
            pc.set_facecolor(plt.cm.jet(avg/theta_np.max()))
            ax.add_collection3d(pc)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        norm = plt.Normalize(theta_np.min(), theta_np.max())
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Water Content (θ)')

    def plot_nitrogen_species_final(
        self,
        points: jnp.ndarray,
        triangles: jnp.ndarray,
        NH4_final: jnp.ndarray,
        NO2_final: jnp.ndarray,
        NO3_final: jnp.ndarray,
        time_step: Optional[str] = None,
        filename: Optional[str] = None
    ) -> None:
        """Plot final concentrations of NH₄⁺, NO₂⁻, and NO₃⁻ over the mesh."""
        import matplotlib.tri as tri
        # Extract coords
        x = np.array(points[:,0]); y = np.array(points[:,1])
        tri_obj = tri.Triangulation(x, y, np.array(triangles))
        fig, axes = plt.subplots(1,3, figsize=(18,6))
        species = [(NH4_final,'NH₄⁺',axes[0]),(NO2_final,'NO₂⁻',axes[1]),(NO3_final,'NO₃⁻',axes[2])]
        for conc,name,ax in species:
            data = np.array(conc)
            ctf = ax.tricontourf(tri_obj, data, levels=20, cmap='jet')
            ax.tricontour(tri_obj, data, levels=10, colors='black', linewidths=0.5, alpha=0.3)
            cbar = plt.colorbar(ctf, ax=ax); cbar.set_label(f'{name} [mg/L]')
            ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]')
            title = name + (' - '+time_step if time_step else '')
            ax.set_title(title)
            mn,mx,mean = float(jnp.min(conc)),float(jnp.max(conc)),float(jnp.mean(conc))
            ax.text(0.02,0.98,f'Min:{mn:.2f}\nMax:{mx:.2f}\nMean:{mean:.2f}',
                    transform=ax.transAxes, va='top', bbox=dict(facecolor='wheat',alpha=0.8))
            ax.set_aspect('equal','box')
        if filename:
            plt.tight_layout(); plt.savefig(self.output_dir/filename,dpi=self.default_dpi,bbox_inches='tight'); plt.close()
        total = jnp.sum(NH4_final)+jnp.sum(NO2_final)+jnp.sum(NO3_final)
        print(f"Total Nitrogen: {float(total):.2f} mg")


def plot_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    visualizer = ResultsVisualizer(output_dir)
    visualizer.plot_all_results(results)
