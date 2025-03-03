#src/utils/plotting.py
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
        
        # Set up plotting style with standard matplotlib
        plt.style.use('classic')  # Using classic style instead of seaborn
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
        
        # Create triangulation
        tri = Triangulation(points[:, 0], points[:, 1], triangles[:, 0:3])
        
        # Create contour plot with improved style
        contour = ax.tricontourf(tri, np.array(theta), cmap='jet', levels=10)
        
        # Add colorbar and labels with better formatting
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Water Content (θ)', fontsize=6)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(title, fontsize=8, pad=10)
        ax.grid(False) 
        
        # Save plot
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
        
        # Plot iteration counts
        ax1.plot(iterations, 'b-', linewidth=1.5)
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Iterations', fontsize=10)
        ax1.set_title('Iteration History', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot errors
        ax2.semilogy(errors, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time Step', fontsize=10)
        ax2.set_ylabel('Error', fontsize=10)
        ax2.set_title('Error History', fontsize=82)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
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
        ax.set_ylabel('Time Step Size (dt)', fontsize=6)
        ax.set_title('Adaptive Time Stepping History', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
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
      import matplotlib.pyplot as plt
      from matplotlib.tri import Triangulation
  
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
  
      # ✅ Pass 'points' and 'triangles' explicitly
      tri = Triangulation(points[:, 0], points[:, 1], triangles)
  
      # Plot final water content
      tricontour1 = ax1.tricontourf(tri, final_theta, cmap='jet', levels=8)
      ax1.set_xlabel('X', fontsize=5)
      ax1.set_ylabel('Y', fontsize=5)
      ax1.set_title('Final Water Content', fontsize=5)
      cbar1 = fig.colorbar(tricontour1, ax=ax1)
      cbar1.set_label('Water Content (θ)', fontsize=4)
      cbar1.ax.tick_params(labelsize=4)
      ax1.set_xlim([-0.5, 0.5])
      ax1.set_ylim([1.5, 2])
      ax1.tick_params(axis='both', which='major', labelsize=4)  # Reduce tick font size
  
      # Plot final solute concentration
      tricontour2 = ax2.tricontourf(tri, final_solute, cmap='jet', levels=8)
      ax2.set_xlabel('X', fontsize=5)
      ax2.set_ylabel('Y', fontsize=5)
      ax2.set_title('Final Solute Concentration', fontsize=5)
      cbar2 = fig.colorbar(tricontour2, ax=ax2)
      cbar2.set_label('Solute Concentration', fontsize=4)
      cbar2.ax.tick_params(labelsize=4)
      ax2.set_xlim([-0.5, 0.5])
      ax2.set_ylim([1.5, 2])
      ax2.tick_params(axis='both', which='major', labelsize=4)  # Reduce tick font size
  
      plt.tight_layout()
      plt.savefig(filename, dpi=300)
      plt.close()


    
    
    def plot_3d_results(self, results: Dict[str, Any]) -> None:
        """
        Create all plots for 3D simulation results.

        Args:
            results: Dictionary containing simulation results
        """
        # Plot 3D water content distribution
        self.plot_3d_water_content(
            points=results['points'],
            tetrahedra=results['tetrahedra'],  # Note: This will be tetrahedra for 3D
            theta=results['final_theta'],
            filename='final_water_content_3d.png'
        )

        # Plot convergence history (reuse existing method)
        self.plot_convergence_history(
            iterations=results['iterations'],
            errors=results['errors'],
            filename='convergence_history_3d.png'
        )

        # Plot time stepping (reuse existing method)
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
        """
        Plot 3D water content distribution using tetrahedral mesh.

        Args:
            points: Node coordinates (Nx3 array)
            tetrahedra: Tetrahedral element connectivity
            theta: Water content values at nodes
            title: Plot title
            filename: Output filename
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(15, 10))

        # Create multiple views of the 3D distribution
        # XY view (top)
        ax1 = fig.add_subplot(221, projection='3d')
        self._plot_3d_view(ax1, points, tetrahedra, theta, azim=0, elev=90,
                           title=f"{title}\nTop View (XY)")

        # XZ view (front)
        ax2 = fig.add_subplot(222, projection='3d')
        self._plot_3d_view(ax2, points, tetrahedra, theta, azim=0, elev=0,
                           title=f"{title}\nFront View (XZ)")

        # YZ view (side)
        ax3 = fig.add_subplot(223, projection='3d')
        self._plot_3d_view(ax3, points, tetrahedra, theta, azim=90, elev=0,
                           title=f"{title}\nSide View (YZ)")

        # Isometric view
        ax4 = fig.add_subplot(224, projection='3d')
        self._plot_3d_view(ax4, points, tetrahedra, theta, azim=45, elev=45,
                           title=f"{title}\nIsometric View")

        plt.tight_layout()

        # Save plot
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
        """
        Helper function to create a single 3D view.

        Args:
            ax: Matplotlib axes
            points: Node coordinates
            tetrahedra: Tetrahedral element connectivity
            theta: Water content values
            azim: Azimuthal viewing angle
            elev: Elevation viewing angle
            title: Plot title
        """
        # Convert to numpy for matplotlib compatibility
        points_np = np.array(points)
        tetrahedra_np = np.array(tetrahedra)
        theta_np = np.array(theta)

        # Create the 3D tetrahedral plot
        # Plot partially transparent tetrahedral elements
        for tet in tetrahedra_np:
            tet_points = points_np[tet]
            x = tet_points[:, 0]
            y = tet_points[:, 1]
            z = tet_points[:, 2]

            # Calculate average water content for this element
            avg_theta = np.mean(theta_np[tet])

            # Plot tetrahedral faces with alpha based on water content
            verts = [
                [tet_points[0], tet_points[1], tet_points[2]],
                [tet_points[0], tet_points[1], tet_points[3]],
                [tet_points[0], tet_points[2], tet_points[3]],
                [tet_points[1], tet_points[2], tet_points[3]]
            ]

            # Create polygon collection
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            pc = Poly3DCollection(verts, alpha=0.3)
            pc.set_facecolor(plt.cm.jet(avg_theta/np.max(theta_np)))
            ax.add_collection3d(pc)

        # Set axis properties
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=8, pad=10)

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Add colorbar
        norm = plt.Normalize(vmin=np.min(theta_np), vmax=np.max(theta_np))
        sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Water Content (θ)')

        
def plot_results(results: Dict[str, Any], output_dir: str = "results") -> None:
    """Convenience function to plot all results."""
    visualizer = ResultsVisualizer(output_dir)
    visualizer.plot_all_results(results)




