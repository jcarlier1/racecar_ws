#!/usr/bin/env python3
"""
Visualization script to verify heading angle calculations.
Creates a plot showing the trajectory with heading vectors.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


def plot_trajectory_with_headings(csv_file: str, 
                                output_plot: str = None,
                                max_points: int = 1000,
                                vector_scale: float = 0.00005):
    """
    Plot trajectory with heading vectors to verify calculations.
    
    Args:
        csv_file: Path to CSV file with heading data
        output_plot: Path to save plot (optional)
        max_points: Maximum number of points to plot (for performance)
        vector_scale: Scale factor for heading vectors
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Sample points if dataset is large
    if len(df) > max_points:
        step = len(df) // max_points
        df_plot = df.iloc[::step].copy()
        print(f"Sampling {len(df_plot)} points from {len(df)} total points")
    else:
        df_plot = df.copy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory with heading vectors
    ax1.scatter(df_plot['lon'], df_plot['lat'], c=df_plot['heading_deg'], 
               cmap='hsv', s=20, alpha=0.7)
    
    # Add heading vectors for every N points
    vector_step = max(1, len(df_plot) // 50)  # Show ~50 vectors
    for i in range(0, len(df_plot), vector_step):
        row = df_plot.iloc[i]
        heading_rad = math.radians(row['heading_deg'])
        
        # Calculate vector components
        dx = vector_scale * math.sin(heading_rad)
        dy = vector_scale * math.cos(heading_rad)
        
        ax1.arrow(row['lon'], row['lat'], dx, dy,
                 head_width=vector_scale*0.3, head_length=vector_scale*0.2,
                 fc='red', ec='red', alpha=0.8, linewidth=1)
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Trajectory with Heading Vectors')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Add colorbar
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label('Heading (degrees)')
    
    # Plot 2: Heading angle over time/sequence
    ax2.plot(df_plot.index, df_plot['heading_deg'], 'b-', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Heading (degrees)')
    ax2.set_title('Heading Angle vs Time/Sequence')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 360)
    
    plt.tight_layout()
    
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_plot}")
    else:
        plt.show()
    
    # Print some analysis
    print(f"\nTrajectory Analysis:")
    print(f"  Latitude range:  {df['lat'].min():.6f} to {df['lat'].max():.6f}")
    print(f"  Longitude range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
    print(f"  Heading range:   {df['heading_deg'].min():.1f}° to {df['heading_deg'].max():.1f}°")
    
    # Calculate heading change rate
    heading_diff = np.diff(df['heading_deg'])
    # Handle wraparound
    heading_diff = np.where(heading_diff > 180, heading_diff - 360, heading_diff)
    heading_diff = np.where(heading_diff < -180, heading_diff + 360, heading_diff)
    
    print(f"  Mean heading change per step: {np.mean(np.abs(heading_diff)):.2f}°")
    print(f"  Max heading change per step:  {np.max(np.abs(heading_diff)):.2f}°")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectory with heading angles"
    )
    parser.add_argument("csv_file", help="CSV file with heading data")
    parser.add_argument("--output", "-o", help="Output plot file (optional)")
    parser.add_argument("--max-points", "-m", type=int, default=1000,
                       help="Maximum points to plot (default: 1000)")
    parser.add_argument("--vector-scale", "-s", type=float, default=0.00005,
                       help="Scale factor for heading vectors (default: 0.00005)")
    
    args = parser.parse_args()
    
    plot_trajectory_with_headings(args.csv_file, args.output, 
                                args.max_points, args.vector_scale)


if __name__ == "__main__":
    main()