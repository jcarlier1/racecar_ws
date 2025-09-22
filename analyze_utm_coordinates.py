#!/usr/bin/env python3
"""
Visualization script to demonstrate the benefits of UTM coordinates for racetrack analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math


def calculate_distances_utm(df: pd.DataFrame) -> np.ndarray:
    """Calculate distances between consecutive points using UTM coordinates."""
    eastings = df['utm_easting'].values
    northings = df['utm_northing'].values
    
    distances = np.sqrt(np.diff(eastings)**2 + np.diff(northings)**2)
    return distances


def calculate_distances_latlon(df: pd.DataFrame) -> np.ndarray:
    """Calculate distances using Haversine formula on lat/lon coordinates."""
    lats = df['lat'].values
    lons = df['lon'].values
    
    distances = []
    R = 6371000  # Earth radius in meters
    
    for i in range(1, len(lats)):
        lat1, lon1 = math.radians(lats[i-1]), math.radians(lons[i-1])
        lat2, lon2 = math.radians(lats[i]), math.radians(lons[i])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        distances.append(distance)
    
    return np.array(distances)


def analyze_utm_coordinates(csv_file: str, output_plot: str = None):
    """Analyze and visualize the benefits of UTM coordinates."""
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Verify UTM columns exist
    utm_cols = ['utm_easting', 'utm_northing', 'utm_zone', 'utm_letter']
    missing_cols = [col for col in utm_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing UTM columns: {missing_cols}")
        return
    
    print(f"Analyzing {len(df)} trajectory points...")
    
    # Calculate distances using both methods
    print("Calculating distances using UTM coordinates...")
    distances_utm = calculate_distances_utm(df)
    
    print("Calculating distances using lat/lon coordinates...")
    distances_latlon = calculate_distances_latlon(df)
    
    # Calculate cumulative distances (track position)
    cumulative_utm = np.concatenate([[0], np.cumsum(distances_utm)])
    cumulative_latlon = np.concatenate([[0], np.cumsum(distances_latlon)])
    
    # Statistics
    total_distance_utm = cumulative_utm[-1]
    total_distance_latlon = cumulative_latlon[-1]
    
    print(f"\nDistance Analysis:")
    print(f"  Total track length (UTM): {total_distance_utm:.2f} m")
    print(f"  Total track length (Lat/Lon): {total_distance_latlon:.2f} m")
    print(f"  Difference: {abs(total_distance_utm - total_distance_latlon):.3f} m")
    print(f"  Mean step distance (UTM): {np.mean(distances_utm):.3f} m")
    print(f"  Mean step distance (Lat/Lon): {np.mean(distances_latlon):.3f} m")
    
    # Track dimensions
    easting_span = df['utm_easting'].max() - df['utm_easting'].min()
    northing_span = df['utm_northing'].max() - df['utm_northing'].min()
    
    print(f"\nTrack Dimensions (UTM):")
    print(f"  East-West span: {easting_span:.1f} m")
    print(f"  North-South span: {northing_span:.1f} m")
    print(f"  Approximate area: {easting_span * northing_span / 10000:.1f} hectares")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: UTM trajectory
    ax1.plot(df['utm_easting'], df['utm_northing'], 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(df['utm_easting'].iloc[0], df['utm_northing'].iloc[0], 
               color='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(df['utm_easting'].iloc[-1], df['utm_northing'].iloc[-1], 
               color='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_title('Racetrack Trajectory (UTM Coordinates)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    # Add scale bar
    scale_length = 100  # 100 meters
    scale_x = df['utm_easting'].min() + 0.05 * easting_span
    scale_y = df['utm_northing'].min() + 0.05 * northing_span
    ax1.plot([scale_x, scale_x + scale_length], [scale_y, scale_y], 'k-', linewidth=3)
    ax1.text(scale_x + scale_length/2, scale_y - 10, '100 m', 
             ha='center', va='top', fontweight='bold')
    
    # Plot 2: Lat/Lon trajectory (for comparison)
    ax2.plot(df['lon'], df['lat'], 'r-', linewidth=1, alpha=0.7)
    ax2.scatter(df['lon'].iloc[0], df['lat'].iloc[0], 
               color='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(df['lon'].iloc[-1], df['lat'].iloc[-1], 
               color='red', s=100, marker='s', label='End', zorder=5)
    ax2.set_xlabel('Longitude (degrees)')
    ax2.set_ylabel('Latitude (degrees)')
    ax2.set_title('Racetrack Trajectory (Lat/Lon Coordinates)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # Plot 3: Distance comparison
    sample_indices = np.arange(0, min(1000, len(distances_utm)))
    ax3.plot(sample_indices, distances_utm[sample_indices], 'b-', alpha=0.7, label='UTM')
    ax3.plot(sample_indices, distances_latlon[sample_indices], 'r-', alpha=0.7, label='Lat/Lon (Haversine)')
    ax3.set_xlabel('Data Point Index')
    ax3.set_ylabel('Step Distance (m)')
    ax3.set_title('Distance Calculation Comparison (First 1000 points)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distance vs heading
    if 'heading_deg' in df.columns:
        # Sample for performance
        sample_step = max(1, len(df) // 2000)
        sample_cum = cumulative_utm[::sample_step]
        sample_heading = df['heading_deg'].iloc[::sample_step]
        
        ax4.plot(sample_cum, sample_heading, 'g-', linewidth=1, alpha=0.7)
        ax4.set_xlabel('Cumulative Distance (m)')
        ax4.set_ylabel('Heading (degrees)')
        ax4.set_title('Heading vs Track Position')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 360)
    else:
        # Alternative: Speed analysis
        speeds_utm = distances_utm * 10  # Assuming 10 Hz data rate, convert to m/s
        ax4.hist(speeds_utm, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Speed (m/s)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Speed Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_plot}")
    else:
        plt.show()
    
    # Print UTM zone information
    utm_zone = df['utm_zone'].iloc[0]
    utm_letter = df['utm_letter'].iloc[0]
    print(f"\nUTM Zone Information:")
    print(f"  Zone: {utm_zone}{utm_letter}")
    print(f"  Central Meridian: {(utm_zone - 1) * 6 - 180 + 3:.0f}° longitude")
    print(f"  Hemisphere: {'Northern' if utm_letter == 'N' else 'Southern'}")
    
    # Benefits summary
    print(f"\nBenefits of UTM Coordinates:")
    print(f"  ✅ Direct distance calculations (no trigonometry)")
    print(f"  ✅ Cartesian coordinate system (easier geometry)")
    print(f"  ✅ Meter units (intuitive for racing applications)")
    print(f"  ✅ Minimal distortion over racetrack distances")
    print(f"  ✅ Suitable for mapping and trajectory analysis")
    print(f"  ✅ Compatible with GIS and mapping software")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize UTM coordinate benefits"
    )
    parser.add_argument("csv_file", help="CSV file with UTM coordinates")
    parser.add_argument("--output", "-o", help="Output plot file (optional)")
    
    args = parser.parse_args()
    
    analyze_utm_coordinates(args.csv_file, args.output)


if __name__ == "__main__":
    main()