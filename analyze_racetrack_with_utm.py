#!/usr/bin/env python3
"""
Demonstrate common racetrack analysis tasks using UTM coordinates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def analyze_racetrack_with_utm(csv_file: str):
    """Demonstrate racetrack analysis using UTM coordinates."""
    
    print("=" * 60)
    print("RACETRACK ANALYSIS WITH UTM COORDINATES")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Extract UTM coordinates
    x = df['utm_easting'].values
    y = df['utm_northing'].values
    
    print(f"\n📊 TRACK ANALYSIS:")
    print(f"   Total track points: {len(df):,}")
    print(f"   UTM Zone: {df['utm_zone'].iloc[0]}{df['utm_letter'].iloc[0]}")
    
    # 1. Calculate distances between consecutive points (simple with UTM!)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    total_distance = np.sum(distances)
    
    print(f"\n🏁 DISTANCE CALCULATIONS:")
    print(f"   Total track length: {total_distance:.1f} meters")
    print(f"   Average step distance: {np.mean(distances):.2f} meters")
    print(f"   Max step distance: {np.max(distances):.2f} meters")
    print(f"   Min step distance: {np.min(distances):.2f} meters")
    
    # 2. Calculate speeds (if timestamp available)
    if 't_nsec' in df.columns:
        time_diffs = np.diff(df['t_nsec'].values) / 1e9  # Convert to seconds
        speeds_mps = distances / time_diffs
        speeds_kmh = speeds_mps * 3.6
        
        # Remove outliers for speed calculation
        valid_speeds = speeds_kmh[(speeds_kmh > 0) & (speeds_kmh < 200)]  # Reasonable speed range
        
        print(f"\n🏎️  SPEED ANALYSIS:")
        print(f"   Average speed: {np.mean(valid_speeds):.1f} km/h")
        print(f"   Max speed: {np.max(valid_speeds):.1f} km/h")
        print(f"   Min speed: {np.min(valid_speeds):.1f} km/h")
    
    # 3. Calculate curvature (easier with UTM coordinates)
    if len(x) > 2:
        # Calculate curvature using three-point method
        curvatures = []
        for i in range(1, len(x) - 1):
            # Three points
            x1, y1 = x[i-1], y[i-1]
            x2, y2 = x[i], y[i]
            x3, y3 = x[i+1], y[i+1]
            
            # Calculate curvature
            a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
            c = np.sqrt((x3-x1)**2 + (y3-y1)**2)
            
            # Area of triangle
            s = (a + b + c) / 2
            if s > 0 and s > a and s > b and s > c:
                area = np.sqrt(s * (s-a) * (s-b) * (s-c))
                if area > 0:
                    radius = (a * b * c) / (4 * area)
                    curvature = 1 / radius if radius > 0 else 0
                else:
                    curvature = 0
            else:
                curvature = 0
            
            curvatures.append(curvature)
        
        # Filter out extreme values
        curvatures = np.array(curvatures)
        valid_curvatures = curvatures[curvatures < 0.1]  # Remove extreme outliers
        
        print(f"\n🌊 CURVATURE ANALYSIS:")
        print(f"   Average curvature: {np.mean(valid_curvatures):.6f} 1/m")
        print(f"   Max curvature: {np.max(valid_curvatures):.6f} 1/m")
        print(f"   Tightest turn radius: {1/np.max(valid_curvatures):.1f} meters")
    
    # 4. Track centerline and boundaries (example)
    center_x = np.mean(x)
    center_y = np.mean(y)
    
    print(f"\n📍 TRACK CENTER:")
    print(f"   Center UTM coordinates: ({center_x:.1f}, {center_y:.1f})")
    print(f"   Center lat/lon: ({np.mean(df['lat']):.6f}, {np.mean(df['lon']):.6f})")
    
    # 5. Track sectoring (divide into segments)
    num_sectors = 4
    sector_size = len(df) // num_sectors
    
    print(f"\n🎯 TRACK SECTORS:")
    for sector in range(num_sectors):
        start_idx = sector * sector_size
        end_idx = min((sector + 1) * sector_size, len(df))
        
        sector_x = x[start_idx:end_idx]
        sector_y = y[start_idx:end_idx]
        
        if len(sector_x) > 1:
            sector_distances = np.sqrt(np.diff(sector_x)**2 + np.diff(sector_y)**2)
            sector_length = np.sum(sector_distances)
            
            print(f"   Sector {sector + 1}: {sector_length:.1f} meters")
    
    # 6. Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Track layout
    ax1.plot(x, y, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(x[0], y[0], color='green', s=50, label='Start', zorder=5)
    ax1.scatter(x[-1], y[-1], color='red', s=50, label='End', zorder=5)
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_title('Track Layout (UTM Coordinates)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Distance progression
    cumulative_distance = np.cumsum(np.concatenate(([0], distances)))
    ax2.plot(cumulative_distance, 'b-', linewidth=1)
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Cumulative Distance (m)')
    ax2.set_title('Distance Progression')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Step distances
    ax3.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Step Distance (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Step Distances')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speed profile (if available)
    if 't_nsec' in df.columns and len(valid_speeds) > 0:
        ax4.plot(valid_speeds[:min(len(valid_speeds), 5000)], 'r-', linewidth=0.5, alpha=0.7)
        ax4.set_xlabel('Data Point Index')
        ax4.set_ylabel('Speed (km/h)')
        ax4.set_title('Speed Profile')
        ax4.grid(True, alpha=0.3)
    else:
        # Plot headings if available
        if 'heading_deg' in df.columns:
            ax4.plot(df['heading_deg'][:5000], 'g-', linewidth=0.5, alpha=0.7)
            ax4.set_xlabel('Data Point Index')
            ax4.set_ylabel('Heading (degrees)')
            ax4.set_title('Heading Profile')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/utm_track_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Analysis plot saved to: plots/utm_track_analysis.png")
    
    # 7. Export useful data formats
    output_data = {
        'track_length_m': total_distance,
        'track_center_utm': (center_x, center_y),
        'track_dimensions_m': (np.max(x) - np.min(x), np.max(y) - np.min(y)),
        'utm_zone': f"{df['utm_zone'].iloc[0]}{df['utm_letter'].iloc[0]}",
        'coordinate_range': {
            'easting': (np.min(x), np.max(x)),
            'northing': (np.min(y), np.max(y))
        }
    }
    
    print(f"\n📁 SUMMARY DATA:")
    for key, value in output_data.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 UTM COORDINATE ADVANTAGES DEMONSTRATED:")
    print(f"   ✅ Simple distance calculations (Euclidean)")
    print(f"   ✅ Easy speed computation")
    print(f"   ✅ Straightforward curvature analysis")
    print(f"   ✅ Natural Cartesian plotting")
    print(f"   ✅ Sector analysis with linear indexing")
    print(f"   ✅ Track geometry calculations")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_racetrack_with_utm.py <csv_file>")
        sys.exit(1)
    
    analyze_racetrack_with_utm(sys.argv[1])