#!/usr/bin/env python3
"""
Analyze large heading changes to understand their nature and distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def analyze_heading_changes(csv_file: str):
    """Analyze where large heading changes occur in the trajectory."""
    
    print("Loading data...")
    df = pd.read_csv(csv_file)
    
    # Calculate heading changes
    heading_changes = df['heading_deg'].diff()
    
    # Handle wraparound (e.g., 359° to 1° should be 2°, not 358°)
    heading_changes = np.where(heading_changes > 180, heading_changes - 360, heading_changes)
    heading_changes = np.where(heading_changes < -180, heading_changes + 360, heading_changes)
    
    # Calculate absolute changes
    abs_changes = np.abs(heading_changes)
    
    # Find large changes
    large_change_threshold = 90
    large_changes = abs_changes > large_change_threshold
    large_change_indices = np.where(large_changes)[0]
    
    print(f"\nHeading Change Analysis:")
    print(f"Total points: {len(df)}")
    print(f"Points with large changes (>{large_change_threshold}°): {np.sum(large_changes)}")
    print(f"Percentage: {100 * np.sum(large_changes) / len(df):.2f}%")
    
    if len(large_change_indices) > 0:
        print(f"\nStatistics for large changes:")
        large_change_values = abs_changes[large_changes]
        print(f"  Mean: {np.mean(large_change_values):.1f}°")
        print(f"  Median: {np.median(large_change_values):.1f}°")
        print(f"  Max: {np.max(large_change_values):.1f}°")
        
        # Check clustering of large changes
        print(f"\nDistribution of large changes:")
        # Calculate distances between consecutive large changes
        if len(large_change_indices) > 1:
            distances = np.diff(large_change_indices)
            print(f"  Average distance between large changes: {np.mean(distances):.1f} points")
            print(f"  Min distance: {np.min(distances)} points")
            print(f"  Max distance: {np.max(distances)} points")
            
            # Find clusters (consecutive or very close large changes)
            close_threshold = 10  # points
            clusters = []
            current_cluster = [large_change_indices[0]]
            
            for i in range(1, len(large_change_indices)):
                if large_change_indices[i] - large_change_indices[i-1] <= close_threshold:
                    current_cluster.append(large_change_indices[i])
                else:
                    if len(current_cluster) > 1:
                        clusters.append(current_cluster)
                    current_cluster = [large_change_indices[i]]
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            
            print(f"  Found {len(clusters)} clusters of large changes")
            for i, cluster in enumerate(clusters[:5]):  # Show first 5 clusters
                start_idx, end_idx = cluster[0], cluster[-1]
                print(f"    Cluster {i+1}: indices {start_idx}-{end_idx} ({len(cluster)} points)")
        
        # Examine some specific examples
        print(f"\nExamining specific large changes:")
        examples = large_change_indices[:10]  # First 10 examples
        
        for i, idx in enumerate(examples):
            if idx > 0 and idx < len(df) - 1:
                prev_heading = df.iloc[idx-1]['heading_deg']
                curr_heading = df.iloc[idx]['heading_deg'] 
                change = heading_changes[idx]
                
                # Calculate actual distance moved
                prev_lat, prev_lon = df.iloc[idx-1]['lat'], df.iloc[idx-1]['lon']
                curr_lat, curr_lon = df.iloc[idx]['lat'], df.iloc[idx]['lon']
                
                # Haversine distance
                R = 6371000  # Earth radius in meters
                dlat = math.radians(curr_lat - prev_lat)
                dlon = math.radians(curr_lon - prev_lon)
                a = (math.sin(dlat/2)**2 + math.cos(math.radians(prev_lat)) * 
                     math.cos(math.radians(curr_lat)) * math.sin(dlon/2)**2)
                distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
                
                print(f"    Example {i+1} (index {idx}):")
                print(f"      {prev_heading:.1f}° → {curr_heading:.1f}° (change: {change:.1f}°)")
                print(f"      Distance moved: {distance:.2f}m")
                print(f"      Lat: {prev_lat:.6f} → {curr_lat:.6f}")
                print(f"      Lon: {prev_lon:.6f} → {curr_lon:.6f}")
    
    # Plot analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Heading over time
    ax1.plot(df.index, df['heading_deg'], 'b-', alpha=0.7, linewidth=0.5)
    ax1.scatter(large_change_indices, df.iloc[large_change_indices]['heading_deg'], 
               color='red', s=20, alpha=0.7, label=f'Large changes (>{large_change_threshold}°)')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Heading (degrees)')
    ax1.set_title('Heading Angle Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heading changes histogram
    ax2.hist(abs_changes[1:], bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(large_change_threshold, color='red', linestyle='--', 
               label=f'Threshold ({large_change_threshold}°)')
    ax2.set_xlabel('Absolute Heading Change (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Heading Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Trajectory with large changes highlighted
    ax3.plot(df['lon'], df['lat'], 'b-', alpha=0.5, linewidth=0.5)
    if len(large_change_indices) > 0:
        ax3.scatter(df.iloc[large_change_indices]['lon'], 
                   df.iloc[large_change_indices]['lat'],
                   color='red', s=20, alpha=0.7, label='Large changes')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Trajectory with Large Heading Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Heading change magnitude over time
    ax4.plot(df.index[1:], abs_changes[1:], 'g-', alpha=0.7, linewidth=0.5)
    ax4.axhline(large_change_threshold, color='red', linestyle='--', 
               label=f'Threshold ({large_change_threshold}°)')
    ax4.set_xlabel('Data Point Index')
    ax4.set_ylabel('Absolute Heading Change (degrees)')
    ax4.set_title('Heading Change Magnitude Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/heading_change_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plot saved to: plots/heading_change_analysis.png")
    
    return df, heading_changes, large_change_indices


if __name__ == "__main__":
    csv_file = "output/sequences/M-SOLO-SLOW-70-100/poses/poses_with_heading.csv"
    analyze_heading_changes(csv_file)