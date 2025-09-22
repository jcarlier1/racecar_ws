#!/usr/bin/env python3
"""
Compare multiple racetrack datasets with UTM coordinates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_utm_datasets():
    """Compare multiple racetrack datasets using UTM coordinates."""
    
    print("=" * 70)
    print("MULTI-DATASET UTM COORDINATE COMPARISON")
    print("=" * 70)
    
    # Load datasets
    datasets = {
        'M-SOLO-SLOW-70-100': 'output/sequences/M-SOLO-SLOW-70-100/poses/poses_with_utm_zone11.csv',
        'M-MULTI-SLOW-KAIST': 'output/sequences/M-MULTI-SLOW-KAIST/poses/poses_with_utm.csv'
    }
    
    data = {}
    for name, file_path in datasets.items():
        try:
            df = pd.read_csv(file_path)
            data[name] = df
            print(f"✅ Loaded {name}: {len(df):,} points")
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            continue
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            continue
    
    if len(data) < 2:
        print("Need at least 2 datasets for comparison")
        return
    
    print(f"\n🗺️  UTM ZONE VERIFICATION:")
    for name, df in data.items():
        zones = df['utm_zone'].unique()
        letters = df['utm_letter'].unique()
        print(f"   {name}: Zone {zones[0]}{letters[0]}")
        if len(zones) > 1 or len(letters) > 1:
            print(f"     ⚠️  Multiple zones detected!")
    
    # Compare coordinate ranges
    print(f"\n📍 COORDINATE RANGE COMPARISON:")
    print(f"{'Dataset':<20} {'Easting Range (m)':<20} {'Northing Range (m)':<20} {'Dimensions (m)'}")
    print("-" * 80)
    
    for name, df in data.items():
        e_min, e_max = df['utm_easting'].min(), df['utm_easting'].max()
        n_min, n_max = df['utm_northing'].min(), df['utm_northing'].max()
        e_span = e_max - e_min
        n_span = n_max - n_min
        
        print(f"{name:<20} {e_min:.1f} - {e_max:.1f}     {n_min:.1f} - {n_max:.1f}     {e_span:.1f} × {n_span:.1f}")
    
    # Track statistics comparison
    print(f"\n📊 TRACK STATISTICS COMPARISON:")
    print(f"{'Dataset':<20} {'Points':<8} {'Length (km)':<12} {'Avg Speed (km/h)':<15}")
    print("-" * 65)
    
    for name, df in data.items():
        # Calculate track length
        x = df['utm_easting'].values
        y = df['utm_northing'].values
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        total_length = np.sum(distances) / 1000  # Convert to km
        
        # Calculate average speed if possible
        if 't_nsec' in df.columns and len(distances) > 0:
            time_diffs = np.diff(df['t_nsec'].values) / 1e9
            speeds_mps = distances / time_diffs
            speeds_kmh = speeds_mps * 3.6
            valid_speeds = speeds_kmh[(speeds_kmh > 0) & (speeds_kmh < 200)]
            avg_speed = np.mean(valid_speeds) if len(valid_speeds) > 0 else 0
        else:
            avg_speed = 0
        
        print(f"{name:<20} {len(df):<8} {total_length:<12.1f} {avg_speed:<15.1f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Track overlays
    for i, (name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        ax1.plot(df['utm_easting'], df['utm_northing'], 
                color=color, linewidth=1, alpha=0.7, label=name)
        
        # Mark start points
        ax1.scatter(df['utm_easting'].iloc[0], df['utm_northing'].iloc[0], 
                   color=color, s=50, marker='o', edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_title('Track Overlay Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Track centers and extents
    for i, (name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        
        # Track center
        center_x = df['utm_easting'].mean()
        center_y = df['utm_northing'].mean()
        
        # Track extents
        min_x, max_x = df['utm_easting'].min(), df['utm_easting'].max()
        min_y, max_y = df['utm_northing'].min(), df['utm_northing'].max()
        
        # Plot bounding box
        ax2.add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                   fill=False, edgecolor=color, linewidth=2, alpha=0.7))
        
        # Plot center
        ax2.scatter(center_x, center_y, color=color, s=100, marker='x', 
                   linewidth=3, label=f'{name} center')
    
    ax2.set_xlabel('UTM Easting (m)')
    ax2.set_ylabel('UTM Northing (m)')
    ax2.set_title('Track Centers and Extents')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Distance distributions
    for i, (name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        x = df['utm_easting'].values
        y = df['utm_northing'].values
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        
        ax3.hist(distances, bins=30, alpha=0.5, color=color, label=name, density=True)
    
    ax3.set_xlabel('Step Distance (m)')
    ax3.set_ylabel('Density')
    ax3.set_title('Step Distance Distributions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heading distributions (if available)
    if all('heading_deg' in df.columns for df in data.values()):
        for i, (name, df) in enumerate(data.items()):
            color = colors[i % len(colors)]
            ax4.hist(df['heading_deg'], bins=36, alpha=0.5, color=color, 
                    label=name, density=True, range=(0, 360))
        
        ax4.set_xlabel('Heading (degrees)')
        ax4.set_ylabel('Density')
        ax4.set_title('Heading Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Heading data not available\nfor all datasets', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Heading Analysis (N/A)')
    
    plt.tight_layout()
    plt.savefig('plots/utm_dataset_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Comparison plot saved to: plots/utm_dataset_comparison.png")
    
    # Calculate track similarity
    print(f"\n🔍 TRACK SIMILARITY ANALYSIS:")
    
    if len(data) >= 2:
        datasets_list = list(data.items())
        for i in range(len(datasets_list)):
            for j in range(i + 1, len(datasets_list)):
                name1, df1 = datasets_list[i]
                name2, df2 = datasets_list[j]
                
                # Calculate center distance
                center1 = (df1['utm_easting'].mean(), df1['utm_northing'].mean())
                center2 = (df2['utm_easting'].mean(), df2['utm_northing'].mean())
                center_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Calculate overlap of bounding boxes
                min_x1, max_x1 = df1['utm_easting'].min(), df1['utm_easting'].max()
                min_y1, max_y1 = df1['utm_northing'].min(), df1['utm_northing'].max()
                min_x2, max_x2 = df2['utm_easting'].min(), df2['utm_easting'].max()
                min_y2, max_y2 = df2['utm_northing'].min(), df2['utm_northing'].max()
                
                overlap_x = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
                overlap_y = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
                area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
                total_area = area1 + area2 - overlap_area
                overlap_ratio = overlap_area / total_area if total_area > 0 else 0
                
                print(f"   {name1} vs {name2}:")
                print(f"     Center distance: {center_distance:.1f} m")
                print(f"     Overlap ratio: {overlap_ratio:.3f} ({100*overlap_ratio:.1f}%)")
                
                if center_distance < 100 and overlap_ratio > 0.8:
                    print(f"     🎯 Same track, different runs")
                elif center_distance < 1000 and overlap_ratio > 0.3:
                    print(f"     📍 Similar location, possible track variants")
                else:
                    print(f"     🗺️  Different tracks/locations")
    
    print(f"\n💡 UTM COORDINATE BENEFITS FOR MULTI-DATASET ANALYSIS:")
    print(f"   ✅ Direct distance measurements between datasets")
    print(f"   ✅ Easy track overlay and comparison")
    print(f"   ✅ Geometric analysis without coordinate system conversions")
    print(f"   ✅ Consistent reference frame for all Las Vegas tracks")
    print(f"   ✅ Simplified multi-run analysis and statistics")
    
    print(f"\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    compare_utm_datasets()