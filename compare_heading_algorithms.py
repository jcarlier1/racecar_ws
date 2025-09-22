#!/usr/bin/env python3
"""
Compare original and improved heading calculations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the shortest angular difference between two angles."""
    diff = angle2 - angle1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def compare_heading_algorithms():
    """Compare original and improved heading calculations."""
    
    # Load both datasets
    print("Loading datasets...")
    df_original = pd.read_csv("output/sequences/M-SOLO-SLOW-70-100/poses/poses_with_heading.csv")
    df_improved = pd.read_csv("output/sequences/M-SOLO-SLOW-70-100/poses/poses_with_improved_heading.csv")
    
    # Extract headings
    headings_original = df_original['heading_deg'].values
    headings_improved = df_improved['heading_deg_improved'].values
    
    # Calculate heading changes
    changes_original = np.array([abs(angle_difference(headings_original[i-1], headings_original[i])) 
                                for i in range(1, len(headings_original))])
    changes_improved = np.array([abs(angle_difference(headings_improved[i-1], headings_improved[i])) 
                                for i in range(1, len(headings_improved))])
    
    # Analysis
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Total points: {len(headings_original)}")
    
    # Large change analysis
    large_threshold = 90
    large_original = np.sum(changes_original > large_threshold)
    large_improved = np.sum(changes_improved > large_threshold)
    
    print(f"\nLarge heading changes (>{large_threshold}°):")
    print(f"  Original algorithm: {large_original}")
    print(f"  Improved algorithm: {large_improved}")
    print(f"  Reduction: {large_original - large_improved} ({100*(large_original-large_improved)/large_original:.1f}%)")
    
    # Smoothness analysis
    print(f"\nHeading change statistics:")
    print(f"  Original - Mean: {np.mean(changes_original):.2f}°, Std: {np.std(changes_original):.2f}°")
    print(f"  Improved - Mean: {np.mean(changes_improved):.2f}°, Std: {np.std(changes_improved):.2f}°")
    
    # Very large changes (likely GPS errors)
    very_large_threshold = 170
    very_large_original = np.sum(changes_original > very_large_threshold)
    very_large_improved = np.sum(changes_improved > very_large_threshold)
    
    print(f"\nVery large changes (>{very_large_threshold}°, likely GPS errors):")
    print(f"  Original: {very_large_original}")
    print(f"  Improved: {very_large_improved}")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Heading time series comparison (sample)
    sample_indices = np.arange(0, min(2000, len(headings_original)), 10)  # Every 10th point
    ax1.plot(sample_indices, headings_original[sample_indices], 'r-', alpha=0.7, linewidth=1, label='Original')
    ax1.plot(sample_indices, headings_improved[sample_indices], 'b-', alpha=0.7, linewidth=1, label='Improved')
    ax1.set_xlabel('Data Point Index')
    ax1.set_ylabel('Heading (degrees)')
    ax1.set_title('Heading Time Series Comparison (Sample)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Heading change magnitude comparison
    ax2.hist(changes_original, bins=50, alpha=0.5, color='red', label='Original', density=True)
    ax2.hist(changes_improved, bins=50, alpha=0.5, color='blue', label='Improved', density=True)
    ax2.axvline(large_threshold, color='black', linestyle='--', alpha=0.7, label=f'{large_threshold}° threshold')
    ax2.set_xlabel('Absolute Heading Change (degrees)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Heading Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Trajectory with problematic points highlighted
    # Find problematic points in original
    problematic_indices = np.where(changes_original > large_threshold)[0] + 1  # +1 because changes array is shifted
    
    ax3.plot(df_original['lon'], df_original['lat'], 'b-', alpha=0.3, linewidth=0.5, label='Trajectory')
    if len(problematic_indices) > 0:
        ax3.scatter(df_original.iloc[problematic_indices]['lon'], 
                   df_original.iloc[problematic_indices]['lat'],
                   color='red', s=15, alpha=0.7, label=f'Original large changes ({len(problematic_indices)})')
    
    # Find problematic points in improved
    improved_problematic = np.where(changes_improved > large_threshold)[0] + 1
    if len(improved_problematic) > 0:
        ax3.scatter(df_improved.iloc[improved_problematic]['lon'], 
                   df_improved.iloc[improved_problematic]['lat'],
                   color='orange', s=10, alpha=0.7, label=f'Improved large changes ({len(improved_problematic)})')
    
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Trajectory with Large Heading Changes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Plot 4: Heading difference between methods
    heading_differences = np.array([angle_difference(headings_original[i], headings_improved[i]) 
                                   for i in range(len(headings_original))])
    
    ax4.plot(range(len(heading_differences)), heading_differences, 'g-', alpha=0.7, linewidth=0.5)
    ax4.set_xlabel('Data Point Index')
    ax4.set_ylabel('Heading Difference (degrees)')
    ax4.set_title('Difference Between Original and Improved Headings')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
    
    # Add statistics to plot
    mean_diff = np.mean(np.abs(heading_differences))
    max_diff = np.max(np.abs(heading_differences))
    ax4.text(0.02, 0.98, f'Mean abs diff: {mean_diff:.1f}°\nMax abs diff: {max_diff:.1f}°', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/heading_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: plots/heading_comparison.png")
    
    # Additional analysis
    print(f"\nHeading difference analysis:")
    print(f"  Mean absolute difference: {mean_diff:.2f}°")
    print(f"  Max absolute difference: {max_diff:.2f}°")
    print(f"  95th percentile difference: {np.percentile(np.abs(heading_differences), 95):.2f}°")
    
    # Identify regions of largest improvement
    large_improvements = np.where(np.abs(heading_differences) > 30)[0]
    if len(large_improvements) > 0:
        print(f"\nRegions with large improvements (>30° difference): {len(large_improvements)} points")
        print("Sample indices:", large_improvements[:10])


if __name__ == "__main__":
    compare_heading_algorithms()