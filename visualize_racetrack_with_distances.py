#!/usr/bin/env python3
"""
Script to visualize racetrack coordinates and show distance-based image selection.

This script:
1. Plots the racetrack coordinates from the UTM data
2. Selects a random point and draws a 5m radius circle
3. Finds images at approximately 5m, 10m, and 20m distances
4. Displays the selected images alongside the plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import os
from PIL import Image
from scipy.spatial.distance import cdist

def load_poses_data(csv_path):
    """Load poses data from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two UTM points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_closest_image_at_distance(target_point, all_points, all_indices, target_distance, tolerance=2.0):
    """Find the closest image to the target distance from a reference point"""
    distances = [calculate_distance(target_point, point) for point in all_points]
    
    # Find indices where distance is close to target_distance
    valid_indices = []
    for i, dist in enumerate(distances):
        if abs(dist - target_distance) <= tolerance:
            valid_indices.append((i, dist))
    
    if not valid_indices:
        # If no exact match, find the closest one to target distance
        closest_idx = min(range(len(distances)), key=lambda i: abs(distances[i] - target_distance))
        return all_indices[closest_idx], distances[closest_idx]
    
    # Return the one closest to exact target distance
    best_idx, best_dist = min(valid_indices, key=lambda x: abs(x[1] - target_distance))
    return all_indices[best_idx], best_dist

def plot_racetrack_with_distances(df, sequence_path):
    """Main plotting function"""
    # Extract UTM coordinates
    easting = df['utm_easting'].values
    northing = df['utm_northing'].values
    
    # Create the plot with better layout
    fig = plt.figure(figsize=(20, 12))
    
    # Create main plot for racetrack (left side, takes up 60% width)
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3)
    
    # Plot the racetrack trajectory
    ax1.plot(easting, northing, 'b-', alpha=0.6, linewidth=1, label='Racetrack trajectory')
    ax1.scatter(easting, northing, c='lightblue', s=1, alpha=0.5)
    
    # Select a random point
    random_idx = random.randint(0, len(df) - 1)
    random_point = (easting[random_idx], northing[random_idx])
    
    # Draw thin circles: 5m and 25m - make them thin so they don't dominate the plot
    circle_5m = Circle(random_point, 5, fill=False, color='red', linewidth=0.8, linestyle='--', label='5m radius')
    ax1.add_patch(circle_5m)
    circle_25m = Circle(random_point, 25, fill=False, color='magenta', linewidth=0.8, linestyle='--', label='25m radius')
    ax1.add_patch(circle_25m)
    
    # Mark the random point (smaller marker)
    ax1.scatter(random_point[0], random_point[1], c='red', s=20, marker='*', 
               label=f'Random point (Image {random_idx})', zorder=5)
    
    # Prepare coordinate arrays for distance calculations
    all_points = list(zip(easting, northing))
    all_indices = list(range(len(df)))
    
    # Find images at target distances (include 25m)
    target_distances = [5, 10, 20, 25]
    colors = ['orange', 'green', 'purple', 'brown']
    markers = ['o', 's', '^', 'D']
    selected_images = [random_idx]  # Start with the random point
    selected_distances = [0]  # Distance from itself is 0

    for i, target_dist in enumerate(target_distances):
        closest_idx, actual_dist = find_closest_image_at_distance(
            random_point, all_points, all_indices, target_dist, tolerance=3.0
        )

        # Mark this point on the plot
        point = (easting[closest_idx], northing[closest_idx])
        ax1.scatter(point[0], point[1], c=colors[i], s=20, marker=markers[i], 
            label=f'~{target_dist}m away (Image {closest_idx}, actual: {actual_dist:.1f}m)', zorder=5)

        selected_images.append(closest_idx)
        selected_distances.append(actual_dist)
    
    # Set axis labels and title
    ax1.set_xlabel('UTM Easting (m)')
    ax1.set_ylabel('UTM Northing (m)')
    ax1.set_title('M-MULTI-SLOW-KAIST Racetrack with Distance Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Load and display images in a 2x2 (now up to 5) grid on the right side
    n_images = len(selected_images)
    # Extend positions to fit up to 5 images (we'll place the 5th image below)
    image_positions = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 2)]  # Grid positions for images
    
    for i, (img_idx, distance) in enumerate(zip(selected_images, selected_distances)):
        img_path = os.path.join(sequence_path, df.iloc[img_idx]['img_relpath'])
        
        if os.path.exists(img_path) and i < len(image_positions):
            try:
                img = Image.open(img_path)
                
                # Create subplot for each image
                row, col = image_positions[i]
                subplot_ax = plt.subplot2grid((3, 4), (row, col), colspan=1, rowspan=1)
                subplot_ax.imshow(img)
                subplot_ax.set_title(f'Image {img_idx}\nDistance: {distance:.1f}m', fontsize=10)
                subplot_ax.axis('off')
                
            except Exception as e:
                print(f"Could not load image {img_path}: {e}")
        else:
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
    

    plt.tight_layout(pad=1.0)
    return fig, selected_images, selected_distances

def main():
    """Main function"""
    # Paths
    csv_path = '/media/pragyan/Data/racecar_ws/output/sequences/M-MULTI-SLOW-KAIST/poses/poses_with_utm.csv'
    sequence_path = '/media/pragyan/Data/racecar_ws/output/sequences/M-MULTI-SLOW-KAIST'
    
    # Load data
    print("Loading poses data...")
    df = load_poses_data(csv_path)
    print(f"Loaded {len(df)} poses")
    
    # Create visualization
    print("Creating visualization...")
    fig, selected_images, selected_distances = plot_racetrack_with_distances(df, sequence_path)
    
    # Print summary
    print("\nSelected Images Summary:")
    print("Index\tDistance (m)\tImage Path")
    print("-" * 50)
    for i, (idx, dist) in enumerate(zip(selected_images, selected_distances)):
        img_path = df.iloc[idx]['img_relpath']
        print(f"{idx}\t{dist:.1f}\t\t{img_path}")
    
    # Save plot
    output_path = '/media/pragyan/Data/racecar_ws/plots/racetrack_distance_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()