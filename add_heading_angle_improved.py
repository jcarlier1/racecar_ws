#!/usr/bin/env python3
"""
Improved heading angle calculation script with robust smoothing and outlier detection.

This enhanced version addresses GPS noise and precision issues while preserving
legitimate sharp turns on racetracks.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from typing import Tuple, List, Optional
import math
import warnings
warnings.filterwarnings('ignore')


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in meters."""
    R = 6371000  # Earth's radius in meters
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat_rad = math.radians(lat2 - lat1)
    dlon_rad = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat_rad / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon_rad / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def bearing_between_points(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the initial bearing from point 1 to point 2."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360


def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate the shortest angular difference between two angles."""
    diff = angle2 - angle1
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


def detect_gps_stationary_points(df: pd.DataFrame, 
                                distance_threshold: float = 0.5,
                                window_size: int = 5) -> np.ndarray:
    """
    Detect points where GPS coordinates are essentially stationary.
    
    Args:
        df: DataFrame with lat, lon columns
        distance_threshold: Maximum distance to consider stationary (meters)
        window_size: Window size for checking stationary state
    
    Returns:
        Boolean array indicating stationary points
    """
    stationary = np.zeros(len(df), dtype=bool)
    
    for i in range(len(df)):
        # Check if this point is part of a stationary cluster
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(df), i + window_size // 2 + 1)
        
        # Calculate max distance from center point in window
        center_lat = df.iloc[i]['lat']
        center_lon = df.iloc[i]['lon']
        
        max_distance = 0
        for j in range(start_idx, end_idx):
            if j != i:
                dist = haversine_distance(center_lat, center_lon,
                                        df.iloc[j]['lat'], df.iloc[j]['lon'])
                max_distance = max(max_distance, dist)
        
        stationary[i] = max_distance < distance_threshold
    
    return stationary


def find_reliable_trajectory_segment(df: pd.DataFrame,
                                   center_idx: int,
                                   min_distance: float = 2.0,
                                   max_search: int = 100) -> Tuple[int, int]:
    """
    Find a reliable trajectory segment around a point for heading calculation.
    
    This looks for a segment where the vehicle has actually moved a significant
    distance, avoiding GPS noise artifacts.
    """
    center_lat = df.iloc[center_idx]['lat']
    center_lon = df.iloc[center_idx]['lon']
    
    # Search backwards for a reliable start point
    start_idx = center_idx
    for i in range(1, min(max_search + 1, center_idx + 1)):
        test_idx = center_idx - i
        test_lat = df.iloc[test_idx]['lat']
        test_lon = df.iloc[test_idx]['lon']
        
        distance = haversine_distance(center_lat, center_lon, test_lat, test_lon)
        if distance >= min_distance:
            start_idx = test_idx
            break
    
    # Search forwards for a reliable end point
    end_idx = center_idx
    for i in range(1, min(max_search + 1, len(df) - center_idx)):
        test_idx = center_idx + i
        test_lat = df.iloc[test_idx]['lat']
        test_lon = df.iloc[test_idx]['lon']
        
        distance = haversine_distance(center_lat, center_lon, test_lat, test_lon)
        if distance >= min_distance:
            end_idx = test_idx
            break
    
    return start_idx, end_idx


def median_filter_angles(angles: List[float], window_size: int = 5) -> List[float]:
    """
    Apply median filter to angles, handling wraparound properly.
    
    This is more robust to outliers than mean filtering.
    """
    if window_size < 3:
        return angles
    
    filtered = []
    half_window = window_size // 2
    
    for i in range(len(angles)):
        # Get window indices
        start_idx = max(0, i - half_window)
        end_idx = min(len(angles), i + half_window + 1)
        
        # Get angles in window
        window_angles = angles[start_idx:end_idx]
        
        if len(window_angles) >= 3:
            # Convert to vectors for median calculation
            vectors = [(math.cos(math.radians(a)), math.sin(math.radians(a))) 
                      for a in window_angles]
            
            # Find median vector (approximation)
            x_coords = [v[0] for v in vectors]
            y_coords = [v[1] for v in vectors]
            
            median_x = np.median(x_coords)
            median_y = np.median(y_coords)
            
            # Convert back to angle
            median_angle = math.degrees(math.atan2(median_y, median_x))
            filtered.append((median_angle + 360) % 360)
        else:
            filtered.append(angles[i])
    
    return filtered


def adaptive_smooth_angles(angles: List[float], 
                          distances: List[float],
                          max_change_threshold: float = 45.0,
                          distance_threshold: float = 2.0) -> List[float]:
    """
    Adaptively smooth angles based on distance moved and change magnitude.
    
    Args:
        angles: Raw heading angles
        distances: Distance moved between consecutive points
        max_change_threshold: Maximum allowable change per step (degrees)
        distance_threshold: Minimum distance to allow large changes
    
    Returns:
        Smoothed angles
    """
    if len(angles) < 2:
        return angles
    
    smoothed = [angles[0]]
    
    for i in range(1, len(angles)):
        prev_angle = smoothed[-1]
        curr_angle = angles[i]
        
        # Calculate angle change
        change = angle_difference(prev_angle, curr_angle)
        abs_change = abs(change)
        
        # Distance moved to this point
        dist_moved = distances[i-1] if i-1 < len(distances) else 0
        
        # Decide if this change is acceptable
        if abs_change <= max_change_threshold:
            # Small change - accept as is
            smoothed.append(curr_angle)
        elif dist_moved >= distance_threshold:
            # Large change but significant movement - likely legitimate turn
            smoothed.append(curr_angle)
        else:
            # Large change with small movement - likely GPS noise
            # Use a weighted average based on distance
            weight = min(1.0, dist_moved / distance_threshold)
            
            # Interpolate between previous angle and current angle
            if abs_change < 180:
                # Normal interpolation
                interpolated = prev_angle + weight * change
            else:
                # Handle wraparound case
                if change > 0:
                    interpolated = prev_angle + weight * (change - 360)
                else:
                    interpolated = prev_angle + weight * (change + 360)
            
            smoothed.append((interpolated + 360) % 360)
    
    return smoothed


def calculate_improved_heading_angles(df: pd.DataFrame,
                                    min_segment_distance: float = 2.0,
                                    stationary_threshold: float = 0.5,
                                    max_change_per_step: float = 45.0,
                                    median_window: int = 5) -> List[float]:
    """
    Calculate heading angles with improved noise handling.
    
    Args:
        df: DataFrame with lat, lon columns
        min_segment_distance: Minimum distance for reliable heading calculation
        stationary_threshold: Threshold for detecting stationary GPS points
        max_change_per_step: Maximum heading change per step (degrees)
        median_window: Window size for median filtering
    
    Returns:
        List of heading angles in degrees
    """
    if len(df) < 2:
        return [0.0] * len(df)
    
    print(f"Calculating headings for {len(df)} points...")
    
    # Step 1: Detect stationary points
    print("  Detecting GPS stationary points...")
    stationary_points = detect_gps_stationary_points(df, stationary_threshold)
    num_stationary = np.sum(stationary_points)
    print(f"  Found {num_stationary} stationary points ({100*num_stationary/len(df):.1f}%)")
    
    # Step 2: Calculate raw headings using reliable segments
    print("  Calculating raw headings...")
    raw_headings = []
    distances = []
    
    for i in range(len(df)):
        if i == 0:
            # First point
            start_idx, end_idx = find_reliable_trajectory_segment(df, i, min_segment_distance)
            if end_idx > i:
                heading = bearing_between_points(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[end_idx]['lat'], df.iloc[end_idx]['lon']
                )
                distance = haversine_distance(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[end_idx]['lat'], df.iloc[end_idx]['lon']
                )
            else:
                heading = 0.0
                distance = 0.0
        else:
            # Calculate distance from previous point
            distance = haversine_distance(
                df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
                df.iloc[i]['lat'], df.iloc[i]['lon']
            )
            distances.append(distance)
            
            if stationary_points[i]:
                # Stationary point - use previous heading
                heading = raw_headings[-1] if raw_headings else 0.0
            else:
                # Moving point - calculate heading
                start_idx, end_idx = find_reliable_trajectory_segment(df, i, min_segment_distance)
                
                if start_idx < i < end_idx:
                    # Use segment-based heading
                    heading = bearing_between_points(
                        df.iloc[start_idx]['lat'], df.iloc[start_idx]['lon'],
                        df.iloc[end_idx]['lat'], df.iloc[end_idx]['lon']
                    )
                elif start_idx < i:
                    # Use backward heading
                    heading = bearing_between_points(
                        df.iloc[start_idx]['lat'], df.iloc[start_idx]['lon'],
                        df.iloc[i]['lat'], df.iloc[i]['lon']
                    )
                elif end_idx > i:
                    # Use forward heading
                    heading = bearing_between_points(
                        df.iloc[i]['lat'], df.iloc[i]['lon'],
                        df.iloc[end_idx]['lat'], df.iloc[end_idx]['lon']
                    )
                else:
                    # Fallback to previous heading
                    heading = raw_headings[-1] if raw_headings else 0.0
        
        raw_headings.append(heading)
    
    # Step 3: Apply adaptive smoothing
    print("  Applying adaptive smoothing...")
    smoothed_headings = adaptive_smooth_angles(
        raw_headings, distances, max_change_per_step, min_segment_distance
    )
    
    # Step 4: Apply median filtering to remove remaining outliers
    print("  Applying median filtering...")
    final_headings = median_filter_angles(smoothed_headings, median_window)
    
    # Calculate improvement statistics
    if len(raw_headings) > 1:
        raw_changes = [abs(angle_difference(raw_headings[i-1], raw_headings[i])) 
                      for i in range(1, len(raw_headings))]
        final_changes = [abs(angle_difference(final_headings[i-1], final_headings[i])) 
                        for i in range(1, len(final_headings))]
        
        raw_large_changes = sum(1 for change in raw_changes if change > 90)
        final_large_changes = sum(1 for change in final_changes if change > 90)
        
        print(f"  Large changes reduced from {raw_large_changes} to {final_large_changes}")
        print(f"  Mean absolute change: {np.mean(raw_changes):.1f}° → {np.mean(final_changes):.1f}°")
    
    return final_headings


def add_improved_heading_to_csv(input_file: str, 
                               output_file: str,
                               min_segment_distance: float = 2.0,
                               stationary_threshold: float = 0.5,
                               max_change_per_step: float = 45.0,
                               median_window: int = 5):
    """Add improved heading angle column to CSV file."""
    
    print(f"Reading CSV file: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Validate required columns
    required_cols = ['lat', 'lon']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Processing {len(df)} trajectory points...")
    print(f"Parameters:")
    print(f"  Minimum segment distance: {min_segment_distance}m")
    print(f"  Stationary threshold: {stationary_threshold}m")
    print(f"  Max change per step: {max_change_per_step}°")
    print(f"  Median window: {median_window}")
    
    # Calculate improved heading angles
    headings = calculate_improved_heading_angles(
        df, min_segment_distance, stationary_threshold, 
        max_change_per_step, median_window
    )
    
    # Add heading column (replace if exists)
    df['heading_deg_improved'] = headings
    df['heading_deg_improved'] = df['heading_deg_improved'].round(2)
    
    print(f"Writing output CSV file: {output_file}")
    
    try:
        df.to_csv(output_file, index=False)
        print("Successfully added improved heading angles!")
        
        # Print statistics
        print(f"\nImproved heading angle statistics:")
        print(f"  Mean: {np.mean(headings):.2f}°")
        print(f"  Std:  {np.std(headings):.2f}°")
        print(f"  Min:  {np.min(headings):.2f}°")
        print(f"  Max:  {np.max(headings):.2f}°")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add improved heading angle column to racetrack pose CSV data"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--min-distance", "-d", type=float, default=2.0,
                       help="Minimum segment distance for reliable calculation (meters)")
    parser.add_argument("--stationary-threshold", "-s", type=float, default=0.5,
                       help="Threshold for detecting stationary points (meters)")
    parser.add_argument("--max-change", "-c", type=float, default=45.0,
                       help="Maximum heading change per step (degrees)")
    parser.add_argument("--median-window", "-w", type=int, default=5,
                       help="Median filter window size")
    
    args = parser.parse_args()
    
    add_improved_heading_to_csv(
        args.input_csv, args.output_csv,
        args.min_distance, args.stationary_threshold,
        args.max_change, args.median_window
    )


if __name__ == "__main__":
    main()