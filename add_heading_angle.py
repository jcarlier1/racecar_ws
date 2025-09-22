#!/usr/bin/env python3
"""
Script to add heading angle column to racetrack pose CSV data.

This script processes GPS coordinates from a racetrack and calculates robust
heading angles accounting for:
- GPS noise and precision limitations
- Multiple laps with repeated coordinates
- Smooth trajectory estimation
- Proper angle wraparound handling

Usage:
    python add_heading_angle.py input.csv output.csv [--window-size N] [--min-distance M]
"""

import pandas as pd
import numpy as np
import argparse
import sys
from typing import Tuple, List
import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Distance in meters
    """
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
    """
    Calculate the initial bearing from point 1 to point 2.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in degrees
        lat2, lon2: Latitude and longitude of second point in degrees
    
    Returns:
        Bearing in degrees (0-360, where 0 is North)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360 degrees
    return (bearing_deg + 360) % 360


def smooth_angles(angles: List[float], window_size: int = 5) -> List[float]:
    """
    Smooth a sequence of angles accounting for wraparound at 0/360 degrees.
    
    Args:
        angles: List of angles in degrees
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed angles in degrees
    """
    if len(angles) < 2:
        return angles
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(angles)):
        # Get window indices
        start_idx = max(0, i - half_window)
        end_idx = min(len(angles), i + half_window + 1)
        
        # Get angles in window
        window_angles = angles[start_idx:end_idx]
        
        # Convert to unit vectors to handle wraparound
        sin_sum = sum(math.sin(math.radians(angle)) for angle in window_angles)
        cos_sum = sum(math.cos(math.radians(angle)) for angle in window_angles)
        
        # Calculate average angle
        avg_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        smoothed.append((avg_angle + 360) % 360)
    
    return smoothed


def find_valid_trajectory_points(df: pd.DataFrame, 
                                current_idx: int, 
                                min_distance: float = 1.0,
                                max_look_ahead: int = 50) -> Tuple[int, int]:
    """
    Find previous and next points that are far enough from current point for reliable heading calculation.
    
    Args:
        df: DataFrame with lat, lon columns
        current_idx: Current point index
        min_distance: Minimum distance in meters for reliable calculation
        max_look_ahead: Maximum number of points to look ahead/behind
    
    Returns:
        Tuple of (previous_valid_idx, next_valid_idx)
    """
    current_lat = df.iloc[current_idx]['lat']
    current_lon = df.iloc[current_idx]['lon']
    
    # Find previous valid point
    prev_idx = current_idx
    for i in range(1, min(max_look_ahead + 1, current_idx + 1)):
        test_idx = current_idx - i
        test_lat = df.iloc[test_idx]['lat']
        test_lon = df.iloc[test_idx]['lon']
        
        distance = haversine_distance(current_lat, current_lon, test_lat, test_lon)
        if distance >= min_distance:
            prev_idx = test_idx
            break
    
    # Find next valid point
    next_idx = current_idx
    for i in range(1, min(max_look_ahead + 1, len(df) - current_idx)):
        test_idx = current_idx + i
        test_lat = df.iloc[test_idx]['lat']
        test_lon = df.iloc[test_idx]['lon']
        
        distance = haversine_distance(current_lat, current_lon, test_lat, test_lon)
        if distance >= min_distance:
            next_idx = test_idx
            break
    
    return prev_idx, next_idx


def calculate_heading_angles(df: pd.DataFrame, 
                           window_size: int = 5, 
                           min_distance: float = 1.0) -> List[float]:
    """
    Calculate heading angles for each point in the trajectory.
    
    Args:
        df: DataFrame with 'lat' and 'lon' columns
        window_size: Size of smoothing window
        min_distance: Minimum distance for reliable heading calculation
    
    Returns:
        List of heading angles in degrees
    """
    if len(df) < 2:
        return [0.0] * len(df)
    
    raw_headings = []
    
    for i in range(len(df)):
        if i == 0:
            # First point: use heading to next valid point
            _, next_idx = find_valid_trajectory_points(df, i, min_distance)
            if next_idx > i:
                heading = bearing_between_points(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[next_idx]['lat'], df.iloc[next_idx]['lon']
                )
            else:
                heading = 0.0
                
        elif i == len(df) - 1:
            # Last point: use heading from previous valid point
            prev_idx, _ = find_valid_trajectory_points(df, i, min_distance)
            if prev_idx < i:
                heading = bearing_between_points(
                    df.iloc[prev_idx]['lat'], df.iloc[prev_idx]['lon'],
                    df.iloc[i]['lat'], df.iloc[i]['lon']
                )
            else:
                heading = raw_headings[-1] if raw_headings else 0.0
                
        else:
            # Middle points: use average of backward and forward headings
            prev_idx, next_idx = find_valid_trajectory_points(df, i, min_distance)
            
            # Calculate backward heading (from previous to current)
            if prev_idx < i:
                back_heading = bearing_between_points(
                    df.iloc[prev_idx]['lat'], df.iloc[prev_idx]['lon'],
                    df.iloc[i]['lat'], df.iloc[i]['lon']
                )
            else:
                back_heading = raw_headings[-1] if raw_headings else 0.0
            
            # Calculate forward heading (from current to next)
            if next_idx > i:
                forward_heading = bearing_between_points(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[next_idx]['lat'], df.iloc[next_idx]['lon']
                )
            else:
                forward_heading = back_heading
            
            # Average the headings, accounting for wraparound
            sin_avg = (math.sin(math.radians(back_heading)) + 
                      math.sin(math.radians(forward_heading))) / 2
            cos_avg = (math.cos(math.radians(back_heading)) + 
                      math.cos(math.radians(forward_heading))) / 2
            
            heading = math.degrees(math.atan2(sin_avg, cos_avg))
            heading = (heading + 360) % 360
        
        raw_headings.append(heading)
    
    # Apply smoothing
    if window_size > 1:
        smoothed_headings = smooth_angles(raw_headings, window_size)
        return smoothed_headings
    else:
        return raw_headings


def add_heading_to_csv(input_file: str, 
                      output_file: str,
                      window_size: int = 5,
                      min_distance: float = 1.0):
    """
    Add heading angle column to CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        window_size: Size of smoothing window
        min_distance: Minimum distance for reliable heading calculation
    """
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
    print(f"Using window size: {window_size}, minimum distance: {min_distance}m")
    
    # Calculate heading angles
    headings = calculate_heading_angles(df, window_size, min_distance)
    
    # Add heading column
    df['heading_deg'] = headings
    
    # Round to reasonable precision
    df['heading_deg'] = df['heading_deg'].round(2)
    
    print(f"Writing output CSV file: {output_file}")
    
    try:
        df.to_csv(output_file, index=False)
        print("Successfully added heading angles!")
        
        # Print some statistics
        print(f"\nHeading angle statistics:")
        print(f"  Mean: {np.mean(headings):.2f}°")
        print(f"  Std:  {np.std(headings):.2f}°")
        print(f"  Min:  {np.min(headings):.2f}°")
        print(f"  Max:  {np.max(headings):.2f}°")
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add heading angle column to racetrack pose CSV data"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--window-size", "-w", type=int, default=5,
                       help="Smoothing window size (default: 5)")
    parser.add_argument("--min-distance", "-d", type=float, default=1.0,
                       help="Minimum distance for reliable heading calculation in meters (default: 1.0)")
    
    args = parser.parse_args()
    
    add_heading_to_csv(args.input_csv, args.output_csv, 
                      args.window_size, args.min_distance)


if __name__ == "__main__":
    main()