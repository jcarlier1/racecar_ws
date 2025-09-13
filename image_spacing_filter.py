#!/usr/bin/env python3
"""
Program to process GPS coordinates from a CSV file and calculate the number of images
that would remain if spaced at least 0.25m apart.

Usage:
    python image_spacing_filter.py <csv_file_path> [min_distance_m]
"""

import pandas as pd
import numpy as np
import argparse
import sys
from math import radians, cos, sin, asin, sqrt


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees) in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r


def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calculate 3D distance between two GPS points including altitude.
    """
    # Horizontal distance using Haversine formula
    horizontal_dist = haversine_distance(lat1, lon1, lat2, lon2)
    
    # Vertical distance
    vertical_dist = abs(alt2 - alt1)
    
    # 3D distance using Pythagorean theorem
    return sqrt(horizontal_dist**2 + vertical_dist**2)


def filter_images_by_distance(csv_file_path, min_distance_m=0.25):
    """
    Filter images to maintain minimum distance spacing.
    
    Args:
        csv_file_path: Path to the CSV file with GPS coordinates
        min_distance_m: Minimum distance in meters between selected images
    
    Returns:
        tuple: (filtered_df, original_count, filtered_count, total_distance)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None, 0, 0, 0
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, 0, 0, 0
    
    # Validate required columns
    required_columns = ['camera_id', 'img_relpath', 't_nsec', 'lat', 'lon', 'alt']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, 0, 0, 0
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('t_nsec').reset_index(drop=True)
    
    original_count = len(df)
    if original_count == 0:
        print("Error: No data found in CSV file.")
        return None, 0, 0, 0
    
    # Initialize filtered list with first image
    filtered_indices = [0]
    total_distance = 0
    
    # Last selected position
    last_lat = df.iloc[0]['lat']
    last_lon = df.iloc[0]['lon']
    last_alt = df.iloc[0]['alt']
    
    # Process remaining images
    for i in range(1, len(df)):
        current_lat = df.iloc[i]['lat']
        current_lon = df.iloc[i]['lon']
        current_alt = df.iloc[i]['alt']
        
        # Calculate distance from last selected image
        distance = calculate_3d_distance(
            last_lat, last_lon, last_alt,
            current_lat, current_lon, current_alt
        )
        
        # If distance is greater than minimum threshold, select this image
        if distance >= min_distance_m:
            filtered_indices.append(i)
            total_distance += distance
            
            # Update last selected position
            last_lat = current_lat
            last_lon = current_lon
            last_alt = current_alt
    
    filtered_df = df.iloc[filtered_indices].copy()
    filtered_count = len(filtered_df)
    
    return filtered_df, original_count, filtered_count, total_distance


def main():
    parser = argparse.ArgumentParser(
        description="Filter images by minimum distance spacing"
    )
    parser.add_argument(
        "csv_file", 
        help="Path to CSV file with GPS coordinates"
    )
    parser.add_argument(
        "-d", "--min-distance", 
        type=float, 
        default=0.25,
        help="Minimum distance in meters between images (default: 0.25)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path for filtered results (optional)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Process the CSV file
    filtered_df, original_count, filtered_count, total_distance = filter_images_by_distance(
        args.csv_file, args.min_distance
    )
    
    if filtered_df is None:
        sys.exit(1)
    
    # Print results
    print(f"Original number of images: {original_count}")
    print(f"Filtered number of images: {filtered_count}")
    print(f"Reduction: {original_count - filtered_count} images ({(original_count - filtered_count)/original_count*100:.1f}%)")
    print(f"Minimum distance threshold: {args.min_distance}m")
    print(f"Total distance covered: {total_distance:.2f}m")
    
    if args.verbose:
        print(f"\nAverage distance between selected images: {total_distance/(filtered_count-1):.2f}m")
        print(f"First image timestamp: {filtered_df.iloc[0]['t_nsec']}")
        print(f"Last image timestamp: {filtered_df.iloc[-1]['t_nsec']}")
        
        # Calculate time span
        time_span_ns = filtered_df.iloc[-1]['t_nsec'] - filtered_df.iloc[0]['t_nsec']
        time_span_s = time_span_ns / 1e9
        print(f"Time span: {time_span_s:.2f} seconds")
        
        if time_span_s > 0:
            print(f"Average speed: {total_distance/time_span_s:.2f} m/s ({total_distance/time_span_s*3.6:.2f} km/h)")
    
    # Save filtered results if output file specified
    if args.output:
        try:
            filtered_df.to_csv(args.output, index=False)
            print(f"\nFiltered results saved to: {args.output}")
        except Exception as e:
            print(f"Error saving output file: {e}")


if __name__ == "__main__":
    main()
