#!/usr/bin/env python3
"""
Script to add UTM coordinates to a CSV file containing GPS coordinates.

This script converts lat/lon coordinates to UTM (Universal Transverse Mercator)
coordinates while preserving the precision of the original coordinates.

UTM coordinates provide:
- Local Cartesian coordinate system
- Meters as units (easier for distance calculations)
- Reduced distortion over local areas
- Better suited for mapping and trajectory analysis

Usage:
    python add_utm_coordinates.py input.csv output.csv [--utm-zone ZONE] [--precision DIGITS]
"""

import pandas as pd
import numpy as np
import argparse
import sys
import math
from typing import Tuple, Optional


def determine_utm_zone(longitude: float) -> int:
    """
    Determine the UTM zone number from longitude.
    
    Args:
        longitude: Longitude in degrees
    
    Returns:
        UTM zone number (1-60)
    """
    return int((longitude + 180) / 6) + 1


def lat_lon_to_utm(latitude: float, longitude: float, force_zone: Optional[int] = None) -> Tuple[float, float, int, str]:
    """
    Convert latitude/longitude to UTM coordinates.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        force_zone: Force a specific UTM zone (optional)
    
    Returns:
        Tuple of (easting, northing, zone_number, zone_letter)
    """
    # Constants for WGS84 ellipsoid
    a = 6378137.0  # Semi-major axis
    e2 = 0.00669437999014  # Eccentricity squared
    k0 = 0.9996  # Scale factor
    
    # Determine UTM zone
    if force_zone is not None:
        zone_number = force_zone
    else:
        zone_number = determine_utm_zone(longitude)
    
    # Determine zone letter (N for northern hemisphere, S for southern)
    zone_letter = 'N' if latitude >= 0 else 'S'
    
    # Convert to radians
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    
    # Central meridian for this UTM zone
    lon0_rad = math.radians((zone_number - 1) * 6 - 180 + 3)
    
    # Calculate UTM coordinates
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    T = math.tan(lat_rad)**2
    C = e2 * math.cos(lat_rad)**2 / (1 - e2)
    A = math.cos(lat_rad) * (lon_rad - lon0_rad)
    
    M = a * ((1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
             - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*lat_rad)
             + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*lat_rad)
             - (35*e2**3/3072) * math.sin(6*lat_rad))
    
    # Easting
    easting = (k0 * N * (A + (1-T+C)*A**3/6 + (5-18*T+T**2+72*C-58*e2)*A**5/120) + 500000)
    
    # Northing
    northing = k0 * (M + N*math.tan(lat_rad) * 
                    (A**2/2 + (5-T+9*C+4*C**2)*A**4/24 + 
                     (61-58*T+T**2+600*C-330*e2)*A**6/720))
    
    # Add false northing for southern hemisphere
    if latitude < 0:
        northing += 10000000
    
    return easting, northing, zone_number, zone_letter


def analyze_coordinate_precision(values: pd.Series) -> int:
    """
    Analyze the precision of coordinate values to determine appropriate decimal places.
    
    Args:
        values: Series of coordinate values
    
    Returns:
        Number of decimal places to preserve precision
    """
    # Convert to string and find max decimal places
    max_decimals = 0
    
    for value in values.head(1000):  # Sample first 1000 values for efficiency
        str_val = f"{value:.15f}".rstrip('0')  # Remove trailing zeros
        if '.' in str_val:
            decimals = len(str_val.split('.')[1])
            max_decimals = max(max_decimals, decimals)
    
    return min(max_decimals, 6)  # Cap at 6 decimal places for reasonable precision


def add_utm_coordinates_to_csv(input_file: str,
                              output_file: str,
                              utm_zone: Optional[int] = None,
                              precision: Optional[int] = None):
    """
    Add UTM coordinates to a CSV file containing lat/lon data.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        utm_zone: Force specific UTM zone (optional)
        precision: Number of decimal places for UTM coordinates (optional)
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
    
    print(f"Processing {len(df)} coordinate points...")
    
    # Analyze coordinate precision if not specified
    if precision is None:
        lat_precision = analyze_coordinate_precision(df['lat'])
        lon_precision = analyze_coordinate_precision(df['lon'])
        precision = max(lat_precision, lon_precision)
        print(f"Auto-detected coordinate precision: {precision} decimal places")
    
    # Determine UTM zone from data if not specified
    if utm_zone is None:
        # Use the center longitude of the dataset
        center_lon = (df['lon'].min() + df['lon'].max()) / 2
        utm_zone = determine_utm_zone(center_lon)
        print(f"Auto-detected UTM zone: {utm_zone}")
        
        # Verify all points would naturally fall in the same zone
        min_zone = determine_utm_zone(df['lon'].min())
        max_zone = determine_utm_zone(df['lon'].max())
        if min_zone != max_zone:
            print(f"Warning: Data spans multiple UTM zones ({min_zone} to {max_zone})")
            print(f"Using zone {utm_zone} for all points to ensure consistency")
    else:
        print(f"Using specified UTM zone: {utm_zone}")
    
    # Verify zone choice is reasonable for the data
    center_lon = (df['lon'].min() + df['lon'].max()) / 2
    natural_zone = determine_utm_zone(center_lon)
    if utm_zone != natural_zone:
        print(f"Note: Specified zone {utm_zone} differs from natural zone {natural_zone}")
        print(f"This may introduce additional distortion but ensures zone consistency")
    
    print(f"All {len(df)} points will be converted to UTM zone {utm_zone}N")
    
    # Convert coordinates
    print("Converting lat/lon to UTM coordinates...")
    
    eastings = []
    northings = []
    zone_numbers = []
    zone_letters = []
    
    # Process in batches for better performance
    batch_size = 1000
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        
        for j in range(i, batch_end):
            lat = df.iloc[j]['lat']
            lon = df.iloc[j]['lon']
            
            easting, northing, zone_num, zone_letter = lat_lon_to_utm(lat, lon, utm_zone)
            
            eastings.append(easting)
            northings.append(northing)
            zone_numbers.append(zone_num)
            zone_letters.append(zone_letter)
        
        if i > 0 and i % 5000 == 0:
            print(f"  Processed {i}/{len(df)} points...")
    
    # Add UTM columns to dataframe
    df['utm_easting'] = eastings
    df['utm_northing'] = northings
    df['utm_zone'] = zone_numbers
    df['utm_letter'] = zone_letters
    
    # Round UTM coordinates to appropriate precision
    # UTM coordinates are in meters, so precision should be reasonable
    utm_precision = min(precision, 3)  # Max 3 decimal places for UTM (mm precision)
    df['utm_easting'] = df['utm_easting'].round(utm_precision)
    df['utm_northing'] = df['utm_northing'].round(utm_precision)
    
    print(f"Writing output CSV file: {output_file}")
    
    try:
        df.to_csv(output_file, index=False)
        print("Successfully added UTM coordinates!")
        
        # Print statistics
        print(f"\nUTM Coordinate Statistics:")
        print(f"  Zone: {utm_zone}{zone_letters[0]}")
        print(f"  Easting range:  {df['utm_easting'].min():.3f} to {df['utm_easting'].max():.3f} m")
        print(f"  Northing range: {df['utm_northing'].min():.3f} to {df['utm_northing'].max():.3f} m")
        
        # Calculate track dimensions
        easting_span = df['utm_easting'].max() - df['utm_easting'].min()
        northing_span = df['utm_northing'].max() - df['utm_northing'].min()
        print(f"  Track dimensions: {easting_span:.1f} × {northing_span:.1f} meters")
        
        # Show sample of new columns
        print(f"\nSample UTM coordinates:")
        sample_cols = ['lat', 'lon', 'utm_easting', 'utm_northing', 'utm_zone', 'utm_letter']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head().to_string(index=False))
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Add UTM coordinates to CSV file with GPS coordinates"
    )
    parser.add_argument("input_csv", help="Input CSV file path")
    parser.add_argument("output_csv", help="Output CSV file path")
    parser.add_argument("--utm-zone", "-z", type=int,
                       help="Force specific UTM zone (1-60, auto-detect if not specified)")
    parser.add_argument("--precision", "-p", type=int,
                       help="Decimal places for UTM coordinates (auto-detect if not specified)")
    
    args = parser.parse_args()
    
    # Validate UTM zone if specified
    if args.utm_zone is not None and (args.utm_zone < 1 or args.utm_zone > 60):
        print("Error: UTM zone must be between 1 and 60")
        sys.exit(1)
    
    add_utm_coordinates_to_csv(args.input_csv, args.output_csv, 
                              args.utm_zone, args.precision)


if __name__ == "__main__":
    main()