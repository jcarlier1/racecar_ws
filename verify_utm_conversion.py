#!/usr/bin/env python3
"""
Verify UTM coordinate conversion and demonstrate zone consistency.
"""

import pandas as pd
import numpy as np
import math


def verify_utm_conversion(csv_file: str):
    """Verify UTM coordinate conversion and zone consistency."""
    
    print("=" * 60)
    print("UTM COORDINATE VERIFICATION")
    print("=" * 60)
    
    # Load data
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Basic info
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total points: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check UTM zone consistency
    print(f"\n🗺️  UTM ZONE ANALYSIS:")
    utm_zones = df['utm_zone'].unique()
    utm_letters = df['utm_letter'].unique()
    
    print(f"   UTM zones found: {utm_zones}")
    print(f"   UTM letters found: {utm_letters}")
    
    if len(utm_zones) == 1 and len(utm_letters) == 1:
        print(f"   ✅ All points in same UTM zone: {utm_zones[0]}{utm_letters[0]}")
    else:
        print(f"   ⚠️  Multiple UTM zones detected - this may cause coordinate issues")
    
    # Coordinate ranges
    print(f"\n📍 COORDINATE RANGES:")
    print(f"   Latitude:  {df['lat'].min():.6f}° to {df['lat'].max():.6f}°")
    print(f"   Longitude: {df['lon'].min():.6f}° to {df['lon'].max():.6f}°")
    print(f"   Easting:   {df['utm_easting'].min():.3f} to {df['utm_easting'].max():.3f} m")
    print(f"   Northing:  {df['utm_northing'].min():.3f} to {df['utm_northing'].max():.3f} m")
    
    # Track dimensions
    easting_span = df['utm_easting'].max() - df['utm_easting'].min()
    northing_span = df['utm_northing'].max() - df['utm_northing'].min()
    print(f"\n📏 TRACK DIMENSIONS:")
    print(f"   East-West span:  {easting_span:.1f} meters")
    print(f"   North-South span: {northing_span:.1f} meters")
    print(f"   Approximate area: {(easting_span * northing_span / 10000):.1f} hectares")
    
    # Precision analysis
    print(f"\n🔍 COORDINATE PRECISION:")
    
    # Original precision
    lat_str = f"{df['lat'].iloc[0]:.10f}"
    lon_str = f"{df['lon'].iloc[0]:.10f}"
    lat_decimals = len(lat_str.split('.')[1].rstrip('0'))
    lon_decimals = len(lon_str.split('.')[1].rstrip('0'))
    print(f"   Original lat/lon precision: ~{max(lat_decimals, lon_decimals)} decimal places")
    
    # UTM precision
    utm_e_str = f"{df['utm_easting'].iloc[0]:.10f}"
    utm_n_str = f"{df['utm_northing'].iloc[0]:.10f}"
    utm_e_decimals = len(utm_e_str.split('.')[1].rstrip('0'))
    utm_n_decimals = len(utm_n_str.split('.')[1].rstrip('0'))
    print(f"   UTM coordinate precision: ~{max(utm_e_decimals, utm_n_decimals)} decimal places")
    
    # Calculate distances in both coordinate systems
    print(f"\n📐 DISTANCE CALCULATION COMPARISON:")
    
    # Sample points for distance calculation
    idx1, idx2 = 0, min(100, len(df)-1)
    
    # Lat/lon distance (Haversine)
    lat1, lon1 = df.iloc[idx1]['lat'], df.iloc[idx1]['lon']
    lat2, lon2 = df.iloc[idx2]['lat'], df.iloc[idx2]['lon']
    
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * 
         math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    haversine_dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # UTM distance (Euclidean)
    utm_e1, utm_n1 = df.iloc[idx1]['utm_easting'], df.iloc[idx1]['utm_northing']
    utm_e2, utm_n2 = df.iloc[idx2]['utm_easting'], df.iloc[idx2]['utm_northing']
    utm_dist = math.sqrt((utm_e2 - utm_e1)**2 + (utm_n2 - utm_n1)**2)
    
    print(f"   Sample distance calculation (points {idx1} to {idx2}):")
    print(f"     Haversine (lat/lon): {haversine_dist:.3f} m")
    print(f"     Euclidean (UTM):     {utm_dist:.3f} m")
    print(f"     Difference:          {abs(haversine_dist - utm_dist):.3f} m ({100*abs(haversine_dist - utm_dist)/haversine_dist:.3f}%)")
    
    # Benefits of UTM
    print(f"\n🎯 BENEFITS OF UTM COORDINATES:")
    print(f"   ✅ Simple Euclidean distance: sqrt((x2-x1)² + (y2-y1)²)")
    print(f"   ✅ Meters as units (easier for analysis)")
    print(f"   ✅ Local Cartesian system (suitable for mapping)")
    print(f"   ✅ Reduced distortion over local areas")
    print(f"   ✅ Better for trajectory analysis and visualization")
    print(f"   ✅ Consistent zone ensures no coordinate jumps")
    
    # Sample data
    print(f"\n📋 SAMPLE DATA POINTS:")
    sample_cols = ['lat', 'lon', 'utm_easting', 'utm_northing', 'utm_zone', 'utm_letter']
    if 'heading_deg' in df.columns:
        sample_cols.append('heading_deg')
    available_cols = [col for col in sample_cols if col in df.columns]
    
    print(df[available_cols].head(5).to_string(index=False, float_format='%.6f'))
    
    print(f"\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python verify_utm_conversion.py <csv_file>")
        sys.exit(1)
    
    verify_utm_conversion(sys.argv[1])