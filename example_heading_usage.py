#!/usr/bin/env python3
"""
Example usage of the heading angle calculation script.
"""

import pandas as pd
import sys
import os

def example_usage():
    """Demonstrate usage of the heading angle script."""
    
    print("=== Heading Angle Calculation Example ===\n")
    
    # Paths
    input_file = "output/sequences/M-SOLO-SLOW-70-100/poses/poses.csv"
    output_file = "output/sequences/M-SOLO-SLOW-70-100/poses/poses_with_heading.csv"
    
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    # Read original data
    print(f"\nReading original data...")
    df_original = pd.read_csv(input_file)
    print(f"Original data shape: {df_original.shape}")
    print(f"Columns: {list(df_original.columns)}")
    
    # Read processed data
    if os.path.exists(output_file):
        print(f"\nReading processed data...")
        df_processed = pd.read_csv(output_file)
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Columns: {list(df_processed.columns)}")
        
        # Show sample data
        print(f"\nSample data with heading angles:")
        print(df_processed[['lat', 'lon', 'heading_deg']].head(10).to_string(index=False))
        
        # Heading statistics
        print(f"\nHeading angle statistics:")
        print(f"  Count: {len(df_processed)}")
        print(f"  Mean:  {df_processed['heading_deg'].mean():.2f}°")
        print(f"  Std:   {df_processed['heading_deg'].std():.2f}°") 
        print(f"  Min:   {df_processed['heading_deg'].min():.2f}°")
        print(f"  Max:   {df_processed['heading_deg'].max():.2f}°")
        
        # Check for any potential issues
        heading_changes = df_processed['heading_deg'].diff().abs()
        large_changes = heading_changes > 90  # Changes larger than 90 degrees
        if large_changes.any():
            print(f"\nWarning: {large_changes.sum()} points with large heading changes (>90°)")
            print("This could indicate sharp turns or data quality issues.")
        
    else:
        print(f"\nProcessed file not found. Run the following command to create it:")
        print(f"python add_heading_angle.py {input_file} {output_file}")


if __name__ == "__main__":
    example_usage()