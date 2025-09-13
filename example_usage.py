#!/usr/bin/env python3
"""
Example usage of the image spacing filter program.
"""

import subprocess
import sys
import os

def run_example():
    """Run example with different parameters."""
    
    # Path to the sample CSV file
    csv_file = "output/sequences/M-MULTI-SLOW-KAIST/poses/poses.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Sample file {csv_file} not found.")
        return
    
    print("=" * 60)
    print("Image Spacing Filter - Example Usage")
    print("=" * 60)
    
    # Example 1: Default 0.25m spacing
    print("\n1. Default spacing (0.25m):")
    print("-" * 30)
    subprocess.run([sys.executable, "image_spacing_filter.py", csv_file])
    
    # Example 2: 1m spacing
    print("\n2. 1 meter spacing:")
    print("-" * 30)
    subprocess.run([sys.executable, "image_spacing_filter.py", csv_file, "-d", "1.0"])
    
    # Example 3: 5m spacing with verbose output
    print("\n3. 5 meter spacing (verbose):")
    print("-" * 30)
    subprocess.run([sys.executable, "image_spacing_filter.py", csv_file, "-d", "5.0", "-v"])
    
    # Example 4: Save filtered results
    print("\n4. Save filtered results to file:")
    print("-" * 30)
    output_file = "filtered_poses_5m.csv"
    subprocess.run([
        sys.executable, "image_spacing_filter.py", csv_file, 
        "-d", "5.0", "-o", output_file
    ])
    
    if os.path.exists(output_file):
        print(f"Filtered results saved to: {output_file}")
        # Show first few lines of the output
        with open(output_file, 'r') as f:
            lines = f.readlines()[:6]  # Header + 5 data lines
            print("\nFirst few lines of filtered results:")
            for line in lines:
                print(line.strip())

if __name__ == "__main__":
    run_example()
