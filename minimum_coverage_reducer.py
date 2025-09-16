#!/usr/bin/env python3
"""
Minimum Coverage Dataset Reducer

This script implements an algorithm to find the minimum number of images needed
to achieve 100% spatial coverage of a track. It uses a greedy approach to solve
the minimum set cover problem with spatial constraints.

Usage:
    python minimum_coverage_reducer.py <csv_path> [--coverage_radius M] [--output path]
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
from typing import List, Set, Tuple
import math

class MinimumCoverageReducer:
    def __init__(self, csv_path: str):
        """
        Initialize the reducer with a CSV file containing pose data.
        
        Args:
            csv_path: Path to CSV file with columns: camera_id, img_relpath, t_nsec, lat, lon, alt
        """
        self.df = pd.read_csv(csv_path)
        self.coords = self.df[['lat', 'lon']].values
        self.total_points = len(self.coords)
        
        print(f"Loaded {self.total_points} images from {csv_path}")
        
        # Validate required columns
        required_cols = ['lat', 'lon', 'img_relpath']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the haversine distance between two points on Earth in meters.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def get_points_within_radius(self, center_idx: int, radius_m: float) -> Set[int]:
        """
        Get all point indices within a given radius of a center point.
        
        Args:
            center_idx: Index of the center point
            radius_m: Radius in meters
            
        Returns:
            Set of point indices within the radius
        """
        center_lat, center_lon = self.coords[center_idx]
        covered_points = set()
        
        for i in range(self.total_points):
            if i == center_idx:
                covered_points.add(i)
                continue
                
            dist = self.haversine_distance(
                center_lat, center_lon,
                self.coords[i, 0], self.coords[i, 1]
            )
            
            if dist <= radius_m:
                covered_points.add(i)
        
        return covered_points
    
    def precompute_coverage_sets(self, coverage_radius_m: float) -> List[Set[int]]:
        """
        Precompute the coverage set for each point to speed up the algorithm.
        
        Args:
            coverage_radius_m: Coverage radius in meters
            
        Returns:
            List where index i contains the set of points covered by point i
        """
        print("Precomputing coverage sets...")
        coverage_sets = []
        
        for i in range(self.total_points):
            coverage_set = self.get_points_within_radius(i, coverage_radius_m)
            coverage_sets.append(coverage_set)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{self.total_points} points")
        
        print("Coverage sets precomputed.")
        return coverage_sets
    
    def minimum_set_cover_greedy(self, coverage_radius_m: float = 1.0) -> List[int]:
        """
        Find the minimum set of images that covers all points using a greedy algorithm.
        This is an approximation to the minimum set cover problem.
        
        Args:
            coverage_radius_m: Coverage radius in meters for each selected point
            
        Returns:
            List of selected image indices that achieve 100% coverage
        """
        print(f"Finding minimum set cover with {coverage_radius_m}m radius...")
        
        # Precompute coverage sets for efficiency
        coverage_sets = self.precompute_coverage_sets(coverage_radius_m)
        
        selected_indices = []
        uncovered_points = set(range(self.total_points))
        
        start_time = time.time()
        iteration = 0
        
        while uncovered_points:
            iteration += 1
            best_idx = -1
            best_new_coverage = 0
            best_coverage_set = set()
            
            # Find the candidate that covers the most uncovered points
            for candidate_idx in range(self.total_points):
                if candidate_idx in selected_indices:
                    continue  # Skip already selected points
                
                # Calculate intersection with uncovered points
                new_coverage_set = coverage_sets[candidate_idx] & uncovered_points
                new_coverage_count = len(new_coverage_set)
                
                if new_coverage_count > best_new_coverage:
                    best_new_coverage = new_coverage_count
                    best_idx = candidate_idx
                    best_coverage_set = new_coverage_set
            
            # If no candidate provides new coverage, something is wrong
            if best_idx == -1 or best_new_coverage == 0:
                print(f"Warning: No beneficial candidates found with {len(uncovered_points)} points remaining")
                break
            
            # Select the best candidate
            selected_indices.append(best_idx)
            uncovered_points -= best_coverage_set
            
            # Progress reporting
            coverage_pct = (self.total_points - len(uncovered_points)) / self.total_points * 100
            elapsed = time.time() - start_time
            
            if iteration % 10 == 0 or iteration <= 20 or len(uncovered_points) == 0:
                print(f"Iteration {iteration:3d}: Selected image {best_idx:5d} | "
                      f"Covered {best_new_coverage:3d} new points | "
                      f"Remaining: {len(uncovered_points):5d} | "
                      f"Coverage: {coverage_pct:6.2f}% | "
                      f"Time: {elapsed:.1f}s")
        
        total_time = time.time() - start_time
        
        print(f"\nMinimum set cover completed:")
        print(f"  Selected images: {len(selected_indices)}")
        print(f"  Coverage achieved: 100.0% ({self.total_points - len(uncovered_points)}/{self.total_points})")
        print(f"  Reduction ratio: {len(selected_indices)}/{self.total_points} = {len(selected_indices)/self.total_points:.6f}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time per selection: {total_time/len(selected_indices):.2f}s")
        
        return selected_indices
    
    def find_minimum_coverage(self, coverage_radius_m: float = 1.0) -> pd.DataFrame:
        """
        Find the minimum set of images needed for 100% coverage.
        
        Args:
            coverage_radius_m: Coverage radius in meters (default: 1.0)
            
        Returns:
            DataFrame with the minimum set of images for 100% coverage
        """
        selected_indices = self.minimum_set_cover_greedy(coverage_radius_m)
        
        # Sort indices to maintain temporal order in output
        selected_indices.sort()
        
        reduced_df = self.df.iloc[selected_indices].copy()
        reduced_df.reset_index(drop=True, inplace=True)
        
        return reduced_df
    
    def verify_coverage(self, selected_df: pd.DataFrame, coverage_radius_m: float = 1.0):
        """
        Verify that the selected images achieve 100% coverage.
        
        Args:
            selected_df: DataFrame with selected images
            coverage_radius_m: Coverage radius used for verification
        """
        selected_coords = selected_df[['lat', 'lon']].values
        covered_points = set()
        
        print(f"\nVerifying coverage with {len(selected_df)} selected images...")
        
        for i, (lat, lon) in enumerate(selected_coords):
            for j in range(self.total_points):
                dist = self.haversine_distance(
                    lat, lon,
                    self.coords[j, 0], self.coords[j, 1]
                )
                if dist <= coverage_radius_m:
                    covered_points.add(j)
        
        coverage_pct = len(covered_points) / self.total_points * 100
        uncovered_count = self.total_points - len(covered_points)
        
        print(f"Coverage verification:")
        print(f"  Covered points: {len(covered_points)}/{self.total_points}")
        print(f"  Coverage percentage: {coverage_pct:.4f}%")
        print(f"  Uncovered points: {uncovered_count}")
        print(f"  Verification: {'✓ PASSED' if uncovered_count == 0 else '✗ FAILED'}")
        
        if uncovered_count > 0:
            print(f"  Warning: {uncovered_count} points remain uncovered!")
    
    def analyze_spacing(self, selected_df: pd.DataFrame):
        """
        Analyze the spatial distribution of selected images.
        
        Args:
            selected_df: DataFrame with selected images
        """
        if len(selected_df) < 2:
            return
            
        selected_coords = selected_df[['lat', 'lon']].values
        distances = []
        
        # Calculate distances between consecutive selected points
        for i in range(len(selected_coords) - 1):
            dist = self.haversine_distance(
                selected_coords[i, 0], selected_coords[i, 1],
                selected_coords[i + 1, 0], selected_coords[i + 1, 1]
            )
            distances.append(dist)
        
        if distances:
            print(f"\nSpacing analysis:")
            print(f"  Mean distance between selections: {np.mean(distances):.2f}m")
            print(f"  Median distance: {np.median(distances):.2f}m")
            print(f"  Min distance: {np.min(distances):.2f}m")
            print(f"  Max distance: {np.max(distances):.2f}m")
            print(f"  Std deviation: {np.std(distances):.2f}m")

def main():
    parser = argparse.ArgumentParser(
        description='Find minimum set of images for 100% track coverage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python minimum_coverage_reducer.py poses.csv
  python minimum_coverage_reducer.py poses.csv --coverage_radius 2.0
  python minimum_coverage_reducer.py poses.csv --output minimal_poses.csv --verify
        """
    )
    
    parser.add_argument('csv_path', help='Path to input CSV file with pose data')
    parser.add_argument('--coverage_radius', type=float, default=1.0,
                       help='Coverage radius in meters (default: 1.0)')
    parser.add_argument('--output', help='Output CSV path (default: adds _minimal_coverage suffix)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify that 100%% coverage is achieved')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze spacing between selected images')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_path):
        print(f"Error: Input file '{args.csv_path}' not found")
        return 1
    
    try:
        # Initialize reducer
        reducer = MinimumCoverageReducer(args.csv_path)
        
        # Find minimum coverage
        minimal_df = reducer.find_minimum_coverage(args.coverage_radius)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(args.csv_path)[0]
            output_path = f"{base_name}_minimal_coverage_{args.coverage_radius}m.csv"
        
        # Save minimal dataset
        minimal_df.to_csv(output_path, index=False)
        print(f"\nMinimal coverage dataset saved to: {output_path}")
        
        # Optional verification
        if args.verify:
            reducer.verify_coverage(minimal_df, args.coverage_radius)
        
        # Optional spacing analysis
        if args.analyze:
            reducer.analyze_spacing(minimal_df)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
