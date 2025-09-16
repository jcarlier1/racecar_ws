#!/usr/bin/env python3
"""
Greedy Maximum Coverage Dataset Reducer

This script implements a greedy algorithm to reduce a dataset while maximizing
spatial coverage. It iteratively selects the next image that covers the most
"uncovered" geographical area.

Usage:
    python greedy_coverage_reducer.py <csv_path> [--target_count N] [--coverage_radius M] [--output path]
"""

import pandas as pd
import numpy as np
import argparse
import os
import time
from typing import List, Set, Tuple
import math

class GreedyCoverageReducer:
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
    
    def calculate_new_coverage(self, candidate_idx: int, already_covered: Set[int], 
                             coverage_radius_m: float) -> int:
        """
        Calculate how many new points would be covered by selecting a candidate point.
        
        Args:
            candidate_idx: Index of candidate point to evaluate
            already_covered: Set of already covered point indices
            coverage_radius_m: Coverage radius in meters
            
        Returns:
            Number of new points that would be covered
        """
        points_in_radius = self.get_points_within_radius(candidate_idx, coverage_radius_m)
        new_points = points_in_radius - already_covered
        return len(new_points)
    
    def greedy_max_coverage(self, target_count: int, coverage_radius_m: float = 100.0) -> List[int]:
        """
        Implement the greedy maximum coverage algorithm.
        
        Args:
            target_count: Number of images to select
            coverage_radius_m: Coverage radius in meters for each selected point
            
        Returns:
            List of selected image indices
        """
        print(f"Starting greedy coverage selection for {target_count} images...")
        print(f"Coverage radius: {coverage_radius_m}m")
        
        selected_indices = []
        covered_points = set()
        
        # Progress tracking
        start_time = time.time()
        
        for iteration in range(target_count):
            if len(covered_points) >= self.total_points:
                print(f"All points covered after {iteration} selections")
                break
            
            best_idx = -1
            best_new_coverage = 0
            
            # Find the candidate that covers the most uncovered points
            for candidate_idx in range(self.total_points):
                if candidate_idx in selected_indices:
                    continue  # Skip already selected points
                
                new_coverage = self.calculate_new_coverage(
                    candidate_idx, covered_points, coverage_radius_m
                )
                
                if new_coverage > best_new_coverage:
                    best_new_coverage = new_coverage
                    best_idx = candidate_idx
            
            # If no candidate provides new coverage, we're done
            if best_idx == -1 or best_new_coverage == 0:
                print(f"No more beneficial candidates found after {iteration} selections")
                break
            
            # Select the best candidate
            selected_indices.append(best_idx)
            
            # Update covered points
            newly_covered = self.get_points_within_radius(best_idx, coverage_radius_m)
            covered_points.update(newly_covered)
            
            # Progress reporting
            if (iteration + 1) % 50 == 0 or iteration < 10:
                elapsed = time.time() - start_time
                coverage_pct = len(covered_points) / self.total_points * 100
                print(f"Selected {iteration + 1:4d}/{target_count} images | "
                      f"Coverage: {len(covered_points):5d}/{self.total_points} ({coverage_pct:5.1f}%) | "
                      f"New points: {best_new_coverage:3d} | "
                      f"Time: {elapsed:.1f}s")
        
        final_coverage_pct = len(covered_points) / self.total_points * 100
        total_time = time.time() - start_time
        
        print(f"\nGreedy selection completed:")
        print(f"  Selected images: {len(selected_indices)}")
        print(f"  Total coverage: {len(covered_points)}/{self.total_points} ({final_coverage_pct:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        
        return selected_indices
    
    def reduce_dataset(self, target_count: int = 1000, coverage_radius_m: float = 100.0) -> pd.DataFrame:
        """
        Reduce the dataset using greedy maximum coverage.
        
        Args:
            target_count: Number of images to select (default: 1000)
            coverage_radius_m: Coverage radius in meters (default: 100)
            
        Returns:
            DataFrame with selected images
        """
        selected_indices = self.greedy_max_coverage(target_count, coverage_radius_m)
        
        # Sort indices to maintain temporal order in output
        selected_indices.sort()
        
        reduced_df = self.df.iloc[selected_indices].copy()
        reduced_df.reset_index(drop=True, inplace=True)
        
        return reduced_df
    
    def analyze_coverage(self, selected_df: pd.DataFrame, coverage_radius_m: float = 100.0):
        """
        Analyze the spatial coverage of the selected dataset.
        
        Args:
            selected_df: DataFrame with selected images
            coverage_radius_m: Coverage radius used for analysis
        """
        selected_coords = selected_df[['lat', 'lon']].values
        covered_points = set()
        
        print(f"\nAnalyzing coverage with {len(selected_df)} selected images...")
        
        for i, (lat, lon) in enumerate(selected_coords):
            for j in range(self.total_points):
                dist = self.haversine_distance(
                    lat, lon,
                    self.coords[j, 0], self.coords[j, 1]
                )
                if dist <= coverage_radius_m:
                    covered_points.add(j)
        
        coverage_pct = len(covered_points) / self.total_points * 100
        
        print(f"Final coverage analysis:")
        print(f"  Covered points: {len(covered_points)}/{self.total_points}")
        print(f"  Coverage percentage: {coverage_pct:.2f}%")
        print(f"  Reduction ratio: {len(selected_df)}/{self.total_points} = {len(selected_df)/self.total_points:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description='Reduce dataset using greedy maximum coverage algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python greedy_coverage_reducer.py poses.csv
  python greedy_coverage_reducer.py poses.csv --target_count 500 --coverage_radius 150
  python greedy_coverage_reducer.py poses.csv --output reduced_poses.csv
        """
    )
    
    parser.add_argument('csv_path', help='Path to input CSV file with pose data')
    parser.add_argument('--target_count', type=int, default=1000,
                       help='Number of images to select (default: 1000)')
    parser.add_argument('--coverage_radius', type=float, default=100.0,
                       help='Coverage radius in meters (default: 100.0)')
    parser.add_argument('--output', help='Output CSV path (default: adds _greedy_reduced suffix)')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform coverage analysis after reduction')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_path):
        print(f"Error: Input file '{args.csv_path}' not found")
        return 1
    
    try:
        # Initialize reducer
        reducer = GreedyCoverageReducer(args.csv_path)
        
        # Reduce dataset
        reduced_df = reducer.reduce_dataset(args.target_count, args.coverage_radius)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            base_name = os.path.splitext(args.csv_path)[0]
            output_path = f"{base_name}_greedy_reduced_{args.target_count}.csv"
        
        # Save reduced dataset
        reduced_df.to_csv(output_path, index=False)
        print(f"\nReduced dataset saved to: {output_path}")
        
        # Optional coverage analysis
        if args.analyze:
            reducer.analyze_coverage(reduced_df, args.coverage_radius)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
