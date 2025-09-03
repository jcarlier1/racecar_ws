# Visual Localization Database - README

This repo is dedicated to producing a visual localization database for the RACECAR dataset. It contains scripts and resources for extracting, processing, and visualizing image and pose data from various RACECAR sequences.

## RACECAR Dataset
This repository is designed to work with the [RACECAR dataset](https://github.com/jcarlier1/racecar-dataset). Please refer to their official repository for dataset details, download instructions, and documentation.

## Purpose
The main goal is to facilitate the creation and analysis of a database that supports visual localization tasks. This includes converting raw data (such as ROS bags) into structured image and pose datasets, and providing tools to visualize and inspect the results.

## Folder Structure
- `data/`: Contains raw and processed racecar datasets organized by sequence and scenario.
- `output/`: Stores generated sequences with extracted images and poses.
- `plots/`: Contains visualizations and animations produced by the plotting scripts.
- `racecar_ws/`: Includes conversion configurations and utility scripts for data processing.

## Plotting Scripts
- `plot_image_positions.py`: Plots the positions of images in a sequence, providing a spatial overview of the dataset.
- `plot_image_positions_local.py`: Similar to the above, but focuses on local coordinate frames for more detailed analysis.
- `plot_image_positions_anim.py`: Generates animations showing the movement and image positions over time, useful for dynamic visualization.

## Data Conversion and Utilities
- `racecar_py/`: Contains Python scripts for converting ROS bag files to images and poses, building localization datasets, and various image processing tools.
  - `bag_to_images.py`: Extracts images from ROS bag files.
  - `build_vloc_from_bag.py`: Builds the visual localization database from bag files.
  - `image_tools.py`, `local_odom_conversion.py`, `racecar_utils.py`: Utility scripts for image and odometry processing.

## Usage
1. Use the conversion scripts in `racecar_py/` to process raw data and generate image/pose sequences.
2. Visualize the results using the plotting scripts to inspect image positions and trajectories.
3. Refer to the configuration files in `racecar_ws/conversion_configs/` for customizing data extraction and processing.

## Notes
- The repo structure is organized to support multiple scenarios and datasets.
- Animations and plots are useful for validating the quality and coverage of the visual localization database.

For more details on individual scripts, refer to their inline documentation or usage comments.
