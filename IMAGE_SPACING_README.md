# Image Spacing Filter

A Python program that processes GPS coordinates from CSV files and calculates the number of images that would remain if spaced at least a specified distance apart.

## Features

- Reads CSV files with GPS coordinates (latitude, longitude, altitude)
- Calculates 3D distances between consecutive image positions
- Filters images to maintain minimum distance spacing
- Provides detailed statistics and analysis
- Supports custom distance thresholds
- Can save filtered results to a new CSV file

## Requirements

- Python 3.x
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
```

## Usage

### Basic Usage

```bash
python3 image_spacing_filter.py <csv_file_path>
```

### With Custom Distance Threshold

```bash
python3 image_spacing_filter.py <csv_file_path> --min-distance 1.0
```

### Verbose Output

```bash
python3 image_spacing_filter.py <csv_file_path> --verbose
```

### Save Filtered Results

```bash
python3 image_spacing_filter.py <csv_file_path> --output filtered_results.csv
```

### Complete Example

```bash
python3 image_spacing_filter.py output/sequences/M-MULTI-SLOW-KAIST/poses/poses.csv \
    --min-distance 5.0 \
    --output filtered_poses.csv \
    --verbose
```

## Command Line Options

- `csv_file`: Path to the input CSV file (required)
- `-d, --min-distance`: Minimum distance in meters between images (default: 0.25)
- `-o, --output`: Output CSV file path for filtered results (optional)
- `-v, --verbose`: Print detailed information

## CSV File Format

The input CSV file must contain the following columns:

- `camera_id`: Camera identifier
- `img_relpath`: Relative path to the image file
- `t_nsec`: Timestamp in nanoseconds
- `lat`: Latitude in decimal degrees
- `lon`: Longitude in decimal degrees
- `alt`: Altitude in meters

Example CSV format:
```csv
camera_id,img_relpath,t_nsec,lat,lon,alt
front_left_center,images/front_left_center/1641163600034686435.png,1641163600034686435,36.275509146,-115.007278964,597.102
front_left_center,images/front_left_center/1641163600238028081.png,1641163600238028081,36.275518710,-115.007322144,596.535
```

## Output

The program provides the following information:

- Original number of images
- Number of images after filtering
- Reduction count and percentage
- Total distance covered
- Average distance between selected images (verbose mode)
- Time span and average speed (verbose mode)

### Example Output

```
Original number of images: 5092
Filtered number of images: 2377
Reduction: 2715 images (53.3%)
Minimum distance threshold: 5.0m
Total distance covered: 17790.21m

Average distance between selected images: 7.49m
First image timestamp: 1641163600034686435
Last image timestamp: 1641164299968566982
Time span: 699.93 seconds
Average speed: 25.42 m/s (91.50 km/h)
```

## Algorithm

The program uses the following approach:

1. **Distance Calculation**: Uses the Haversine formula for horizontal distance and Pythagorean theorem for 3D distance including altitude
2. **Filtering Strategy**: Iterates through chronologically sorted images, selecting each image that is at least the minimum distance away from the previously selected image
3. **Greedy Selection**: Uses a greedy algorithm that ensures no two selected images are closer than the threshold distance

## Distance Calculation

The program calculates true 3D distances by:
1. Computing horizontal distance using the Haversine formula (great circle distance)
2. Computing vertical distance from altitude difference
3. Combining using 3D Pythagorean theorem: `sqrt(horizontal² + vertical²)`

This ensures accurate distance calculations that account for both geographic and elevation changes.

## Examples

Run the example script to see different usage scenarios:

```bash
python3 example_usage.py
```

This will demonstrate:
- Default 0.25m spacing
- Custom 1m and 5m spacing
- Verbose output
- Saving filtered results to a file
