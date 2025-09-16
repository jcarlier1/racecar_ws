#!/usr/bin/env python3
"""
Export a minimal visual localization dataset from a RACECAR rosbag2 folder.

Output:
  <out_root>/images/front_left_center/<t_nsec>.png
  <out_root>/poses/poses.csv

poses.csv columns:
  camera_id, img_relpath, t_nsec, lat, lon, alt

Notes:
  - Only extracts the front_left_center camera (camera_id1).
  - Location from NavSatFix GPS messages or NovAtel BESTPOS messages.
  - No calibration is saved.
"""

import argparse
import csv
import re
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import get_types_from_msg, Stores, get_typestore


# Topics and types
FRONT_LEFT_CENTER_RE = re.compile(
    r'^/(vehicle_\d+)/camera_id1_imageU8$'
)
GPS_TYPES = {
    'sensor_msgs/msg/NavSatFix',
    'novatel_oem7_msgs/msg/BESTPOS',
    'novatel_gps_msgs/msg/BESTPOS',
}
IMAGE_TYPE = 'sensor_msgs/msg/Image'


def _iter_msg_files(msg_dir: Path):
    if not msg_dir.exists():
        return
    for p in msg_dir.glob('**/*.msg'):
        yield p


def register_custom_msgs(custom_root: Path):
    add_types = {}
    pkgs = [
        custom_root / 'ros2_custom_msgs' / 'novatel_oem7_msgs' / 'msg',
        custom_root / 'ros2_custom_msgs' / 'novatel_gps_msgs' / 'msg',
    ]
    for pkg_dir in pkgs:
        for msgpath in _iter_msg_files(pkg_dir):
            name = msgpath.relative_to(msgpath.parents[2]).with_suffix('')
            if 'msg' not in name.parts:
                name = name.parent / 'msg' / name.name
            msgdef = msgpath.read_text(encoding='utf-8')
            add_types.update(get_types_from_msg(msgdef, str(name)))

    # Create typestore for ROS 2 Humble and register custom types
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    if add_types:
        typestore.register(add_types)
    return typestore


def msg_get_lat_lon_alt(msg):
    """Extract (lat, lon, alt) for BESTPOS variants and NavSatFix."""
    # Handle NavSatFix messages
    if hasattr(msg, 'latitude') and hasattr(msg, 'longitude') and hasattr(msg, 'altitude'):
        return float(msg.latitude), float(msg.longitude), float(msg.altitude)
    
    # Handle BESTPOS variants
    for lat_key, lon_key, alt_key in [
        ('lat', 'lon', 'hgt'),     # novatel_oem7 BESTPOS
        ('lat', 'lon', 'height'),  # novatel_gps BESTPOS
    ]:
        lat = getattr(msg, lat_key, None)
        lon = getattr(msg, lon_key, None)
        alt = getattr(msg, alt_key, None)
        if lat is not None and lon is not None and alt is not None:
            return float(lat), float(lon), float(alt)
    return None


def nearest_idx(sorted_times, t):
    """Index of nearest timestamp in a sorted list of ints."""
    if not sorted_times:
        return None
    i = bisect_left(sorted_times, t)
    if i == 0:
        return 0
    if i == len(sorted_times):
        return len(sorted_times) - 1
    before = sorted_times[i - 1]
    after = sorted_times[i]
    return i - 1 if (t - before) <= (after - t) else i


def build_dataset(bag_dir: Path, out_dir: Path, custom_root: Path):
    print("Starting dataset build...")
    out_images = out_dir / 'images' / 'front_left_center'
    out_poses = out_dir / 'poses'
    out_images.mkdir(parents=True, exist_ok=True)
    out_poses.mkdir(parents=True, exist_ok=True)

    # Register custom NovAtel types for BESTPOS
    print("Registering custom messages...")
    register_custom_msgs(custom_root)

    # Discover relevant connections
    print("Discovering connections...")
    with Reader(bag_dir) as reader:
        conns = list(reader.connections)

    flc_img_conns = []
    gps_conns = []
    vehicle_ns = None

    for c in conns:
        if c.msgtype == IMAGE_TYPE:
            m = FRONT_LEFT_CENTER_RE.match(c.topic)
            if m:
                vehicle_ns = m.group(1)
                flc_img_conns.append(c)
        elif c.msgtype in GPS_TYPES:
            if vehicle_ns and f'/{vehicle_ns}/' in c.topic:
                gps_conns.append(c)
            elif not vehicle_ns:
                gps_conns.append(c)

    if not flc_img_conns:
        raise RuntimeError('No camera_id1_imageU8 topic found.')
    if not gps_conns:
        raise RuntimeError('No GPS messages found for location.')

    print(f"Found {len(flc_img_conns)} image connections, {len(gps_conns)} GPS connections")

    # Register custom NovAtel types for BESTPOS and get typestore
    typestore = register_custom_msgs(custom_root)

    # --- Pass 1: read GPS timeline ---
    print("Starting GPS pass...")
    gps_times, gps_vals = [], []
    with Reader(bag_dir) as reader:
        for conn, t_nsec, raw in reader.messages(connections=gps_conns):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)   # CHANGED
            llh = msg_get_lat_lon_alt(msg)
            if llh is not None:
                gps_times.append(t_nsec)
                gps_vals.append(llh)

    if not gps_times:
        raise RuntimeError('No valid GPS samples found.')

    order = np.argsort(gps_times)
    gps_times_sorted = [gps_times[i] for i in order]
    gps_vals_sorted = [gps_vals[i] for i in order]

    print(f"GPS pass done, collected {len(gps_times)} GPS points")

    # Pass 2: iterate only front_left_center images, save PNGs, write CSV
    print("Starting image processing pass...")
    poses_csv = out_poses / 'poses.csv'
    with open(poses_csv, 'w', newline='') as fcsv:
        W = csv.writer(fcsv)
        W.writerow(['camera_id', 'img_relpath', 't_nsec', 'lat', 'lon', 'alt'])

        count = 0
        with Reader(bag_dir) as reader:
            for conn, t_nsec, raw in reader.messages(connections=flc_img_conns):
                msg = typestore.deserialize_cdr(raw, conn.msgtype)   # CHANGED
                
                # Handle uncompressed Image messages
                if not hasattr(msg, 'data') or not hasattr(msg, 'width') or not hasattr(msg, 'height'):
                    continue
                
                # Convert ROS Image to OpenCV format
                if msg.encoding == 'rgb8':
                    img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'bgr8':
                    img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    img = img_array
                elif msg.encoding == 'mono8':
                    img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                    img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                else:
                    print(f"Warning: Unsupported encoding {msg.encoding}, skipping frame")
                    continue

                if img is None:
                    continue

                # nearest GPS
                idx = nearest_idx(gps_times_sorted, t_nsec)
                if idx is None:
                    continue
                lat, lon, alt = gps_vals_sorted[idx]

                # write image
                fname = f'{t_nsec}.png'
                cv2.imwrite(str(out_images / fname), img)

                # write row
                rel = f'images/front_left_center/{fname}'
                W.writerow(['front_left_center', rel, t_nsec,
                            f'{lat:.9f}', f'{lon:.9f}', f'{alt:.3f}'])

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} images")

    print(f"Image processing done, total images: {count}")
    print('[OK] Dataset written:')
    print(f'  Images: {out_images}')
    print(f'  CSV:    {poses_csv}')


def default_custom_root(script_path: Path) -> Path:
    # Your tree places ros2_custom_msgs next to racecar_py under racecar_ws
    return script_path.parent.parent


def main():
    ap = argparse.ArgumentParser(description='Export front_left_center images (camera_id1) + GPS from a RACECAR rosbag2 folder.')
    ap.add_argument('--bag', required=True, help='Path to rosbag2 folder (contains metadata.yaml and *.db3)')
    ap.add_argument('--out', required=True, help='Output dataset folder')
    ap.add_argument('--custom-root', default=None, help='Path containing ros2_custom_msgs (default: racecar_ws next to racecar_py)')
    args = ap.parse_args()

    bag_dir = Path(args.bag).resolve()
    out_dir = Path(args.out).resolve()
    custom_root = Path(args.custom_root).resolve() if args.custom_root else default_custom_root(Path(__file__).resolve())

    if not bag_dir.exists():
        raise SystemExit(f'Bag folder not found: {bag_dir}')
    if not (custom_root / 'ros2_custom_msgs').exists():
        raise SystemExit(f'Could not find ros2_custom_msgs under: {custom_root}')

    out_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(bag_dir, out_dir, custom_root)


if __name__ == '__main__':
    main()