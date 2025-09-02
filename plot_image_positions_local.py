#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

# Reference origin (lat, lon, alt)
LAT0 = 36.272371177449344
LON0 = -115.01030828834901
ALT0 = 594.0

# Earth radius approximation (WGS-84 ellipsoid simplified)
R_EARTH = 6378137.0  # meters

def lla_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert (lat, lon, alt) to local ENU in meters relative to (lat0, lon0, alt0)."""
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    # Differences
    dlat = lat - lat0
    dlon = lon - lon0
    dalt = alt - alt0

    # Simple equirectangular approximation (valid for small areas like a track)
    east  = dlon * np.cos(lat0) * R_EARTH
    north = dlat * R_EARTH
    up    = dalt

    return east, north, up

def main():
    ap = argparse.ArgumentParser(description="Plot image positions (ENU) with time color scale.")
    ap.add_argument("--csv", required=True, help="Path to poses.csv")
    ap.add_argument("--camera", default="front_left_center", help="Camera ID to filter")
    args = ap.parse_args()

    east, north, tns = [], [], []

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.camera and row["camera_id"] != args.camera:
                continue
            lat = float(row["lat"])
            lon = float(row["lon"])
            alt = float(row["alt"])
            e, n, _ = lla_to_enu(lat, lon, alt, LAT0, LON0, ALT0)
            east.append(e)
            north.append(n)
            tns.append(int(row["t_nsec"]))

    if not east:
        raise SystemExit("No rows parsed. Check camera_id or CSV.")

    east = np.array(east)
    north = np.array(north)
    tns = np.array(tns, dtype=np.int64)
    t_rel = (tns - tns.min()) / 1e9  # seconds since first frame

    fig, ax = plt.subplots()
    sc = ax.scatter(east, north, c=t_rel, s=6, marker="o")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time since start (s)")

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Trajectory of {args.camera}")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
