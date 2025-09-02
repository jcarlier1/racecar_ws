#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot image positions with time color scale.")
    ap.add_argument("--csv", required=True, help="Path to poses.csv")
    ap.add_argument("--camera", default=None,
                    help="Optional camera_id filter, e.g., front_left_center")
    ap.add_argument("--equal", action="store_true",
                    help="Set equal aspect ratio for lat lon axes")
    args = ap.parse_args()

    lats, lons, tns = [], [], []

    with open(args.csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.camera and row["camera_id"] != args.camera:
                continue
            try:
                lats.append(float(row["lat"]))
                lons.append(float(row["lon"]))
                tns.append(int(row["t_nsec"]))
            except Exception:
                continue

    if not lats:
        raise SystemExit("No rows parsed. Check --camera filter or CSV path.")

    lats = np.array(lats)
    lons = np.array(lons)
    tns = np.array(tns, dtype=np.int64)

    # Convert nanoseconds to seconds relative to the first timestamp
    t_rel = (tns - tns.min()) / 1e9

    fig, ax = plt.subplots()
    sc = ax.scatter(lons, lats, c=t_rel, s=6, marker="o")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time since start (s)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Image Positions with Time Color Scale")

    if args.equal:
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
