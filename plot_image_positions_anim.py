#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def main():
    ap = argparse.ArgumentParser(description="Animate image positions with time color scale.")
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
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Image Positions Animation")
    if args.equal:
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    scat = ax.scatter([], [], c=[], s=1, marker=".", cmap="viridis")
    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label("Time since start (s)")

    # Animation settings
    duration = 10  # seconds
    fps = 30
    n_frames = duration * fps
    interval = 1000 / fps  # ms per frame

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        return scat,

    def update(frame):
        # Calculate index for current frame
        end = int((frame / n_frames) * len(lats))
        if end < 1:
            end = 1
        offsets = np.column_stack((lons[:end], lats[:end]))
        scat.set_offsets(offsets)
        scat.set_array(t_rel[:end])
        return scat,

    anim = FuncAnimation(fig, update, frames=range(1, n_frames + 1),
                         init_func=init, interval=interval, blit=True)

    # Save animation
    bag_name = os.path.basename(os.path.dirname(os.path.dirname(args.csv)))
    save_name = f"{bag_name}_image_positions_anim.mp4"
    anim.save(save_name, writer='ffmpeg', fps=fps)
    print(f"Animation saved as {save_name}")

if __name__ == "__main__":
    main()
