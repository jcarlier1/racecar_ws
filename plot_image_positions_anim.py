#!/usr/bin/env python3
import argparse
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Reference origin (lat, lon, alt)
LAT0 = 36.272371177449344
LON0 = -115.01030828834901
ALT0 = 594.0

# Earth radius approximation (WGS-84 ellipsoid simplified)
R_EARTH = 6378137.0  # meters

def lla_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert (lat, lon, alt) to local ENU in meters relative to (lat0, lon0, alt0)."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    dlat = lat - lat0
    dlon = lon - lon0
    dalt = alt - alt0

    east  = dlon * np.cos(lat0) * R_EARTH
    north = dlat * R_EARTH
    up    = dalt
    return east, north, up

def read_positions(csv_path, camera_id):
    east, north, tns = [], [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if camera_id and row["camera_id"] != camera_id:
                continue
            lat = float(row["lat"])
            lon = float(row["lon"])
            alt = float(row["alt"])
            e, n, _ = lla_to_enu(lat, lon, alt, LAT0, LON0, ALT0)
            east.append(e)
            north.append(n)
            tns.append(int(row["t_nsec"]))
    if not east:
        raise SystemExit("No rows parsed. Check camera_id or CSV path.")
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    tns = np.asarray(tns, dtype=np.int64)
    t_rel = (tns - tns.min()) / 1e9  # seconds since first frame
    return east, north, t_rel

def main():
    ap = argparse.ArgumentParser(description="Animate image positions around racetrack.")
    ap.add_argument("--csv", required=True, help="Path to poses.csv")
    ap.add_argument("--camera", default="front_left_center",
                    help="Camera ID to filter, e.g., front_left_center")
    ap.add_argument("--step", type=int, default=30,
                    help="Number of points to add per frame")
    ap.add_argument("--fps", type=int, default=30,
                    help="Frames per second for the animation")
    ap.add_argument("--save", default="", choices=["", "mp4", "gif"],
                    help="Save animation as mp4 or gif. Empty means do not save.")
    ap.add_argument("--s", type=float, default=8.0,
                    help="Marker size")
    args = ap.parse_args()

    # Load and prep data
    east, north, t_rel = read_positions(args.csv, args.camera)
    n_total = east.shape[0]
    n_frames = math.ceil(n_total / args.step)

    # Figure and static scaffolding
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter([], [], c=[], s=args.s, marker="o")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time in minute (s), color restarts every 60s")

    # Fix limits from the full dataset to avoid autoscale flicker
    pad = 5.0  # meters of padding
    ax.set_xlim(east.min() - pad, east.max() + pad)
    ax.set_ylim(north.min() - pad, north.max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")

    # Derive a concise title from folder name
    try:
        bag_name = os.path.basename(os.path.dirname(os.path.dirname(args.csv)))
    except Exception:
        bag_name = "sequence"
    ax.set_title(f"Trajectory animation: {args.camera}  [{bag_name}]")

    # Pre-allocate for speed
    pts = np.empty((0, 2), dtype=float)
    colors = np.empty((0,), dtype=float)

    # For consistent color scale across frames (reset every 60s)
    sc.set_clim(vmin=0, vmax=60)

    def init():
        sc.set_offsets(np.empty((0, 2)))
        sc.set_array(np.empty((0,), dtype=float))
        return (sc,)

    def update(frame_idx):
        end = min((frame_idx + 1) * args.step, n_total)
        # Slice all points up to this frame
        pts = np.column_stack([east[:end], north[:end]])
        colors = t_rel[:end] % 60  # Reset color every 60s
        sc.set_offsets(pts)
        sc.set_array(colors)
        ax.set_title(f"Trajectory animation: {args.camera}  [{bag_name}]  "
                    f"Points: {end}/{n_total}")
        return (sc,)

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / args.fps,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()

    if args.save:
        out_base = f"{bag_name}_{args.camera}_traj_anim"
        if args.save == "mp4":
            try:
                anim.save(f"{out_base}.mp4", writer="ffmpeg", fps=args.fps, dpi=150)
                print(f"Saved {out_base}.mp4")
            except Exception as e:
                print(f"MP4 save failed: {e}")
        elif args.save == "gif":
            anim.save(f"{out_base}.gif", writer="pillow", fps=args.fps)
            print(f"Saved {out_base}.gif")

    plt.show()

if __name__ == "__main__":
    main()
