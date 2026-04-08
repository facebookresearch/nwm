#!/usr/bin/env python3
"""
Download SACSon dataset from HuggingFace (HDF5 format) and convert
to the NWM training format with synthetic linear-motion odometry.

Usage:
    python prepare_sacson_hf.py --num-trajectories 200 --output-dir data/sacson
"""

import argparse
import os
import io
import pickle
import random

import numpy as np
from PIL import Image


METRIC_WAYPOINT_SPACING = 0.255  # SACSon's metric_waypoint_spacing from data_config.yaml


def download_hdf5(local_dir: str = "./datasets/huron_hdf5"):
    """Download the SACSon HDF5 dataset from HuggingFace."""
    hdf5_path = os.path.join(local_dir, "sacson.h5")
    if os.path.exists(hdf5_path):
        print(f"HDF5 file already exists at {hdf5_path}, skipping download.")
        return hdf5_path

    print("Downloading SACSon HDF5 from HuggingFace...")
    os.makedirs(local_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="konpat/huron-sacson-hdf5",
            filename="sacson.h5",
            repo_type="dataset",
            local_dir=local_dir,
        )
    except ImportError:
        import subprocess
        subprocess.run([
            "huggingface-cli", "download",
            "konpat/huron-sacson-hdf5",
            "sacson.h5",
            "--repo-type", "dataset",
            "--local-dir", local_dir,
        ], check=True)

    assert os.path.exists(hdf5_path), f"Download failed: {hdf5_path} not found"
    print(f"Downloaded to {hdf5_path}")
    return hdf5_path


def generate_synthetic_odometry(num_frames: int) -> dict:
    """
    Generate synthetic linear-motion odometry for a trajectory.
    Robot moves forward at metric_waypoint_spacing per frame, yaw = 0.
    """
    positions = np.zeros((num_frames, 2), dtype=np.float64)
    positions[:, 0] = np.arange(num_frames) * METRIC_WAYPOINT_SPACING  # x = forward
    yaw = np.zeros(num_frames, dtype=np.float64)
    return {"position": positions, "yaw": yaw}


def convert_hdf5_to_nwm(
    hdf5_path: str,
    output_dir: str,
    split_dir: str,
    num_trajectories: int = 200,
    split: str = "train",
    seed: int = 42,
    dry_run: bool = False,
):
    """
    Read trajectories from the HDF5 file and write them in NWM format:
      output_dir/<traj_name>/0.jpg, 1.jpg, ..., traj_data.pkl
    """
    import h5py

    print(f"Reading HDF5 from {hdf5_path}, split={split}")
    with h5py.File(hdf5_path, "r") as f:
        if split not in f:
            available = list(f.keys())
            raise ValueError(f"Split '{split}' not found in HDF5. Available: {available}")

        group = f[split]
        all_traj_names = sorted(list(group.keys()))
        print(f"  Total trajectories in '{split}': {len(all_traj_names)}")

        # Sample a subset
        random.seed(seed)
        if num_trajectories < len(all_traj_names):
            sampled = random.sample(all_traj_names, num_trajectories)
        else:
            sampled = all_traj_names
            print(f"  Requested {num_trajectories} but only {len(all_traj_names)} available, using all.")

        print(f"  Sampled {len(sampled)} trajectories")

        if dry_run:
            print("Dry run — not writing files.")
            for name in sampled[:5]:
                traj = group[name]
                if "frames" in traj:
                    print(f"    {name}: {len(traj['frames'])} frames")
                else:
                    print(f"    {name}: keys={list(traj.keys())}")
            return sampled

        os.makedirs(output_dir, exist_ok=True)
        written_names = []

        for i, traj_name in enumerate(sampled):
            traj_group = group[traj_name]

            # Extract frames
            if "frames" not in traj_group:
                print(f"  Skipping {traj_name}: no 'frames' dataset")
                continue

            frames = traj_group["frames"]
            num_frames = len(frames)
            if num_frames < 10:  # skip very short trajectories
                print(f"  Skipping {traj_name}: only {num_frames} frames")
                continue

            traj_dir = os.path.join(output_dir, traj_name)
            os.makedirs(traj_dir, exist_ok=True)

            # Decode and save JPEG frames
            for j in range(num_frames):
                frame_bytes = frames[j]
                if isinstance(frame_bytes, np.ndarray):
                    frame_bytes = frame_bytes.tobytes()
                img = Image.open(io.BytesIO(frame_bytes))
                img.save(os.path.join(traj_dir, f"{j}.jpg"))

            # Generate synthetic odometry
            odom = generate_synthetic_odometry(num_frames)
            with open(os.path.join(traj_dir, "traj_data.pkl"), "wb") as pf:
                pickle.dump(odom, pf)

            written_names.append(traj_name)

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(sampled)} trajectories")

    print(f"Wrote {len(written_names)} trajectories to {output_dir}")

    # Write data split
    os.makedirs(split_dir, exist_ok=True)
    traj_names_file = os.path.join(split_dir, "traj_names.txt")
    with open(traj_names_file, "w") as f:
        f.write("\n".join(written_names) + "\n")
    print(f"Wrote trajectory list to {traj_names_file}")

    return written_names


def main():
    parser = argparse.ArgumentParser(description="Prepare SACSon data from HuggingFace for NWM distillation")
    parser.add_argument("--hdf5-dir", type=str, default="./datasets/huron_hdf5",
                        help="Directory to download/find the HDF5 file")
    parser.add_argument("--output-dir", type=str, default="data/sacson",
                        help="Output directory for NWM-format trajectories")
    parser.add_argument("--split-dir", type=str, default="data_splits/sacson_distill/train",
                        help="Directory for the data split traj_names.txt")
    parser.add_argument("--num-trajectories", type=int, default=200,
                        help="Number of trajectories to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only inspect the HDF5 without writing files")
    args = parser.parse_args()

    hdf5_path = download_hdf5(args.hdf5_dir)
    convert_hdf5_to_nwm(
        hdf5_path=hdf5_path,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
