import open3d as o3d
import numpy as np
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    input_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/plant_only.ply")

    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_path))

    points = np.asarray(pcd.points)

    print("Point count:", len(points))

    print("Computing nearest neighbor distances...")
    distances = pcd.compute_nearest_neighbor_distance()
    distances = np.asarray(distances)

    print("\n==== Surface Thickness Diagnostic ====")

    print("Mean NN distance (mm):", distances.mean())
    print("Median NN distance (mm):", np.median(distances))
    print("95th percentile (mm):", np.percentile(distances, 95))
    print("99th percentile (mm):", np.percentile(distances, 99))
    print("Max distance (mm):", distances.max())

    print("\nInterpretation:")
    print("Good dense reconstruction usually has:")
    print("Mean NN distance ~0.3–1.5 mm")
    print("If values >3–5 mm appear, surfaces may be inflated.")


if __name__ == "__main__":
    main()