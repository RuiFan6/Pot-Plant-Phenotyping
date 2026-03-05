import open3d as o3d
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    input_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/plant_only.ply")
    output_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/plant_clean.ply")

    print("Loading plant point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_path))

    print("Initial point count:", len(pcd.points))

    # -----------------------------
    # 1. Statistical Outlier Removal
    # -----------------------------
    pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )

    print("After statistical filter:", len(pcd.points))

    # -----------------------------
    # 2. Radius Outlier Removal
    # -----------------------------
    pcd, ind = pcd.remove_radius_outlier(
        nb_points=10,
        radius=3.0
    )

    print("After radius filter:", len(pcd.points))

    print("Saving cleaned cloud...")
    o3d.io.write_point_cloud(str(output_path), pcd)

    print("Saved:", output_path)


if __name__ == "__main__":
    main()