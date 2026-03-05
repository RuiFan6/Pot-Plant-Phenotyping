import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import ConvexHull


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    input_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/plant_clean.ply")

    print("Loading plant-only cloud...")
    pcd = o3d.io.read_point_cloud(str(input_path))
    points = np.asarray(pcd.points)

    # ------------------------
    # 1. Height
    # ------------------------
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()
    height = z_max - z_min

    # ------------------------
    # 2. Canopy diameter
    # ------------------------
    xy = points[:, :2]
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    diameter_x = xy_max[0] - xy_min[0]
    diameter_y = xy_max[1] - xy_min[1]
    canopy_diameter = max(diameter_x, diameter_y)

    # ------------------------
    # 3. Projected area (Convex hull in XY)
    # ------------------------
    hull_2d = ConvexHull(xy)
    projected_area = hull_2d.volume  # 2D hull area

    # ------------------------
    # 4. 3D Volume (Convex hull)
    # ------------------------
    hull_3d = ConvexHull(points)
    volume = hull_3d.volume

    print("\n==== Basic Traits ====")
    print(f"Height (mm): {height:.2f}")
    print(f"Canopy diameter (mm): {canopy_diameter:.2f}")
    print(f"Projected area (mm^2): {projected_area:.2f}")
    print(f"Convex hull volume (mm^3): {volume:.2f}")


if __name__ == "__main__":
    main()