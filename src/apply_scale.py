import open3d as o3d
import numpy as np
from pathlib import Path

PLANT_ID = "001"
SCALE = 152.42568384345927  # <- paste your value here

dense_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/fused.ply")
scaled_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/fused_scaled.ply")

print("Loading dense cloud...")
pcd = o3d.io.read_point_cloud(str(dense_path))

points = np.asarray(pcd.points)
points *= SCALE
pcd.points = o3d.utility.Vector3dVector(points)

print("Saving scaled cloud...")
o3d.io.write_point_cloud(str(scaled_path), pcd)

print("Done.")