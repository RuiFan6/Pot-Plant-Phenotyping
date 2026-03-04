import open3d as o3d
import numpy as np
from pathlib import Path

# ========== INPUT PARAMETERS ==========
PLANT_ID = "002"

CLEARANCE_MM = 60.0         # remove everything below plane + clearance
POT_HEIGHT_MM = 130   #160.0      # <-- set manually
POT_DIAMETER_MM = 140    #170.0    # <-- set manually
# ======================================

input_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/fused_aligned.ply")
output_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/plant_only.ply")

print("Loading aligned cloud...")
pcd = o3d.io.read_point_cloud(str(input_path))
points = np.asarray(pcd.points)

print("Detecting turntable plane...")
plane_model, inliers = pcd.segment_plane(
    distance_threshold=2.0,
    ransac_n=3,
    num_iterations=2000
)

[a, b, c, d] = plane_model
plane_normal = np.array([a, b, c])
plane_normal /= np.linalg.norm(plane_normal)

inlier_pts = points[inliers]
z_plane = np.median(inlier_pts[:, 2])

print("Turntable plane z:", z_plane)

# --------- STEP 1: Remove below turntable ----------
mask_above = points[:, 2] > (z_plane + CLEARANCE_MM)
filtered = points[mask_above]

print("After ground removal:", len(filtered))

# --------- STEP 2: Estimate pot center ----------
turntable_center = np.mean(inlier_pts[:, :2], axis=0)
cx, cy = turntable_center
print("Estimated pot center:", cx, cy)

R = POT_DIAMETER_MM / 2.0

# --------- STEP 3: Remove pot cylinder ----------
dx = filtered[:, 0] - cx
dy = filtered[:, 1] - cy
radial = np.sqrt(dx**2 + dy**2)

z_limit = z_plane + POT_HEIGHT_MM

mask_not_pot = ~((radial <= R) & (filtered[:, 2] <= z_limit))
plant_points = filtered[mask_not_pot]

print("After pot removal:", len(plant_points))

# --------- Save result (preserve colors) ----------
colors = np.asarray(pcd.colors)
filtered_colors = colors[mask_above]
plant_colors = filtered_colors[mask_not_pot]

plant_cloud = o3d.geometry.PointCloud()
plant_cloud.points = o3d.utility.Vector3dVector(plant_points)
plant_cloud.colors = o3d.utility.Vector3dVector(plant_colors)

o3d.io.write_point_cloud(str(output_path), plant_cloud)

print("Saved:", output_path)