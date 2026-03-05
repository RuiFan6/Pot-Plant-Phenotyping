import open3d as o3d
import numpy as np
import argparse
from pathlib import Path


DIST_THRESH_MM = 2.0


def rotation_align_vectors(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = float(np.clip(np.dot(v_from, v_to), -1.0, 1.0))

    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        # 180°: pick any axis orthogonal to v_from
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)

    axis = np.cross(v_from, v_to)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    input_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/fused_scaled.ply")
    output_path = Path(f"outputs/models/{PLANT_ID}/sfm/dense/fused_aligned.ply")

    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(str(input_path))

    print("Estimating turntable plane (RANSAC)...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=DIST_THRESH_MM,
        ransac_n=3,
        num_iterations=2000
    )
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=float)
    n /= np.linalg.norm(n)

    # 1) Rotate so plane normal aligns to +Z
    R = rotation_align_vectors(n, np.array([0.0, 0.0, 1.0]))
    pcd.rotate(R, center=(0, 0, 0))

    # Recompute plane z level using inliers after rotation
    pts = np.asarray(pcd.points)
    inlier_pts = pts[np.array(inliers, dtype=int)]
    z0 = float(np.median(inlier_pts[:, 2]))

    # 2) Decide which side is "up" by vertical extent away from the plane
    above = pts[:, 2] - z0
    below = z0 - pts[:, 2]
    extent_above = float(np.percentile(above, 99))
    extent_below = float(np.percentile(below, 99))

    # If "below" has larger extent, cloud is upside-down → flip 180° around X
    if extent_below > extent_above:
        print("[INFO] Detected upside-down orientation. Flipping...")
        Rflip = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
        pcd.rotate(Rflip, center=(0, 0, 0))
    else:
        print("[INFO] Orientation looks correct.")

    print("Saving aligned cloud...")
    o3d.io.write_point_cloud(str(output_path), pcd)
    print("Done:", output_path)


if __name__ == "__main__":
    main()