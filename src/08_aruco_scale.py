import cv2
import numpy as np
import pycolmap
import open3d as o3d
import argparse
from pathlib import Path
from scipy.spatial import cKDTree

# =========================
# CONFIG
# =========================
MARKER_SIZE_MM = 15.0
PIXEL_TOLERANCE = 5.0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    BASE = Path(f"outputs/models/{PLANT_ID}/sfm")
    IMAGES_DIR = BASE / "images"
    SPARSE_DIR = BASE / "sparse/0"
    DENSE_PATH = BASE / "dense/fused.ply"
    SCALED_PATH = BASE / "dense/fused_scaled.ply"

    print("\n==============================")
    print("Plant ID:", PLANT_ID)
    print("Sparse model:", SPARSE_DIR.resolve())
    print("Dense model:", DENSE_PATH.resolve())
    print("==============================\n")

    # =========================
    # Load sparse reconstruction
    # =========================
    print("[INFO] Loading sparse model...")
    reconstruction = pycolmap.Reconstruction(str(SPARSE_DIR))

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    edge_lengths = []

    print("[INFO] Detecting ArUco markers...")

    for image_id, image in reconstruction.images.items():

        img_path = IMAGES_DIR / image.name
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        keypoints = np.array([p.xy for p in image.points2D if p.has_point3D()])
        point3D_ids = [p.point3D_id for p in image.points2D if p.has_point3D()]

        if len(keypoints) < 10:
            continue

        tree = cKDTree(keypoints)

        for marker in corners:
            marker = marker.reshape(-1, 2)

            pts3D = []
            for corner in marker:
                dist, idx = tree.query(corner, k=1)

                if dist < PIXEL_TOLERANCE:
                    point3D = reconstruction.points3D[point3D_ids[idx]].xyz
                    pts3D.append(point3D)

            if len(pts3D) == 4:
                pts3D = np.array(pts3D)

                edges = [
                    np.linalg.norm(pts3D[0] - pts3D[1]),
                    np.linalg.norm(pts3D[1] - pts3D[2]),
                    np.linalg.norm(pts3D[2] - pts3D[3]),
                    np.linalg.norm(pts3D[3] - pts3D[0]),
                ]

                edge_lengths.extend(edges)

    edge_lengths = np.array(edge_lengths)

    if len(edge_lengths) == 0:
        raise RuntimeError("No valid marker edges detected.")

    # =========================
    # Compute scale
    # =========================
    mean_edge = edge_lengths.mean()
    median_edge = np.median(edge_lengths)

    scale_factor = MARKER_SIZE_MM / median_edge

    print("\n========== SCALE ESTIMATION ==========")
    print("Total edges measured:", len(edge_lengths))
    print("Mean edge length (SfM units):", mean_edge)
    print("Median edge length (SfM units):", median_edge)
    print("Scale factor (mm per SfM unit):", scale_factor)

    # =========================
    # Scale dense cloud
    # =========================
    print("\n[INFO] Loading dense point cloud...")
    print("Input file:", DENSE_PATH.resolve())

    pcd = o3d.io.read_point_cloud(str(DENSE_PATH))

    points = np.asarray(pcd.points)
    points *= scale_factor
    pcd.points = o3d.utility.Vector3dVector(points)

    print("\n[INFO] Saving scaled point cloud...")
    print("Output file:", SCALED_PATH.resolve())

    o3d.io.write_point_cloud(str(SCALED_PATH), pcd)

    print("Scaling complete.")
    print("Scaled point cloud saved to:")
    print(SCALED_PATH.resolve())


if __name__ == "__main__":
    main()