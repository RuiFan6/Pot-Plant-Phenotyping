import cv2
import numpy as np
import pycolmap
from pathlib import Path
from scipy.spatial import cKDTree

PLANT_ID = "001"
MARKER_SIZE_MM = 15.0

BASE = Path(f"outputs/models/{PLANT_ID}/sfm")
IMAGES_DIR = BASE / "images"
SPARSE_DIR = BASE / "sparse/0"

print("Loading sparse model...")
reconstruction = pycolmap.Reconstruction(str(SPARSE_DIR))

# Load ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

edge_lengths = []

for image_id, image in reconstruction.images.items():

    img_path = IMAGES_DIR / image.name
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        continue

    # Build KD-tree of COLMAP keypoints
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
            if dist < 5:  # 5 pixel tolerance
                point3D = reconstruction.points3D[point3D_ids[idx]].xyz
                pts3D.append(point3D)

        if len(pts3D) == 4:
            pts3D = np.array(pts3D)

            # compute 4 edge lengths
            edges = [
                np.linalg.norm(pts3D[0] - pts3D[1]),
                np.linalg.norm(pts3D[1] - pts3D[2]),
                np.linalg.norm(pts3D[2] - pts3D[3]),
                np.linalg.norm(pts3D[3] - pts3D[0]),
            ]

            edge_lengths.extend(edges)

edge_lengths = np.array(edge_lengths)

print("Total measured edges:", len(edge_lengths))
print("Mean edge length (SfM units):", edge_lengths.mean())

scale_factor = MARKER_SIZE_MM / edge_lengths.mean()
print("Scale factor (mm per SfM unit):", scale_factor)