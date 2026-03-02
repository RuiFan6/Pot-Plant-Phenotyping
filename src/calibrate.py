import cv2
import numpy as np
from pathlib import Path
import random

CHECKERBOARD = (7, 5)     # inner corners
SQUARE_SIZE_MM = 30.0     # cell size
FRAME_DIR = Path("data/processed/calib/frames")

SAMPLE_MAX = 120          # use at most this many frames for speed/diversity
OUT_NPZ = Path("configs/camera_video_4k30_1x.npz")
PREVIEW_OUT = Path("outputs/figures/calib_undistort_preview.jpg")

def main():
    paths = sorted(FRAME_DIR.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No frames found in {FRAME_DIR.resolve()}")

    if len(paths) > SAMPLE_MAX:
        paths = random.sample(paths, SAMPLE_MAX)
        print(f"[INFO] Subsampled to {len(paths)} frames for calibration")
    else:
        print(f"[INFO] Using {len(paths)} frames for calibration")

    # 3D points on checkerboard plane (Z=0)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []
    img_shape = None
    good_paths = []

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        # mild contrast help (harmless if already good)
        gray = cv2.equalizeHist(gray)

        ret, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            good_paths.append(p)

    print(f"[INFO] Checkerboard detected in {len(good_paths)}/{len(paths)} sampled frames")
    if len(good_paths) < 15:
        raise RuntimeError("Not enough detections for reliable calibration. Need ~15+.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Mean reprojection error
    total = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        total += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total / len(objpoints)

    print("\n==== Calibration Result ====")
    print("K:\n", K)
    print("dist:", dist.ravel())
    print(f"Mean reprojection error: {mean_err:.4f} px")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(OUT_NPZ), K=K, dist=dist, checkerboard=CHECKERBOARD, square_mm=SQUARE_SIZE_MM)
    print(f"[INFO] Saved: {OUT_NPZ.resolve()}")

    # Save an undistortion preview for sanity check
    preview_img = cv2.imread(str(good_paths[0]))
    h, w = preview_img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    und = cv2.undistort(preview_img, K, dist, None, newK)

    PREVIEW_OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW_OUT), und)
    print(f"[INFO] Saved undistortion preview: {PREVIEW_OUT.resolve()}")

if __name__ == "__main__":
    main()