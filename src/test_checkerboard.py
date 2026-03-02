import cv2
from pathlib import Path

CHECKERBOARD = (7, 5)  # inner corners for 8x6 cells
FRAME_DIR = Path("data/processed/calib/frames")
N_SHOW = 30  # how many frames to preview

def main():
    paths = sorted(FRAME_DIR.glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No frames found in {FRAME_DIR.resolve()}")

    ok = 0
    shown = 0

    for p in paths:
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if ret:
            ok += 1
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

            # show only first N_SHOW successes
            if shown < N_SHOW:
                cv2.imshow("checkerboard detections (press any key to advance)", img)
                cv2.waitKey(0)
                shown += 1

    cv2.destroyAllWindows()
    print(f"[INFO] Detected checkerboard in {ok}/{len(paths)} frames")

if __name__ == "__main__":
    main()