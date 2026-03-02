import cv2
from pathlib import Path

VIDEO = Path("data/raw/chessboard.MOV")
OUT_DIR = Path("data/processed/calib/frames")
FPS_OUT = 3  # good for calibration

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    step = max(1, int(round(fps / FPS_OUT)))

    i = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            out_path = OUT_DIR / f"{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        i += 1

    cap.release()
    print(f"[INFO] Video FPS ~ {fps:.2f}, step={step}")
    print(f"[INFO] Saved {saved} frames -> {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()