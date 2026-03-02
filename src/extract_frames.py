import cv2
import os
from pathlib import Path

def extract(video_path, out_dir, fps=2):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    step = int(video_fps / fps)
    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            out_path = out_dir / f"{saved:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_id += 1

    print(f"{video_path.name} → {saved} frames")


def main():
    raw = Path("data/raw/001")
    out = Path("data/processed/001/frames")

    for v in raw.glob("*.MOV"):
        name = v.stem
        extract(v, out / name, fps=2)


if __name__ == "__main__":
    main()