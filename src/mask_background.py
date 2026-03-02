from rembg import remove, new_session
import cv2
import numpy as np
from pathlib import Path

PLANT_ID = "001"

IN_DIR = Path(f"outputs/models/{PLANT_ID}/sfm/images")
OUT_IMG_DIR = Path(f"outputs/models/{PLANT_ID}/sfm/images_masked")
OUT_MASK_DIR = Path(f"outputs/models/{PLANT_ID}/sfm/masks")

# Try better model for objects/plants
session = new_session("isnet-general-use")


def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(IN_DIR.glob("*.jpg"))
    print(f"[INFO] Found {len(paths)} images")

    kernel = np.ones((5, 5), np.uint8)

    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = remove(rgb, session=session)

        rgba = np.array(result)
        mask = rgba[:, :, 3]

        # ---- Make mask LESS aggressive ----

        # 1. remove tiny holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 2. expand edges to keep thin leaves
        mask = cv2.dilate(mask, kernel, iterations=1)

        # 3. optional feather (soft edge)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # threshold back to binary
        _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

        cv2.imwrite(str(OUT_MASK_DIR / p.name), mask)

        masked = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(str(OUT_IMG_DIR / p.name), masked)

        if (i + 1) % 20 == 0:
            print(f"[INFO] {i+1}/{len(paths)} done")

    print("[DONE] Masks saved to:", OUT_MASK_DIR.resolve())
    print("[DONE] Masked images saved to:", OUT_IMG_DIR.resolve())


if __name__ == "__main__":
    main()