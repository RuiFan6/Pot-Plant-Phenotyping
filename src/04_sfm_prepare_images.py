import shutil
import argparse
from pathlib import Path


def copy_with_prefix(src_dir, dst_dir, prefix):
    imgs = sorted(src_dir.glob("*.jpg"))
    print(f"[INFO] {prefix}: {len(imgs)} frames")

    for p in imgs:
        new_name = f"{prefix}_{p.name}"
        shutil.copy2(p, dst_dir / new_name)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id

    base = Path(f"data/processed/{PLANT_ID}/frames")
    out = Path(f"outputs/models/{PLANT_ID}/sfm/images")

    out.mkdir(parents=True, exist_ok=True)

    copy_with_prefix(base / f"{PLANT_ID}_top", out, "top")
    copy_with_prefix(base / f"{PLANT_ID}_front", out, "front")
    copy_with_prefix(base / f"{PLANT_ID}_low", out, "low")

    print(f"[DONE] Images prepared in {out.resolve()}")


if __name__ == "__main__":
    main()