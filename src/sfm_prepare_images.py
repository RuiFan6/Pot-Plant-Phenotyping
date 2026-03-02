import shutil
from pathlib import Path

PLANT_ID = "001"

def copy_with_prefix(src_dir, dst_dir, prefix):
    imgs = sorted(src_dir.glob("*.jpg"))
    print(f"[INFO] {prefix}: {len(imgs)} frames")

    for p in imgs:
        new_name = f"{prefix}_{p.name}"
        shutil.copy2(p, dst_dir / new_name)


def main():
    base = Path(f"data/processed/{PLANT_ID}/frames")
    out = Path(f"outputs/models/{PLANT_ID}/sfm/images")

    out.mkdir(parents=True, exist_ok=True)

    copy_with_prefix(base / "001_top", out, "top")
    copy_with_prefix(base / "001_front", out, "front")
    copy_with_prefix(base / "001_low", out, "low")

    print(f"[DONE] Images prepared in {out.resolve()}")


if __name__ == "__main__":
    main()