import subprocess
from pathlib import Path

PLANT_ID = "001"
COLMAP = "COLMAP.bat"

BASE = Path(f"outputs/models/{PLANT_ID}/sfm")
IMAGES = BASE / "images_masked"
DB = BASE / "database.db"
SPARSE = BASE / "sparse"


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    SPARSE.mkdir(parents=True, exist_ok=True)

    # Feature extraction
    run([
        COLMAP,
        "feature_extractor",
        "--database_path", str(DB),
        "--image_path", str(IMAGES),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.estimate_affine_shape", "1",
        "--SiftExtraction.domain_size_pooling", "1"
    ])

    # Sequential matcher (FAST test)
    run([
        COLMAP,
        "sequential_matcher",
        "--database_path", str(DB),
        "--SequentialMatching.overlap", "20"
    ])

    # Mapper
    run([
        COLMAP,
        "mapper",
        "--database_path", str(DB),
        "--image_path", str(IMAGES),
        "--output_path", str(SPARSE),
        "--Mapper.min_num_matches", "15"
    ])

    print("\n[DONE] Sparse SfM complete")


if __name__ == "__main__":
    main()