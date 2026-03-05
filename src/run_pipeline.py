import argparse
import subprocess
from pathlib import Path


PIPELINE = [
    "03_extract_frames.py",
    "04_sfm_prepare_images.py",
    "05_mask_background.py",
    "06_sfm_sparse.py",
    "07_sfm_dense.py",
    "08_aruco_scale.py",
    "09_align_to_turntable.py",
    "10_extract_plant_only.py",
    "11_clean_pointcloud.py",
    "12_traits_basic.py",
    "13_traits_advanced.py",
    "14_extract_fruit.py",
    "15_traits_fruit.py"
]


def run(cmd):

    print("\n====================================")
    print("RUN:", " ".join(cmd))
    print("====================================\n")

    subprocess.run(cmd, check=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)

    args = parser.parse_args()

    plant_id = args.plant_id

    src = Path(__file__).parent

    print("\n====================================")
    print("Plant phenotyping pipeline")
    print("Plant:", plant_id)
    print("====================================")

    for script in PIPELINE:

        script_path = src / script

        run([
            "python",
            str(script_path),
            "--plant_id",
            plant_id
        ])

    print("\n====================================")
    print("Pipeline finished")
    print("====================================")


if __name__ == "__main__":
    main()