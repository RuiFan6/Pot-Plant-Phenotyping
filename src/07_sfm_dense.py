import argparse
import subprocess
from pathlib import Path


# ------------------------------------------------------------
# Run helper
# ------------------------------------------------------------

def run(cmd):

    print("\n[RUN]")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plant_id",
        required=True,
        help="Plant ID (e.g. 001,002,003)"
    )

    parser.add_argument(
        "--colmap",
        default="COLMAP.bat",
        help="Path to COLMAP executable"
    )

    args = parser.parse_args()

    plant_id = args.plant_id
    COLMAP = args.colmap

    BASE = Path(f"outputs/models/{plant_id}/sfm")

    IMAGES = BASE / "images_masked"
    SPARSE = BASE / "sparse/0"
    DENSE = BASE / "dense"

    fused = DENSE / "fused.ply"

    print("\n========================================")
    print("COLMAP Dense Reconstruction")
    print("Plant:", plant_id)
    print("========================================\n")

    # ------------------------------------------------------------
    # Step 1: Image undistortion
    # ------------------------------------------------------------

    if not (DENSE / "images").exists():

        print("[STEP] image_undistorter")

        run([
            COLMAP,
            "image_undistorter",
            "--image_path", str(IMAGES),
            "--input_path", str(SPARSE),
            "--output_path", str(DENSE),
            "--output_type", "COLMAP"
        ])

    else:

        print("[SKIP] image_undistorter (already done)")


    # ------------------------------------------------------------
    # Step 2: PatchMatch stereo
    # ------------------------------------------------------------

    depth_dir = DENSE / "stereo/depth_maps"
    depth_maps = list(depth_dir.glob("*.bin"))

    if len(depth_maps) == 0:

        print("\n[STEP] patch_match_stereo")

        run([
            COLMAP,
            "patch_match_stereo",
            "--workspace_path", str(DENSE),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true"
        ])

    else:

        print("[SKIP] patch_match_stereo (depth maps exist)")


    # ------------------------------------------------------------
    # Step 3: Stereo fusion
    # ------------------------------------------------------------

    if not fused.exists():

        print("\n[STEP] stereo_fusion")

        run([
            COLMAP,
            "stereo_fusion",
            "--workspace_path", str(DENSE),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(fused)
        ])

    else:

        print("[SKIP] stereo_fusion (fused.ply exists)")


    print("\n========================================")
    print("Dense reconstruction complete")
    print("Output:", fused)
    print("========================================\n")


# ------------------------------------------------------------

if __name__ == "__main__":
    main()