import cv2
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plant_id", required=True)
    args = parser.parse_args()

    PLANT_ID = args.plant_id
    IMG_DIR = Path(f"outputs/models/{PLANT_ID}/sfm/images")

    # ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    paths = sorted(IMG_DIR.glob("*.jpg"))

    print(f"Testing on first 20 images...\n")

    for p in paths[:20]:
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            print(f"{p.name} -> detected IDs: {ids.flatten()}")
        else:
            print(f"{p.name} -> no markers")


if __name__ == "__main__":
    main()