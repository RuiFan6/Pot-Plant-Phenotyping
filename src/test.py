import cv2

img = cv2.imread("outputs/aruco_board.png")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

corners, ids, rejected = detector.detectMarkers(img)

print("Detected:", len(ids) if ids is not None else 0)