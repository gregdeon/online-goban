# calibration.py
# Functions for calibrating camera to board position
import cv2
import numpy as np

from online_goban.utils import *
from online_goban.detect_board import *
import online_goban.settings as settings

# Calibrate the camera by finding the Go board
def calibrate():
    # Connect to camera
    cam = connectToCamera(
        settings.CAM_ID,
        settings.CAM_GAIN,
        settings.CAM_AUTOFOCUS
    )

    # Wait until spacebar
    img = None
    while True: 
        # Take a picture
        ret, img = cam.read()

        # Orient correctly
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)
        cv2.imshow("Raw", img)

        key = cv2.waitKey(1000 // 30)
        if key == ord(' '):
            break

    # Done taking pictures - release everything now
    cam.release()
    cv2.destroyAllWindows()

    # Find the four corners of the board
    corners = findCorners(img)

    # Print calibration data
    corners_ordered = order_points(corners)
    print("Corners: ")
    for corner in corners_ordered:
        x = corner[0]
        y = corner[1]
        print("    [%d, %d]," % (x, y))
    print("Copy corners into CALIBRATION_DATA in settings.py")

# Shared global variables for corner selection
calibration_corners = []
mouse_pos = (0, 0)

# Prompt user to click corners
def findCorners(img):
    global calibration_corners
    global mouse_pos

    # Save corners
    calibration_corners = []

    # Display a window for selecting the corners
    cv2.namedWindow("Calibration")
    cv2.namedWindow("Calibration Detail")
    cv2.setMouseCallback("Calibration", handleCalibrationClick)

    while True:
        # Make a clone of the image for drawing
        img_clone = np.copy(img)

        # Draw the selected points and lines between them
        cv2.circle(img_clone, mouse_pos, 4, (0, 0, 255), 1)
        for i in range(len(calibration_corners)):
            x, y = calibration_corners[i]
            cv2.circle(img_clone, (x, y), 4, (0, 0, 255), 1)

            if i > 0:
                xp, yp = calibration_corners[i-1]
                cv2.line(img_clone, (x, y), (xp, yp), (0, 0, 255), 1)

            if i == 0 or i == len(calibration_corners) - 1:
                cv2.line(img_clone, (x, y), mouse_pos, (0, 0, 255), 1)

        # Show our progress
        cv2.imshow("Calibration", img_clone)

        # Show zoomed version too
        roi_size_in = 20
        roi_size_out = 200
        (img_h, img_w, _) = np.shape(img_clone)
        (mouse_x, mouse_y) = mouse_pos
        roi_x1 = np.clip(mouse_x - roi_size_in, 0, None)
        roi_x2 = np.clip(mouse_x + roi_size_in, None, img_w-1)
        roi_y1 = np.clip(mouse_y - roi_size_in, 0, None)
        roi_y2 = np.clip(mouse_y + roi_size_in, None, img_h-1)

        roi = img_clone[roi_y1:roi_y2, roi_x1:roi_x2]
        roi = cv2.resize(roi, (roi_size_out, roi_size_out))
        cv2.imshow("Calibration Detail", roi)

        # If we have 4 corners, stop
        key = cv2.waitKey(1000 // 60)
        if len(calibration_corners) >= 4:
            break
    cv2.destroyAllWindows()

    return np.array(calibration_corners)

# Handle a mouse movement or click event
def handleCalibrationClick(event, x, y, flags, params):
    global calibration_corners
    global mouse_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_corners.append([x, y])
    mouse_pos = (x, y)
