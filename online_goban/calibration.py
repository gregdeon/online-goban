# Functions for calibration
import cv2
import numpy as np

from scipy.spatial import distance as dist
from online_goban.utils import *
from online_goban.settings import *

# Calibrate the camera by finding the Go board
def calibrate():
    # Connect to camera
    cam = connectToCamera(
        CAM_ID,
        CAM_GAIN,
        CAM_AUTOFOCUS
    )

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

    cam.release()
    cv2.destroyAllWindows()


    # Find the four corners of the board
    corners = findCorners(img)

    # Sort them
    corners_ordered = order_points(corners)

    # Save the points into a file
    # TODO: just print for now?
    print("Corners: ")
    for corner in corners_ordered:
        x = corner[0]
        y = corner[1]
        print("    [%d, %d]," % (x, y))
#        print(corner)
    print("Copy corners into CALIBRATION_DATA in settings.py")

# Shared global variable for corners
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

        # Show zoom too
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

# Taken from 
# pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def order_points(pts):
 # sort the points based on their x-coordinates
 xSorted = pts[np.argsort(pts[:, 0]), :]
 
 # grab the left-most and right-most points from the sorted
 # x-roodinate points
 leftMost = xSorted[:2, :]
 rightMost = xSorted[2:, :]
 
 # now, sort the left-most coordinates according to their
 # y-coordinates so we can grab the top-left and bottom-left
 # points, respectively
 leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
 (tl, bl) = leftMost
 
 # now that we have the top-left coordinate, use it as an
 # anchor to calculate the Euclidean distance between the
 # top-left and right-most points; by the Pythagorean
 # theorem, the point with the largest distance will be
 # our bottom-right point
 D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
 (br, tr) = rightMost[np.argsort(D)[::-1], :]
 
 # return the coordinates in top-left, top-right,
 # bottom-right, and bottom-left order
 return np.array([tl, tr, br, bl], dtype="float32")

def handleCalibrationClick(event, x, y, flags, params):
    global calibration_corners
    global mouse_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        calibration_corners.append([x, y])
    mouse_pos = (x, y)

def test():
    cam = connectToCamera(1, gain=-8.0, autofocus=False)
    cv2.namedWindow("Raw", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Raw",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        cv2.imshow("Raw", img)
        key = cv2.waitKey(1000 // 60)
        if key == 27:
            break

    cv2.destroyAllWindows()
    cam.release()