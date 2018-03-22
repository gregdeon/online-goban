# detect_board.py
# Functions for detecting game state from an image of the board

import cv2
import numpy as np
from scipy.spatial import distance as dist

import online_goban.settings as settings

# Put 4 points in the order [top-left, top-right, bottom-right, bottom-left]
# Helper function for board detection
# Taken from pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
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

# Perform a perspective transform on an image
# Helper function for board detection
# Adapted from pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts, output_size):
    # obtain a consistent order of the points and unpack them
    # individually
    #rect = order_points(pts)
    #(tl, tr, br, bl) = rect
    (tl, tr, br, bl) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    maxWidth = maxHeight = output_size 
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ],
    dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

# Return a 19x19 array containing the game state
# 0 = empty; 1 = black; -1 = white
def detectBoard(img):
    # Warp image
    calib_points = np.array(settings.CALIBRATION_DATA, dtype = "float32")
    img_warped = four_point_transform(img, calib_points, 380)

    # Get grayscale image
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
    
    # Blur the image
    blur_size = 20*8
    blur_kernel = np.ones((blur_size, blur_size)) / (blur_size ** 2)
    blurred = cv2.filter2D(gray, -1, blur_kernel)

    # Get differences between blurred and original
    diff = gray.astype(np.int32) - blurred
    
    # Filter out the positive and negative parts separately
    white = np.clip( diff, 0, 255).astype(np.uint8)
    black = np.clip(-diff, 0, 255).astype(np.uint8)
    
    # Threshold the parts
    _, black_thresh = cv2.threshold(black, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, white_thresh = cv2.threshold(white, 50, 255, cv2.THRESH_BINARY)
    
    # Clean up images with morphological filters
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # White image: get rid of reflections
    white_opened = cv2.morphologyEx(white_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    white_mask = white_opened

    # Black image: remove lines
    black_opened = cv2.morphologyEx(black_thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Remove reflections
    black_closed = cv2.morphologyEx(black_opened, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    # Remove intersections
    black_opened_2 = cv2.morphologyEx(black_closed, cv2.MORPH_OPEN, kernel_small, iterations=4)
    black_mask = black_opened_2

    # Make a board array - default to empty intersections
    board = np.zeros((19, 19))

    # Look through masks
    for y in range(19):
        for x in range(19):
            # Get ROIs
            center_x = 11 + 20*x
            center_y = 11 + 20*y

            roi_x1 = center_x - 5
            roi_x2 = center_x + 5
            roi_y1 = center_y - 5
            roi_y2 = center_y + 5

            roi_w = white_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_b = black_mask[roi_y1:roi_y2, roi_x1:roi_x2]

            # Set thresholds for detecting stones
            thresh_count = 20

            # Get pixel counts
            count_w = np.sum(roi_w) // 255
            count_b = np.sum(roi_b) // 255

            # Mark as stone
            if count_w > thresh_count:
                board[y][x] = -1
            elif count_b > thresh_count:
                board[y][x] = 1

    # Show the image pipeline if debugging is enabled
    if settings.DEBUG:
        cv2.imshow("White", white)
        cv2.imshow("Black", black)
        cv2.imshow("White Thresh", white_thresh)
        cv2.imshow("Black Thresh", black_thresh)
        cv2.imshow("White Mask", white_mask)
        cv2.imshow("Black Mask", black_mask)

    return board
