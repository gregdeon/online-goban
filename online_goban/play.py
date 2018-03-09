# play.py
import cv2
import numpy as np

from online_goban.utils import *
from online_goban.settings import *

debug = True

# Transform an image
# Adapted from pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def four_point_transform(image, pts):
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
    
    # TODO: document 380 px size
    maxWidth = maxHeight = 380 
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

def findAverageColourPoint(img, x, y, radius):
    roi = img[y-radius:y+radius+1, x-radius:x+radius+1]
    
    # circular mask
    oy,ox = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (ox**2 + oy**2 <= radius**2).astype(np.uint8)
    
    return cv2.mean(roi, mask=mask)

# Thanks to https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/ for circle detection code
def detectBoardCircles(img):

    # Get grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bright and dark portions of image
    mid = np.average(gray)
    diff = gray.astype(np.int32) - mid
    

    # Blur the image instead
    blur_size = 20*4
    blur_kernel = np.ones((blur_size, blur_size)) / (blur_size ** 2)
    blurred = cv2.filter2D(gray, -1, blur_kernel)

    blurred_rgb = cv2.filter2D(img, -1, blur_kernel)
    cv2.imshow("Blurred RGB", blurred_rgb)
    #cv2.imshow("Blurred", blurred)
    #extremes = np.abs(gray.astype(np.int32) - mid).astype(np.uint8)
    #_, ext_thresh = cv2.threshold(extremes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    diff = gray.astype(np.int32) - blurred
    
    # TODO: how to pick this automatically?
    move = 0
    white = np.clip( diff-move, 0, 255).astype(np.uint8)
    black = np.clip(-diff+move, 0, 255).astype(np.uint8)
    #cv2.imshow("White", white)
    cv2.imshow("Black", black)
    
    # Threshold
    #_, white_thresh = cv2.threshold(white, 0, 255, cv2.THRESHESH_BINARY | cv2.THRESH_OTSU)
    #_, black_thresh = cv2.threshold(black, 30, 255, cv2.THRESH_BINARY)
    _, black_thresh = cv2.threshold(black, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, white_thresh = cv2.threshold(white, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("White Thresh", white_thresh)
    #cv2.imshow("Black Thresh", black_thresh)
    
    # Clean up images
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    white_opened = cv2.morphologyEx(white_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#    white_closed = cv2.morphologyEx(white_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    white_clean = white_opened
    # TODO: need to close this?
    # white_closed = cv2.morphologyEx(white_opened, cv2.MORPH_CLOSE, kernel, iterations=3)
    #imshow("White Closed", white_closed)

    # Remove lines
    black_opened = cv2.morphologyEx(black_thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Remove reflections
    black_closed = cv2.morphologyEx(black_opened, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # Remove intersections
    black_opened_2 = cv2.morphologyEx(black_closed, cv2.MORPH_OPEN, kernel_small, iterations=4)

    black_clean = black_opened_2
    #cv2.imshow("Black Opened", black_opened)
    #cv2.imshow("Black Closed", black_closed)
    cv2.imshow("White Mask", white_clean)

    # Combine
    mask = cv2.bitwise_or(black_clean, white_clean)

    
    
    # Detect circles
    circles = cv2.HoughCircles(
        mask, 
        cv2.HOUGH_GRADIENT, 
        dp = 1, 
        minDist = 16, 
        param1 = 10,
        param2 = 6,
        minRadius = 7,
        maxRadius = 11
    )

    # Make a board array
    board = np.zeros((19, 19))
    
    # If we didn't find anything, don't do anything to the board
    if circles is None:
        print("Warning: detected no circles in detectCircles()")
        return board
        
    # Look at each circle
    (img_h, img_w, _) = np.shape(img)
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # If the circle goes off the board, skip it
        r_col = r//2
        if x - r_col < 0 or \
           x + r_col >= img_w or \
           y - r_col < 0 or \
           y + r_col >= img_h:
            continue

        # Find closest position 
        xb = np.clip(x // 20, 0, 18)
        yb = np.clip(y // 20, 0, 18)
        
        # Find which colour of stone it is
        avg_colour = findAverageColourPoint(img, x, y, r_col)
        avg_brightness = np.average(avg_colour)
        avg_neighbourhood = blurred[y][x]
        
        if avg_brightness > avg_neighbourhood:
            board[yb][xb] = -1
        else:
            board[yb][xb] = 1

        
        
    if debug:
        cv2.imshow("Mask", mask)

        output = np.copy(img)
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.imshow("Detected", output)

    return board

"""
# Detect the board state by finding circles in the image
def detectBoardCircles(img):
    # Get image size
    (img_h, img_w, _) = np.shape(img)

    # Find all the circles
    circles = detectCircles(img)
    
    # Find in board
    if circles is not None:
        for 

    
    return board
"""

def play():
    cam = connectToCamera(
        CAM_ID,
        CAM_GAIN,
        CAM_AUTOFOCUS
    )

    cv2.namedWindow("Video")

    calib_points = np.array(CALIBRATION_DATA, dtype = "float32")
    prev_board = None

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)

        # Warp image
        img_warped = four_point_transform(img, calib_points)

        # Convert to board
        board = detectBoardCircles(img_warped)

        img_board = drawBoard(board, prev_board)

        cv2.imshow("Video", img)
        cv2.imshow("Board", img_board)
        #cv2.imshow("Warped", img_warped)

        key = cv2.waitKey(1000 // 30)
        # Stop when pressing esc
        if key == 27:
            break
        # Save position when pressing space
        elif key == ord(' '):
            prev_board = board

    cv2.destroyAllWindows()
    cam.release()