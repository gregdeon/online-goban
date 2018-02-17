# detect-board.py
import cv2
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from board_kernels import *

# Global: set to False to disable debug images
debug = True

# Taken from Dan's helpers
def imshow(title, img):
    # hide the x and y axis for images
    plt.axis('off')
    # RGB images are actually BGR in OpenCV, so convert before displaying
    if len(img.shape) == 3: 
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # otherwise, assume it's grayscale and just display it
    else:
        plt.imshow(img, cmap='gray')
    # add a title if specified
    plt.title(title)
    plt.show()
    #plt.close()

def contourTouchesEdge(cnt, img_size):
    # ex: contourTouchesEdge(contours[i], (800, 600))
    [x, y, w, h] = cv2.boundingRect(cnt)
    
    x_min = 0
    y_min = 0
    x_max = img_size[0] - 1
    y_max = img_size[1] - 1
    
    return (x <= x_min or y <= y_min or x+w >= x_max or y+h >= y_max)
    
def contourAsPoly(cnt):
    epsilon = 0.01*cv2.arcLength(cnt, True)
    poly = cv2.approxPolyDP(cnt, epsilon, True)
    return poly
    
def polyIsQuad(poly):
    # Check for 4 corners
    if len(poly) != 4:
        return False
    
    # Check for convex contour
    return cv2.isContourConvex(poly)

def detect_board(img):
    img_h = len(img)
    img_w = len(img[0])
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_bw, 128, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3, 3))
    img_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel, iterations=10)
    
    kernel_edges = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    img_edges = cv2.filter2D(img_closed, -1, kernel_edges)
    
#    img_blur = cv2.blur(img_edges, (5, 5))
    
#    [num, labels, stats, centroids] = cv2.connectedComponentsWithStats(img_blur, 4)
#    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(num)]
#    print(areas)
#    
    _, contours, hierarchy = cv2.findContours(
        img_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    i_best = -1
    area_best = -1
    for i in range(len(contours)):
        if contourTouchesEdge(contours[i], (img_w, img_h)):
            continue
        if hierarchy[0][i][3] != -1:
            continue
            
        poly = contourAsPoly(contours[i])
        if not polyIsQuad(poly):
            continue
        area = cv2.contourArea(contours[i])
        
        if area > area_best:
            i_best = i
            area_best = area
     
#    print(poly)
    poly = contourAsPoly(contours[i_best])
    img_output = np.copy(img)
    cv2.drawContours(img_output, contours, i_best, (0, 255, 0), 10)
    
    if debug:
#        imshow("Board image", img)
#        imshow("Greyscale", img_bw)
#        imshow("Thresholded", img_thresh)
#        imshow("Closed", img_closed)
#        imshow("Edges", img_edges)
#        imshow("Contours", img_output)
        pass
        
    return poly[:, 0]

# Taken from 
# pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# and 
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
    
def four_point_transform(image, pts):
 # obtain a consistent order of the points and unpack them
 # individually
 rect = order_points(pts)
 (tl, tr, br, bl) = rect
 
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
    
    # TODO: document
 maxWidth = maxHeight = 380
 dst = np.array([
  [0, 0],
  [maxWidth - 1, 0],
  [maxWidth - 1, maxHeight - 1],
  [0, maxHeight - 1]], dtype = "float32")
 
 # compute the perspective transform matrix and then apply it
 M = cv2.getPerspectiveTransform(rect, dst)
 warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
 # return the warped image
 return warped
    
def correct_board(img, corners):
    img_warped = four_point_transform(img, corners)
    
    if debug:
#        imshow("Warped", img_warped)
        pass
    
    return img_warped
    
def findAverageColourPoint(img, x, y, radius):
    roi = img[y-radius:y+radius+1, x-radius:x+radius+1]
    
    # circular mask
    oy,ox = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (ox**2 + oy**2 <= radius**2).astype(np.uint8)
    
    return cv2.mean(roi, mask=mask)
    
    
def findAverageColours(img):
    img_output = np.full((380, 380, 3), 255, np.uint8)
    
    for ix in range(19):
        for iy in range(19):
            x = 10 + 20*ix
            y = 10 + 20*iy
            
            colour = findAverageColourPoint(img, x, y-2, 3)
            cv2.circle(img_output, (x, y), 7, colour, -1)
            cv2.circle(img_output, (x, y), 7, (0, 0, 0), 1)
    
    return img_output
    
def findEdges(img_board):
    # Test edge detector
    kernel_edges = np.array([
        [-1, 0, -1],
        [ 0, 4,  0],
        [-1, 0, -1]
    ])    
    
    kernel_deriv2 = np.array([
        [-1,  0,   0, 0, 0],
        [ 0, 16,   0, 0, 0],
        [ 0,  0, -30, 0, 0],
        [ 0,  0,   0, 16, 0],
        [ 0,  0,   0, 0, -1]
    ])
        
    #kernel_edges = np.array([
    #    [-1, -1, -1, -1, -1],
    #    [-1, -1, -1, -1, -1],
    #    [-1,  0 , 24, -1, -1],
    #    [-1, -1, -1, -1, -1],
    #    [-1, -1, -1, -1, -1]
    #])
    x = y = 0
    roi = img_board[y:y+20, x:x+20]
    img_deriv = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img_edges = cv2.filter2D(img_deriv, -1, kernel_deriv2)
    _, thresh_edges = cv2.threshold(img_edges, 600, 255, cv2.THRESH_BINARY)
    imshow("Edge detected", thresh_edges)
    
def scatterColors(img):
    scale_down = 3
    img = img[::scale_down, ::scale_down]
    
#    scale = 1/scale_down
#    img = cv2.resize(img, None, fx=scale, fy=scale)

    
    height = len(img)
    width = len(img[0])
    
    
    img_flat = img.reshape((width*height, 3))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv_flat = img_hsv.reshape((width*height, 3))
    
    imshow("Small", img)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    C = img_flat[:, ::-1] / 255
    R = img_flat[:, 2]
    G = img_flat[:, 1]
    B = img_flat[:, 0]
    ax.scatter(R, G, B, c=C)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    H = img_hsv_flat[:, 0]
    S = img_hsv_flat[:, 1]
    V = img_hsv_flat[:, 2]
    ax.scatter(H, S, V, c=C)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')

    plt.show()
   
def detectIntersection(img, x, y):
    # Detect whether a position on the image is black stone/white stone/empty
    if x < 0 or x >= 19:
        raise ValueError("Expected x in range [0, 18]; got %s" % x)
      
    if y < 0 or y >= 19:
        raise ValueError("Expected y in range [0, 18]; got %s" % y)
    
    (roi_x, roi_y) = (x*20, y*20)
    (roi_w, roi_h) = (20, 20)
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_bw = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    _, roi_thresh = cv2.threshold(roi_bw, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)    
    
    # new plan
    kernels = getKernels(len(roi))
    """
    kernel_intersection = np.array([
        [-3, -2, -1, 0, 1, 0, -1, -2, -3],
        [-2, -1,  0, 0, 1, 0,  0, -1, -2],
        [-1,  0,  0, 0, 1, 0,  0,  0, -1],
        [ 0,  0,  0, 0, 1, 0,  0,  0,  0],
        [ 1,  1,  1, 1, 2, 1,  1,  1,  1],
        [ 0,  0,  0, 0, 1, 0,  0,  0,  0],
        [-1,  0,  0, 0, 1, 0,  0,  0, -1],
        [-2, -1,  0, 0, 1, 0,  0, -1, -2],
        [-3, -2, -1, 0, 1, 0, -1, -2, -3],
    ])
    """
    """
    kernel_intersection = np.array([
        [-2, 0, 0, 0, 0, 0, -2],
        [ 0, 0, 0, 0, 0, 0,  0],
        [ 0, 0, 0, 0, 0, 0,  0],
        [ 0, 0, 0, 2, 0, 0,  0],
        [ 0, 0, 0, 0, 0, 0,  0],
        [ 0, 0, 0, 0, 0, 0,  0],
        [-2, 0, 0, 0, 0, 0, -2]
    ])
    """
    
    #intersections = cv2.filter2D(roi_thresh.astype(np.float32), -1, kernel_cross)
    
    #_, intersect_thresh = cv2.threshold(intersections, 255*30, 255, cv2.THRESH_BINARY)
    #intersect_thresh = intersect_thresh.astype(np.uint8)
#    print(roi_thresh.astype(np.float32))
    has_stone = True
    print("")
    for kernel in kernels:
        filtered = np.sum(np.multiply(roi_thresh.astype(np.float32), kernel))
        print(filtered)
        is_empty = (filtered > -30*255)
        if(is_empty):
            has_stone = False
            break
    
    #print(is_cross)
    fill_val = 0 if has_stone else 255
    intersect_thresh = np.full(np.shape(roi_thresh), fill_val, dtype=np.uint8)
    
    #roi_out = np.copy(roi)
    #roi_out[:, :, 1] |= intersect_thresh
    #for line in lines:
    #    line = line[0]
    #    cv2.line(roi_out, (line[0], line[1]), (line[2], line[3]), (0, 255, 0))
    #imshow("Edges", edges_thresh)
    
    
    #imshow("ROI", roi)
    #imshow("Thresholded", roi_thresh)
    #imshow("Intersections", intersect_thresh)
    #imshow("Detected", roi_out)
    return intersect_thresh
    
def buildThreshImage(img):
    img_output = np.zeros((380, 380), dtype=np.uint8)

    for x in range(19):
        for y in range(19):
            thresh = detectIntersection(img, x, y)
            (roi_x, roi_y) = (x*20, y*20)
            (roi_w, roi_h) = (20, 20)  
            img_output[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] += thresh
    imshow("Output", img_output)
        
def detectStonesInBoard(img):
    radius = 5
    threshold = 5

    board = np.zeros((19, 19))
    
    for x in range(19):
        for y in range(19):
            thresh = detectIntersection(img, x, y)
            
            roi = thresh[10-radius:10+radius+1, 10-radius:10+radius+1]
    
            # circular mask
            oy,ox = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = (ox**2 + oy**2 <= radius**2).astype(np.uint8)
            
            int_masked = cv2.bitwise_and(roi, roi, mask=mask)
            #imshow("Thresh", thresh)
            #imshow("Masked", int_masked)
            
            # Check for intersections
            num_intersection = np.count_nonzero(int_masked)
            if num_intersection < threshold:
                # It's a stone
                # Find colour
                avg_colour = findAverageColourPoint(img, 20*x + 10, 20*y + 8, 3)
                avg_brightness = np.average(avg_colour)
                if avg_brightness > 128:
                    board[y][x] = -1
                else:
                    board[y][x] = 1
    
    return board

def drawBoard(board):
    img_board = np.zeros((380, 380, 3), dtype=np.uint8)
    
    # Set background colour
    img_board[:] = [86, 170, 255]
    
    # Draw lines
    for i in range(19):
        pos = 10 + 20*i
        cv2.line(img_board, (pos, 10), (pos, 370), (0, 0, 0), 1)
        cv2.line(img_board, (10, pos), (370, pos), (0, 0, 0), 1)
        
    # Draw star points
    star_pos = [3, 9, 15]
    star_radius = 4
    for iy in star_pos:
        for ix in star_pos:
            y = 10 + 20*iy
            x = 10 + 20*ix
            cv2.circle(img_board, (x, y), star_radius, (0, 0, 0), -1)
            
            
    # Draw stones
    for iy in range(19):
        for ix in range(19):
            if board[iy][ix] == 0:
                continue
            
            stone_color = (255, 255, 255)
            if board[iy][ix] == 1:
                stone_color = (0, 0, 0)
            
            y = 10 + 20*iy
            x = 10 + 20*ix
            
            cv2.circle(img_board, (x, y), 10, stone_color, -1)
            cv2.circle(img_board, (x, y), 10, (0, 0, 0), 1)
            
    imshow("Board", img_board)
 
def printBoard(board):
    for y in range(19):
        str = ""
        for x in range(19):
            str += "%2d " % board[y][x]
        print(str)

def countErrors(board, exp):
    num = 0
    
    for y in range(19):
        for x in range(19):
            if board[y][x] != exp[y][x]:
                num += 1
                
    return num
    
test_cases = [
    # Working
    {
        'fname': 'images/IMG_20180205_212736509.jpg',
        'board': [
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  1,  0,  1,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  1,  0,  0],
[0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ]
    },
    
    # Missing one stone (offset)
    # Change point for average colour?
    # Change threshold for intersections?
    {
        'fname': 'images/IMG_20180205_212841658.jpg',
        'board': [
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
[0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  1,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ]
    },
    
    {
        'fname': 'images/IMG_20180205_212834421.jpg',
        'board': [
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
[0,  0,  1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
[0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  1,  0,  1,  1,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
[0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ]
    }
]

# Thanks to https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
def detectCircles(img):
    # Get grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bright and dark portions of image
    mid = np.average(gray)
    extremes = np.abs(gray.astype(np.int32) - mid).astype(np.uint8)
    _, ext_thresh = cv2.threshold(extremes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Clean up image
    kernel = np.full((3, 3), 1)
    img_closed = cv2.morphologyEx(ext_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        img_opened, 
        cv2.HOUGH_GRADIENT, 
        dp = 1, 
        minDist = 16, 
        param1 = 10,
        param2 = 5,
        minRadius = 7,
        maxRadius = 12
    )
    print(circles)
    
    if circles is None:
        print("Warning: detected no circles in detectCircles()")
        return
        
    circles = np.round(circles[0, :]).astype("int")
    if debug:
        imshow("Extremes", extremes)
        imshow("Thresholded", ext_thresh)
        imshow("Cleaned", img_opened)
        
        output = np.copy(img)
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        imshow("Detected", output)
        
    return circles

# Detect the board state by finding circles in the image
def detectBoardCircles(img):
    # Find all the circles
    circles = detectCircles(img)
    
    # Find in board
    board = np.zeros((19, 19))
    for (x, y, r) in circles:
        # Find closest position 
        xb = x // 20
        yb = y // 20
        
        # Find which colour of stone it is
        avg_colour = findAverageColourPoint(img, x, y, r//2)
        avg_brightness = np.average(avg_colour)
        
        if avg_brightness > 128:
            board[yb][xb] = -1
        else:
            board[yb][xb] = 1
    
    return board
    
if __name__ == "__main__":
    test_case = 2
    fname = test_cases[test_case]['fname']
    board_exp = test_cases[test_case]['board']
    
    img = cv2.imread(fname)
    
    # Missing one stone (offset)
    # Change point for average colour??????
    # Change threshold for intersections???
    #img = cv2.imread('images/IMG_20180205_212841658.jpg')
    
    # Finds board contour
    # Puts points in wrong order?
    #img = cv2.imread('images/IMG_20180205_212834421.jpg')
    
    
    corners = detect_board(img)
    print(corners)
    corrected = correct_board(img, corners)
    
    #colours = findAverageColours(corrected)
    
    #imshow("Original", img)
    imshow("Corrected", corrected)
    #imshow("Reconstructed", colours)
    
    # Test: 
    #scatterColors(corrected)
    
    # Detect with intersection methods
    #buildThreshImage(corrected)
    #board = detectStonesInBoard(corrected)
    #drawBoard(board)
    #printBoard(board)
    #print("Errors: %d" % countErrors(board, board_exp))
    

    
    #detectIntersection(corrected, 0, 0)
    #detectIntersection(corrected, 1, 0)
    #detectIntersection(corrected, 0, 1)
    #detectIntersection(corrected, 1, 1)
    #detectIntersection(corrected, 3, 2)
    #detectIntersection(corrected, 3, 3)
    #detectIntersection(corrected, 4, 4)
    
    # Detect with circle method
    board = detectBoardCircles(corrected)
    drawBoard(board)
    printBoard(board)
    print("Errors: %d" % countErrors(board, board_exp))
    #detectCircles(corrected)
        
#    cv2.waitKey(0)