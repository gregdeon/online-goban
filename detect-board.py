# detect-board.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
     
    print(poly)
    poly = contourAsPoly(contours[i_best])
    img_output = np.copy(img)
    cv2.drawContours(img_output, poly, -1, (0, 255, 0), 10)
    
    if debug:
#        imshow("Board image", img)
#        imshow("Greyscale", img_bw)
#        imshow("Thresholded", img_thresh)
#        imshow("Closed", img_closed)
#        imshow("Edges", img_edges)
#        imshow("Blurred", img_blur)
#        imshow("Contours", img_output)
        pass
        
    return poly[:, 0]

# Taken from 
# pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	print(rect)
	return rect
    
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
    img_hsv_flat = img.reshape((width*height, 3))
    
    imshow("Small", img)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    C = img_flat[:, ::-1] / 255
#    R = img_flat[:, 2]
#    G = img_flat[:, 1]
#    B = img_flat[:, 0]
#    ax.scatter(R, G, B, c=C)
#    ax.set_xlabel('R')
#    ax.set_ylabel('G')
#    ax.set_zlabel('B')
    H = img_hsv_flat[:, 0]
    S = img_hsv_flat[:, 1]
    V = img_hsv_flat[:, 2]
    ax.scatter(H, S, V, c=C)
    ax.set_xlabel('H')
    ax.set_ylabel('S')
    ax.set_zlabel('V')

    plt.show()
    
if __name__ == "__main__":
    img = cv2.imread('images/IMG_20180205_212736509.jpg')
    
    corners = detect_board(img)
    print(corners)
    corrected = correct_board(img, corners)
    
    colours = findAverageColours(corrected)
    
    #imshow("Original", img)
    #imshow("Reconstructed", colours)
    scatterColors(corrected)
    
#    cv2.waitKey(0)