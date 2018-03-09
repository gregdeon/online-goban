# Utilities that are helpful in several modules
import cv2
import numpy as np

# Connect to webcam
# Make sure to call cam.release() at the end!
# gain: if None, use auto-exposure; otherwise, use fixed gain
# autofocus: if True, use autofocus; if False, use fixed focus 
def connectToCamera(cam_id, gain=None, autofocus=True):
    cam = cv2.VideoCapture(cam_id)
    
    # Disable auto exposure
    if gain is not None:
        cam.set(15, gain)
    
    # Disable auto focus
    if not autofocus:
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
    return cam

# Draw a board from a 19x19 array
#  0 = empty
#  1 = black
# -1 = white
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
            
    return img_board
    imshow("Board", img_board)

# Pretty-print the board    
def printBoard(board):
    for y in range(19):
        str = ""
        for x in range(19):
            str += "%2d " % board[y][x]
        print(str)