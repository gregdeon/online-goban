# play.py
import cv2
import numpy as np

# Web API
import requests
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session
import subprocess

from online_goban.utils import *
from online_goban.settings import *
from online_goban.ogs_auth import *

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

def detectBoardSpirals(img):
    # Get grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bright and dark portions of image
    mid = np.average(gray)
    diff = gray.astype(np.int32) - mid
    

    # Blur the image instead
    blur_size = 20*8
    blur_kernel = np.ones((blur_size, blur_size)) / (blur_size ** 2)
    blurred = cv2.filter2D(gray, -1, blur_kernel)

    blurred_rgb = cv2.filter2D(img, -1, blur_kernel)
    #cv2.imshow("Blurred RGB", blurred_rgb)
    #cv2.imshow("Blurred", blurred)
    #extremes = np.abs(gray.astype(np.int32) - mid).astype(np.uint8)
    #_, ext_thresh = cv2.threshold(extremes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    diff = gray.astype(np.int32) - blurred
    cv2.imshow("Diff", diff)

    #diff_white 

    _, diff_white = cv2.threshold(diff.astype(np.int16), 0, 255, cv2.THRESH_TOZERO)
    _, diff_black = cv2.threshold((-diff).astype(np.int16), 0, 255, cv2.THRESH_TOZERO)
    #_, diff_black = cv2.threshold(-diff, 0, 255, cv2.THRESH_BINARY)
    cv2.imshow("Diff White", diff_white.astype(np.uint8))
    cv2.imshow("Diff Black", diff_black.astype(np.uint8))

    # Make a board array
    board = np.zeros((19, 19))

    # Look through masks
    spiral_rad = 4
    for y in range(19):
        for x in range(19):
            # Get ROIs
            center_x = 11 + 20*x
            center_y = 11 + 20*y

            pix_list = []

            # Spiral
            num_angles = 64
            for i in range(num_angles):
                #rad = i / num_angles * spiral_rad
                rad = spiral_rad
                theta = (i / num_angles + 0.125) * 2 * np.pi 
                xi = int(center_x + rad * np.cos(theta))
                yi = int(center_y + rad * np.sin(theta))

                pix_list.append(img[yi, xi])

            pix_list = np.array(pix_list).astype(np.float32)

            # Get averages
            avg_pix = np.average(pix_list)

            # Get differences

            pix_diff = np.diff(pix_list, axis=0)
            pix_diff_total = np.sum(np.abs(pix_diff), axis=1)
            diff_sum = np.sum(pix_diff_total)


            print(x, y, avg_pix, diff_sum)    
#            print(pix_list)
#            print(pix_diff)
#            print(pix_diff_total)
#            print(diff_sum)


            # Get zero-crossings
            zero_crossings = np.sum(np.diff(np.sign(pix_list)) != 0)

            # If there are too few zero-crossings, this is a stone
            min_zero_crossings = 4
            if 0 < x < 18:
                min_zero_crossings += 2
            if 0 < y < 18:
                min_zero_crossings += 2

            if zero_crossings < min_zero_crossings:
                # Mark as stone
                if avg_pix > 0:
                    board[y][x] = -1
                else:
                    board[y][x] = 1

    return board

# Thanks to https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/ for circle detection code
def detectBoardCircles(img):

    # Get grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bright and dark portions of image
    mid = np.average(gray)
    diff = gray.astype(np.int32) - mid
    

    # Blur the image instead
    blur_size = 20*8
    blur_kernel = np.ones((blur_size, blur_size)) / (blur_size ** 2)
    blurred = cv2.filter2D(gray, -1, blur_kernel)

    blurred_rgb = cv2.filter2D(img, -1, blur_kernel)
    #cv2.imshow("Blurred RGB", blurred_rgb)
    #cv2.imshow("Blurred", blurred)
    #extremes = np.abs(gray.astype(np.int32) - mid).astype(np.uint8)
    #_, ext_thresh = cv2.threshold(extremes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    diff = gray.astype(np.int32) - blurred
    
    # TODO: how to pick this automatically?
    move = 0
    white = np.clip( diff-move, 0, 255).astype(np.uint8)
    black = np.clip(-diff+move, 0, 255).astype(np.uint8)
    cv2.imshow("White", white)
    cv2.imshow("Black", black)
    
    # Threshold
    #_, white_thresh = cv2.threshold(white, 0, 255, cv2.THRESHESH_BINARY | cv2.THRESH_OTSU)
    #_, black_thresh = cv2.threshold(black, 30, 255, cv2.THRESH_BINARY)
    _, black_thresh = cv2.threshold(black, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, white_thresh = cv2.threshold(white, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow("White Thresh", white_thresh)
    cv2.imshow("Black Thresh", black_thresh)
    
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
    cv2.imshow("Black Mask", black_clean)

    # Make a board array
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

            roi_w = white_clean[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_b = black_clean[roi_y1:roi_y2, roi_x1:roi_x2]

            # Set thresholds for detecting stones
            # 32/100
            thresh_count = 20

            # Get pixel counts
            count_w = np.sum(roi_w) // 255
            count_b = np.sum(roi_b) // 255

            # Mark as stone
            if count_w > thresh_count:
                board[y][x] = -1
            elif count_b > thresh_count:
                board[y][x] = 1
    """
    # Combine
    mask = cv2.bitwise_or(black_clean, white_clean)

    # 
    
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
        #cv2.imshow("Mask", mask)

        output = np.copy(img)
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        #cv2.imshow("Detected", output)
    """    

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

# Class for keeping track of connections to OGS
class OGSClient(object):
    def __init__(self, game_id):
        # OGS ID of our game
        self.game_id = game_id

        # OAuth2 client
        self.client = None

        # Access token for OAuth2
        self.client_token = None

        # Subprocess running real-time connections
        self.process = None

        # Current board
        self.board = np.zeros((19, 19))
        self.next_color = 1

    # Connect to the web API and return an authenticated client
    # Return type is an OAuth2Session
    def connect(self):
        # URL for getting tokens
        token_url = 'https://online-go.com/oauth2/token/'

        # Inputs for refreshing an expired token
        refresh_args = {
            'client_id': OGS_CLIENT_ID,
            'client_secret': OGS_CLIENT_SECRET,
        }

        # Make session
        self.client = OAuth2Session(
            client = LegacyApplicationClient(
                client_id = OGS_CLIENT_ID
            ),
            auto_refresh_url = token_url,
            auto_refresh_kwargs = refresh_args,
            token_updater = self._updateToken
        )


        # Fetch an access token
        token = self.client.fetch_token(
            token_url = token_url, 
            username = OGS_USERNAME,
            password = OGS_PASSWORD,
            client_id = OGS_CLIENT_ID,
            client_secret = OGS_CLIENT_SECRET,
        )
        self._updateToken(token)

        # Also, start running the real-time client
        self.process = subprocess.Popen(
            'node ./js/connect_socket',
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        # Send it our credentials
        auth_string = ",".join([
            OGS_USERNAME,
            str(OGS_USER_ID),
            OGS_USER_AUTH,
            str(self.game_id),
        ]) + "\n"
        self.process.stdin.write(auth_string.encode('utf-8'))
        self.process.stdin.flush()

        
    # Update our API token 
    def _updateToken(self, new_token):
        print("DEBUG: updating token")
        print(new_token)
        self.client_token = new_token

    # Get the entire game state from the server
    def _getGameState(self):
        game_url = 'https://online-go.com/api/v1/games/' + self.game_id + '/'
        r = self.client.get(game_url)
        return r.json()

    # Convert (2, 3) into "cd"
    # Special case: (-1, -1) becomes "zz"
    def _convertCoordsToMove(self, x, y):
        letter_list = 'abcdefghijklmnopqrs'
        if x < 0 and y < 0:
            return "zz"
        else:
            return letter_list[x] + letter_list[y]

    # Convert "cd" to (2, 3)
    # Special case: "zz" becomes (-1, -1)
    def _convertMoveToCoords(self, move):
        letter_list = 'abcdefghijklmnopqrsz'
        x = letter_list.index(move[0])
        y = letter_list.index(move[1])

        if x < 19 and y < 19:
            return (x, y)
        else:
            return (-1, -1)

    # Get a 19x19 array of the current game state
    # 0 = empty; 1 = black; -1 = white
    def readBoard(self):
        # Get the list of moves
        game_state = self._getGameState()
        move_list = game_state['gamedata']['moves']

        # Build the board
        self.board = np.zeros((19, 19))

        # Start as black
        self.next_color = 1

        # Add each move
        for [x, y, _] in move_list:
            move = self._convertCoordsToMove(x, y)
            self.addToBoard(move)

    def getNeighbours(self, x, y):
        ret = []
        if x > 0:
            ret.append([x-1, y])
        if x < 18:
            ret.append([x+1, y])

        if y > 0:
            ret.append([x, y-1])
        if y < 18:
            ret.append([x, y+1])

        return ret

    def getGroup(self, x, y):
        if not (0 <= x < 19 and 0 <= y < 19):
            return None

        col = self.board[y][x]
        reached = set()
        group = []
        frontier = [[x, y]]
        liberties = 0

        while len(frontier) > 0:
            # Get the next intersection to check
            [xi, yi] = frontier.pop()

            # If we've already been here, stop
            if (xi, yi) in reached:
                continue

            # Mark that we've reached it
            reached.add((xi, yi))

            # If it's blank, add a liberty
            if self.board[yi][xi] == 0:
                liberties += 1

            # If it's the right colour, add to the group and iterate
            if self.board[yi][xi] == col:
                group.append([xi, yi])

                n_list = self.getNeighbours(xi, yi)
                for [xn, yn] in n_list:
                    frontier.append([xn, yn])

        return [group, liberties]

    # Check if a single intersection has val
    # Set val = 0 for empty check
    def checkBoardValue(self, x, y, val):
        if 0 <= x < 19 and 0 <= y < 19:
            return self.board[y][x] == val
        else:
            return False

    # Add a move to the board
    # Move is formatted like "cd" (ie: [a-s][a-s])
    def addToBoard(self, move):
        print(move)
        (x, y) = self._convertMoveToCoords(move)

        # Ignore passes
        if 0 <= x < 19 and 0 <= y < 19:
            # Check for captured groups
            n_list = self.getNeighbours(x, y)

            for [xn, yn] in n_list:
                # Can't capture own stones
                if self.board[yn][xn] == self.next_color:
                    continue

                # Check if opposing group has 1 liberty
                [group, liberties] = self.getGroup(xn, yn)
                if liberties == 1:
                    print(group)
                    for [xi, yi] in group:
                        self.board[yi][xi] = 0  

            self.board[y][x] = self.next_color
        self.next_color = -self.next_color


    # Attempt to play a move
    def playMove(self, x, y):
        move = self._convertCoordsToMove(x, y)
        self.process.stdin.write((move + "\n").encode('utf-8'))
        self.process.stdin.flush()

    # Get a message from the real-time client
    def getRealTimeMessage(self):
        line = self.process.stdout.readline().rstrip().decode('utf-8')
        print(line)
        if line.startswith(".") or line.startswith("connected"):
            return None
        else:
            return line

def findAndPlayMove(client, online_board, local_board):
    # Find differences between boards
    num_new_stones = 0
    new_move = None
    for y in range(19):
        for x in range(19):
            local_stone = local_board[y][x]
            online_stone = online_board[y][x]
            if local_stone != 0 and online_stone == 0:
                new_move = [x, y]
                num_new_stones += 1

    # If local board doesn't have 1 added stone, fail
    if num_new_stones != 1:
        print("Warning: tried to play move with %d new stones on board (expected 1)" % num_new_stones)
        return

    # Try to play the 1 added stone
    [x, y] = new_move
    client.playMove(x, y)

def play(game_id):
    # Connect to client
    ogs_client = OGSClient(game_id)
    ogs_client.connect()

    # Connect to camera
    cam = connectToCamera(
        CAM_ID,
        CAM_GAIN,
        CAM_AUTOFOCUS
    )
    # TODO: do I need this?
    cv2.namedWindow("Video")
    calib_points = np.array(CALIBRATION_DATA, dtype = "float32")

    # Board states: 
    ogs_client.readBoard()
    online_board = ogs_client.board
    local_board = None

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)

        # Warp image
        img_warped = four_point_transform(img, calib_points)

        # Convert to board
        local_board = detectBoardCircles(img_warped)

        img_board = drawBoard(local_board, online_board)

        cv2.imshow("Video", img)
        cv2.imshow("Board", img_board)
        #cv2.imshow("Warped", img_warped)

        key = cv2.waitKey(1000 // 30)
        # Stop when pressing esc
        if key == 27:
            break
        # Submit move when pressing space
        elif key == ord(' '):
            findAndPlayMove(ogs_client, online_board, local_board)
        # Refresh position from online when pressing R
        # TODO: trigger this with WS messages

        # Check for messages
        line = ogs_client.getRealTimeMessage()
        if line is not None and line != 'connected':
            # Update the board
            ogs_client.addToBoard(line)
            online_board = ogs_client.board

    cv2.destroyAllWindows()
    cam.release()

def play_offline():
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