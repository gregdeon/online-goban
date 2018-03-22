# play.py
# Main loop for playing game

import cv2
import numpy as np

from online_goban.ogs_client import *
from online_goban.detect_board import *
from online_goban.utils import *
import online_goban.settings as settings

# Attempt to play a move
# If there is exactly 1 new stone on the local board,
# submit it to the online game
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

    # If local board doesn't have exactly 1 added stone, fail
    if num_new_stones != 1:
        print("Warning: tried to play move with %d new stones on board (expected 1)" % num_new_stones)
        return

    # Try to play the 1 added stone
    [x, y] = new_move
    client.playMove(x, y)

# Main game loop
# Display local board and differences with online board
# Play moves as indicated by user 
def play(game_id):
    # Connect to client
    ogs_client = OGSClient(game_id)
    ogs_client.connect()

    # Connect to camera
    cam = connectToCamera(
        settings.CAM_ID,
        settings.CAM_GAIN,
        settings.CAM_AUTOFOCUS
    )

    # Get initial board states 
    ogs_client.readBoard()
    online_board = ogs_client.board
    local_board = None

    while True:
        # Get frame from the camera
        ret, img = cam.read()

        # Mirror to get user view of board
        # Future work: make this configurable in settings
        img = cv2.flip(img, 0)
        img = cv2.flip(img, 1)

        # Show the video frame
        cv2.imshow("Video", img)

        # Get board state from image
        local_board = detectBoard(img)

        # Draw the board
        img_board = drawBoard(local_board, online_board)
        cv2.imshow("Board", img_board)

        # Handle keypresses
        key = cv2.waitKey(1000 // 30)
        if key == 27:
            # Stop when pressing esc
            break
        elif key == ord(' '):
            # Submit move when pressing space
            findAndPlayMove(ogs_client, online_board, local_board)

        # Wait for message
        line = ogs_client.getRealTimeMessage()
        if line is not None and line != 'connected':
            # Update the board
            ogs_client.addToBoard(line)
            online_board = ogs_client.board

    cv2.destroyAllWindows()
    cam.release()
