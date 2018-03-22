# ogs_client.py
# Class for communicating with JS process
# Uses separate node.js program for socket.io communication

import numpy as np
import requests
from oauthlib.oauth2 import LegacyApplicationClient
from requests_oauthlib import OAuth2Session
import subprocess

import online_goban.ogs_auth as auth

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

        # Current state of the online board
        self.board = np.zeros((19, 19))
        self.next_color = 1

    # Connect to the server
    def connect(self):
        # URL for getting tokens
        token_url = 'https://online-go.com/oauth2/token/'

        # Inputs for refreshing an expired token
        refresh_args = {
            'client_id': auth.OGS_CLIENT_ID,
            'client_secret': auth.OGS_CLIENT_SECRET,
        }

        # Make session
        self.client = OAuth2Session(
            client = LegacyApplicationClient(
                client_id = auth.OGS_CLIENT_ID
            ),
            auto_refresh_url = token_url,
            auto_refresh_kwargs = refresh_args,
            token_updater = self._updateToken
        )

        # Fetch an access token
        token = self.client.fetch_token(
            token_url = token_url, 
            username = auth.OGS_USERNAME,
            password = auth.OGS_PASSWORD,
            client_id = auth.OGS_CLIENT_ID,
            client_secret = auth.OGS_CLIENT_SECRET,
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
            auth.OGS_USERNAME,
            str(auth.OGS_USER_ID),
            auth.OGS_USER_AUTH,
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

    # Get a list of (x,y) coordinates adjacent to a given position
    # Doesn't list coordinates outside of the board
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

    # Get the status of the group with a stone at (x, y)
    # Returns a list of stone positions and the number of liberties
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
                    # Remove captured stones
                    for [xi, yi] in group:
                        self.board[yi][xi] = 0  

            self.board[y][x] = self.next_color
        self.next_color = -self.next_color


    # Attempt to play a move
    def playMove(self, x, y):
        # Get string version of move
        move = self._convertCoordsToMove(x, y)

        # Send it to the server
        self.process.stdin.write((move + "\n").encode('utf-8'))
        self.process.stdin.flush()

    # Get a message from the real-time client
    # There are 3 types of lines:
    # - "connected": when ready to play
    # - ".": no moves to report
    # - "ab": the position of a move
    # We return None in the first two cases
    def getRealTimeMessage(self):
        line = self.process.stdout.readline().rstrip().decode('utf-8')
        print(line)
        if line.startswith(".") or line.startswith("connected"):
            return None
        else:
            return line
