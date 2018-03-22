# Online Go Server Connection
The program `connect_socket.js` is a standalone Node.js program that handles connecting to the Online Go Server API. You can run it like
```
node connect_socket.js
```

Then, this program will listen to stdin for authentication details. The first line you should send is a comma-separated list with a username, a user ID, an real-time API authentication key for this account, and the ID of the game to connect to. For example, this might look like
```
gregdeon,123456,authkeygoeshere,12345678
```

Finally, you can play a move by sending a line with two letters for the coordinates. For example, to play the upper-left 3-3 point, send the string `cc`. This lettering does not skip the letter `I`, so `ss` is the bottom-right corner. It is not possible to make any actions other than regular moves (ie: passing, resigning, requesting undos, etc).  

While the program is running, it can print 3 different types of lines:

1. `connected`: once the connection to the game is ready
2. `.`: every 200 ms (this is a workaround for some Python limitations)
3. A 2-character move whenever a stone is played in the game. For example, `cc` is the top-left 3-3 point. The move `zz` indicates that the opponent passed.

Note that `online_goban.py` handles all of this input and output automatically, so regular users should not need to worry about any of this.
