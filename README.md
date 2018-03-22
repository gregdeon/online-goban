# Online Goban
Online Goban uses a single webcam and an unmodified Go set to play games on the Online Go server. This is a course project for Dan Vogel's course [CS889: Applied Computer Vision for Human-Computer Interaction](https://cs.uwaterloo.ca/~dvogel/cs889w18/) at the University of Waterloo. 

At a high level, this program does two things. First, it detects stone positions on a Go board using a simple computer vision pipeline. Second, it connects to the Online Go Server API to submit moves and show the opponent's moves.

# Setup
## Installation
This package uses Python 3 and Node.js. Required packages are listed in `requirements.txt` (Python) and `js/package.json` (Javascript).

## Hardware and Calibration
Before playing a game, there are several settings that need to be prepared in `online_goban/settings.py`.

First, Online Goban needs access to a single webcam. In practice, this could be any camera that OpenCV can use. The `CAM_ID` in `settings.py` allows any regular webcam to be selected. `CAM_ID = 0` uses the computer's default camera.

You can use `CAM_AUTOFOCUS` and `CAM_GAIN` to control the camera if autofocus or autoexposure are causing problems. These settings might not work for all cameras - this depends on OpenCV support.

To calibrate the camera to the Go board, run the command
```
python online_goban.py calibrate
```
This calibration has three steps:

1. Ensure that the camera has a good view of the board. The camera should be position so that the entire board is in the frame and there is a minimal amount of glare from overhead lights. It is not necessary for the camera to have a perfect overhead view of the board - a moderate angle is okay. Press `SPACE` when this is done.
2. Select the four corners of the board by clicking on them. The order of the points does not matter. A zoomed view is provided to make this process easier. 
3. The program will print a list of 4 points to the console. Copy these into `settings.py`.

Here is one example of a moderate camera angle that worked well:

![Example camera angle](img/example_camera.png)

## Online Go Authentication
Finally, Online Goban needs some credentials in order to connect to games and submit moves on your behalf. These settings are details in `online_goban/ogs_auth.py`. Note that several of these settings are secret and should not be made public (for example, by committing them to a public Github repository).

# Playing Games
Once the setup is complete, you can connect to an existing game by running a command like
```
python online_goban.py play 12345
```
where `12345` is replaced with the ID of your game. (For example, this command will attempt to connect to `https://online-go.com/game/12345`.)

After you're connected to a game, the screen shows the current state of the Go board: 

![Example game state](img/game_ui.png)

Stones that are on the physical board but not played online are highlighted in green. Stones that are online but missing from the physical board are highlighted in red. To play a move, make sure that exactly one new stone (green highlight) is on the board and press `SPACE`.

# Future Ideas
Things to do from here:

- Improve the UI. This program is effectively a proof-of-concept and it would be straightforward to give more feedback or a real GUI during calibration and games.
- Improve the stone detection. The algorithms from [PhotoKifu](https://arxiv.org/abs/1508.03269) or [VideoKifu](https://arxiv.org/abs/1701.05419) appear to be the state of the art.
- Automatically detect moves and play them without keyboard input. Some hand detection code would work here, but this is relatively difficult because gobans and hands are similar in colour.
- Add a projector to the system and display more information on the board. These projections could show the opponent's moves or identify captured stones. This would require another calibration step and some care to ensure the stone detection algorithms are affected by the projections.