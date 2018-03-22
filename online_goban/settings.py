# settings.py
# Global settings for online-goban

# Which camera to use
CAM_ID = 0

# Should the camera auto-focus?
CAM_AUTOFOCUS = False

# Camera brightness
# None for auto-adjusting
CAM_GAIN = None

# Board position in camera frame
# Get these coordinates from
# > python online-goban.py calibrate
CALIBRATION_DATA = [
    [82, 94],
    [455, 39],
    [500, 423],
    [193, 403],
]

# Set DEBUG = True to show image processing pipeline
DEBUG = True