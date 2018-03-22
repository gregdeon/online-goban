# online_goban.py
# Main program for Online Goban
import argparse

from online_goban.play import play
from online_goban.calibration import calibrate

import traceback

# Main function for playing game
def main_play(args):
    print("Playing game at " + args.game_id)
    play(args.game_id)

# Calibration controller
# Future work: add automatic calibration
def main_calibrate(args):
    print("Calibrating camera")
    calibrate()
    

if __name__ == "__main__":
    # Set up main parser
    parser = argparse.ArgumentParser(description = 'Main program for Online Goban')
    subparsers = parser.add_subparsers(help = "see 'online_goban.py <command> -h' for detailed help")
    
    # Set up parser for 'play' command
    main_parser = subparsers.add_parser(
        'play', 
        description = 'Connect to a live game on online-go.com'
    )
    main_parser.set_defaults(cmd=main_play)
    main_parser.add_argument('game_id', help = 'ID of game (from URL online-go.com/game/<game_id>)')
    
    # Set up parser for 'calibrate'
    calibrate_parser = subparsers.add_parser('calibrate')
    calibrate_parser.set_defaults(cmd=main_calibrate)
    calibrate_parser.add_argument(
        '-m, --manual', 
        help='Manually find position of board'
    )
    
    args = parser.parse_args()
    try:
        # Try to start program
        args.cmd(args)
    except AttributeError as e:
        # Probably used invalid command line arguments 
        parser.print_help()

        # Debugging help: print any exceptions
        traceback.print_exc()
