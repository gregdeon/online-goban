# Main program for Online Goban
import argparse

# Main function for playing game
def play(args):
    print("Playing game at " + args.url)
    print("TODO: connect to game")

# Calibration controller
def calibrate(args):
    print("TODO: Calibrate camera and projector")
    

if __name__ == "__main__":
    # Set up main parser
    parser = argparse.ArgumentParser(description = 'Main program for Online Goban')
    subparsers = parser.add_subparsers(help = "see 'online_goban.py <command> -h' for detailed help")
    
    # Set up parser for 'play' command
    main_parser = subparsers.add_parser(
        'play', 
        description = 'Connect to a live game on online-go.com'
    )
    main_parser.set_defaults(cmd=play)
    main_parser.add_argument('url', help = 'Game URL (online-go.com/game/<game_id>)')
    
    # Set up parser for 'calibrate'
    calibrate_parser = subparsers.add_parser('calibrate')
    calibrate_parser.set_defaults(cmd=calibrate)
    calibrate_parser.add_argument(
        '-m, --manual', 
        help='Manually find position of board'
    )
    
    args = parser.parse_args()
    args.cmd(args)