// connect_socket.js
// Handle the OGS socket.io connection

// Run this program like
// > node connect_socket.js
// Then, input one string into stdin like 
// > username,user_id,real_time_auth_key,game_id
// Finally, send moves via stdin like
// > dc
// stdout will show:
// - "connected": when connection ready
// - ".": when no moves in 200 ms
// - "dc": when move is placed at position "dc"

// Socket.IO library
var io = require('socket.io-client');

// Read lines from stdin
var readline = require('readline');
var rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Socket.IO connection
var socket = null;

// Whether's we're connected to the web API or not yet
var connected = false;

// Authentication info
var auth_key = null;
var user_id = null;
var username = null;

// Game info
var game_id = null;

// Handle one line of input
rl.on('line', function(line) {
    // First line: connect
    if(!connected)
    {
        // Split line into inputs
        var arr = line.trim().split(",");

        username = arr[0];
        user_id = parseInt(arr[1]);
        auth_key = arr[2];
        game_id = parseInt(arr[3]);

        startConnect();
    }

    // Other lines: play a move
    else
    {
        var move = line.trim();
        playMove(move);
    }
});

// When streams close, we should stop too
rl.on('close', function() {
    socket.close();
});

// Print a line to stdout
function println(msg) {
    try {
        process.stdout.write(msg + "\n");
    }
    catch (error) {
        // The Python program must have closed
        process.exit();
    }
}

// Print a line to stderr
// Helpful for debugging
function printerr(msg) {
    process.stderr.write(msg + "\n");
}

// Start connecting to the server
function startConnect() {
    socket = io.connect("https://online-go.com", {
        reconnection: true,
        reconnectionDelay: 500,
        reconnectionDelayMax: 60000,
        transports: ["websocket"]
    });

    // Send authentication info when we're connected
    socket.on('connect', authenticate);

    // Finish our prep when we get a gamedata object
    socket.on('game/'+game_id+'/gamedata', finishConnect);

    // Pass moves to stdout when we see them
    socket.on('game/'+game_id+'/move', handleMove);
}

// Give authentication information to the server and connect to game
function authenticate() {
    socket.emit("authenticate", {auth: auth_key, player_id: user_id, username: username});
    socket.emit("game/connect", {game_id: game_id, player_id: user_id, chat: false});
}

// Perform final setup after receiving game data object
function finishConnect() {
    connected = true;
    println("connected");

    setInterval(function(){
        println(".");
    }, 200);
}

// Play a move encoded like "dc"
function playMove(move) {
    socket.emit("game/move", {game_id: game_id, player_id: user_id, move: move} );
}

// Receive a move from the server
function handleMove(resp) {
    var alphabet = "abcdefghijklmnopqrsz";
    var move = resp["move"];

    // If the move is a pass, print "zz"
    if(move[0] == -1)
    {
        println("zz");
    }

    // Otherwise, print the move's coordinates
    else
    {   
        var ret = alphabet[move[0]] + alphabet[move[1]];
        println(ret);
    }
}
