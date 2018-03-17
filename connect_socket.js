// Run this program like
// node connect_socket.js
// gregdeon,519752,bcf755d8f1d63b004faf3dbb3bcb8eb0,12063461
// dc
// stdout will show moves played

// Authentication info
var ogs_auth = require('./ogs_auth');

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
rl.on('line', function(line){
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

rl.on('close', function(){
    socket.close();
});


// var move = process.argv[3];

// For debugging
function println(msg) {
    process.stdout.write(msg + "\n");
}

// Give authentication information
function startConnect() {
    socket = io.connect("https://online-go.com", {
        reconnection: true,
        reconnectionDelay: 500,
        reconnectionDelayMax: 60000,
        transports: ["websocket"]
    });
    socket.on('connect', authenticate);
    socket.on('game/'+game_id+'/gamedata', finishConnect);
    socket.on('game/'+game_id+'/move', handleMove);
}

function authenticate()
{
    socket.emit("authenticate", {auth: auth_key, player_id: user_id, username: username});
    socket.emit("game/connect", {game_id: game_id, player_id: user_id, chat: false});
}

function finishConnect() {
    connected = true;
    println("connected");
}

// Play a move encoded like "dc"
function playMove(move) {
    socket.emit("game/move", {game_id: game_id, player_id: user_id, move: move} );
}

function handleMove(resp) {
    var alphabet = "abcdefghijklmnopqrsz";
    //println(JSON.stringify(resp, null, 4));
    var move = resp["move"];
    if(move[0] == -1)
    {
        println("zz");
    }
    else
    {   
        var ret = alphabet[move[0]] + alphabet[move[1]];
        println(ret);
    }
}