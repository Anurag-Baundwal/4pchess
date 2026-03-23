const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const readline = require('readline');
const path = require('path');

// --- Engine exe file path ---
const ENGINE_PATH = 'C:/Users/dell3/Downloads/tt_0.177release/tt_0.177release.exe';

let engineProcess = null;
let currentFen = null;
let latestEval = null;

function ensureEngineRunning() {
    if (engineProcess) return;

    console.log("Starting engine process...");
    engineProcess = spawn(ENGINE_PATH);

    // Initialize engine
    engineProcess.stdin.write("post\n");
    engineProcess.stdin.write("cores 8\n");

    const rl = readline.createInterface({
        input: engineProcess.stdout,
        terminal: false
    });

    rl.on('line', (line) => {
        // Engine output format: "Depth Score Time Nodes PV..."
        // e.g. "14 256 86 880042 Bh3xi2 Kh1-g1 b8-c8 Qd9-g6"
        const match = line.match(/^(\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
        
        if (match) {
            latestEval = {
                depth: parseInt(match[1], 10),
                score: parseInt(match[2], 10),
                pv: match[5].trim().split(/\s+/) // handles 1 move or many moves flawlessly
            };
        }
    });

    engineProcess.on('close', () => {
        console.log("Engine process exited.");
        engineProcess = null;
    });
}

function algebraicToCoord(sq) {
    const col = 'abcdefghijklmn'.indexOf(sq.charAt(0));
    const row = 14 - parseInt(sq.slice(1), 10);
    return { row, col };
}

function formatResponse(evalData, turnColorChar) {
    if (!evalData) return {};

    const turnMap = { 'R': 0, 'B': 1, 'Y': 2, 'G': 3 };
    let currentTurn = turnMap[turnColorChar];

    // Engine score is from the perspective of the side doing the search.
    // GUI requires score relative to Red/Yellow.
    let score = evalData.score;
    if (turnColorChar === 'B' || turnColorChar === 'G') {
        score = -score;
    }

    let pvArray = [];
    
    // Slice the PV array so we only draw AT MOST 4 arrows (the immediate next ply for each color)
    // If the PV only contains 1 move, slice(0, 4) safely returns just that 1 move.
    const topMoves = evalData.pv.slice(0, 4);

    for (let move of topMoves) {
        // Extracts the raw squares from variations like: Qc5xQl5, m11-k11, Bh3xi2, h10-h11=Q
        const match = move.match(/[A-Z]?([a-n]\d+)[-x][A-Z]?([a-n]\d+)(?:=[A-Z])?\+?#?/);
        if (match) {
            pvArray.push({
                turn: currentTurn,
                from: algebraicToCoord(match[1]),
                to: algebraicToCoord(match[2])
            });
        }
        currentTurn = (currentTurn + 1) % 4; // Advance to the next player's turn
    }

    return {
        evaluation: score,
        zero_move_evaluation: score, // UI expects this for debugging display
        search_depth: evalData.depth,
        principal_variation: pvArray
    };
}

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

/* POST API to evaluate the board. */
router.post('/chess-api', function(req, res, next) {
    const req_json = req.body;
    const fen = req_json.fen;
    const turnColorChar = req_json.turnColorChar; // 'R', 'B', 'Y', 'G'

    ensureEngineRunning();

    if (fen && fen !== currentFen) {
        currentFen = fen;
        latestEval = null; // Clear old eval data
        
        // Stop current search, change position, and start infinite analysis
        engineProcess.stdin.write("force\n"); 
        engineProcess.stdin.write(`setboard ${fen}\n`);
        // We use st 99999 + go to simulate infinite analysis flawlessly on older xboard variants
        engineProcess.stdin.write("st 99999\n"); 
        engineProcess.stdin.write("go\n");
    }

    // Instantly return whatever the engine has calculated so far
    if (latestEval) {
        res.json(formatResponse(latestEval, turnColorChar));
    } else {
        res.json({});
    }
});

module.exports = router;