const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const readline = require('readline');

// --- Engine exe file path ---
const ENGINE_PATH = 'C:/Users/dell3/source/repos6/stockfish_4pc/stockfish_4pc.exe';

let engineProcess = null;
let currentFen = null;
let latestEval = null;

function ensureEngineRunning() {
    if (engineProcess) return;

    console.log("Starting Stockfish 4PC UCI engine process...");
    engineProcess = spawn(ENGINE_PATH);

    // Initialize engine using UCI protocol
    engineProcess.stdin.write("uci\n");
    engineProcess.stdin.write("setoption name hash value 2048\n");
    engineProcess.stdin.write("setoption name threads value 8\n");
    engineProcess.stdin.write("isready\n");

    const rl = readline.createInterface({
        input: engineProcess.stdout,
        terminal: false
    });

    rl.on('line', (line) => {
        // UCI info line example:
        // "info depth 14 seldepth 18 multipv 1 score cp 84 nodes 590103 ... pv h2h3 b9c9 e13e12 m10l10"
        if (line.startsWith('info ') && line.includes(' pv ')) {
            // Added \b (word boundary) to prevent matching "seldepth" and "multipv"
            const depthMatch = line.match(/\bdepth\s+(\d+)/);
            const scoreCpMatch = line.match(/\bscore\s+cp\s+(-?\d+)/);
            const scoreMateMatch = line.match(/\bscore\s+mate\s+(-?\d+)/);
            const pvMatch = line.match(/\bpv\s+(.*)$/);
            
            if (depthMatch && pvMatch && (scoreCpMatch || scoreMateMatch)) {
                let scoreObj = null;
                if (scoreCpMatch) {
                    scoreObj = { type: 'cp', value: parseInt(scoreCpMatch[1], 10) };
                } else if (scoreMateMatch) {
                    scoreObj = { type: 'mate', value: parseInt(scoreMateMatch[1], 10) };
                }

                latestEval = {
                    depth: parseInt(depthMatch[1], 10),
                    score: scoreObj,
                    pv: pvMatch[1].trim().split(/\s+/) // Splits the PV into an array of moves
                };
            }
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
    let scoreValue = evalData.score.value;
    const isMate = evalData.score.type === 'mate';

    if (turnColorChar === 'B' || turnColorChar === 'G') {
        scoreValue = -scoreValue;
    }

    // Format the score properly: numbers for centipawns, strings like "M3" or "-M5" for mates
    let finalScoreDisplay;
    if (isMate) {
        finalScoreDisplay = scoreValue > 0 ? `M${scoreValue}` : `-M${Math.abs(scoreValue)}`;
    } else {
        finalScoreDisplay = scoreValue;
    }

    let pvArray = [];
    
    // Slice the PV array so we only draw AT MOST 4 arrows (the immediate next ply for each color)
    // If the PV only contains 1 move, slice(0, 4) safely returns just that 1 move.
    const topMoves = evalData.pv.slice(0, 4);

    for (let move of topMoves) {
        // Extracts the raw squares from pure UCI notation: h2h3, m10l10, h10h11q
        // Match Groups: 1 = From (e.g. m10), 2 = To (e.g. l10), 3 = Optional promotion
        const match = move.match(/^([a-n]\d+)([a-n]\d+)([a-z])?$/i);
        if (match) {
            pvArray.push({
                turn: currentTurn,
                from: algebraicToCoord(match[1]),
                to: algebraicToCoord(match[2]),
                promotion: match[3] ? match[3].toLowerCase() : null
            });
        }
        currentTurn = (currentTurn + 1) % 4; // Advance to the next player's turn
    }

    return {
        evaluation: finalScoreDisplay,
        zero_move_evaluation: finalScoreDisplay, // UI expects this for debugging display
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
        
        // Stop current search, change position, and start infinite analysis via UCI
        engineProcess.stdin.write("stop\n"); 
        engineProcess.stdin.write(`position fen ${fen}\n`);
        engineProcess.stdin.write("go infinite\n");
    }

    // Instantly return whatever the engine has calculated so far
    if (latestEval) {
        res.json(formatResponse(latestEval, turnColorChar));
    } else {
        res.json({});
    }
});

module.exports = router;