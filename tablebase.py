# tablebase.py
# small tablebase for a few common start positions

import random

# Maps a tuple of previous moves to the recommended next move.
# The key is a tuple of moves in algebraic notation (e.g., ('h2-h3', 'b7-c7')).
# The value is the best move to play (str) or a list of moves for random selection.
MOVES_TO_BEST_MOVE = {
    (): 'h2-h3',
    ('h2-h3',): 'b7-c7',
    ('h2-h3', 'b7-c7'): 'g13-g12',
    ('h2-h3', 'b7-c7', 'g13-g12'): 'm8-l8',
    # Alternative lines based on player choices
    ('f2-f3',): 'b7-c7',
    ('h2-h3', 'b7-c7', 'i13-i12'): 'm8-l8',
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8'): ['g1-j4', 'g1-k5', 'f2-f3'], # Multiple good options - we randomly pick one of them
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8', 'i1-c7', 'b6-c7'): 'f14-i11',
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8', 'g1-k5'): 'b9-c9',
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8', 'f2-f3'): 'b4-d4',
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8', 'i1-c7'): 'a8-b7',
    ('h2-h3', 'b7-c7', 'g13-g12', 'm8-l8', 'i1-c7', 'a8-b7', 'f14-l8'): 'n7-m8',
}

def get_tablebase_move(moves: list[str]) -> str | None:
    """
    Looks up the current board state (represented by a list of moves)
    in the tablebase. Returns a single move string or None if not found.
    Handles random selection if the tablebase entry is a list.
    """
    key = tuple(moves)
    if key in MOVES_TO_BEST_MOVE:
        result = MOVES_TO_BEST_MOVE[key]
        if isinstance(result, list):
            # Special case for random move selection
            return random.choice(result)
        return result
    return None


# FEN based tablebase for server.py.
FEN_TO_BEST_MOVE = {
  'R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'h2-h3',
  'B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'b7-c7',
  'Y-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'g13-g12',
  'G-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'm8-l8',
  'G-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,1,yP,yP,x,x,x/x,x,x,5,yP,2,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'm8-l8',
  'B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-1-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,8,rQ,1,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,1,rK,rB,rN,rR,x,x,x': 'b9-c9',
  'B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,2,rP,1,rP,3,x,x,x/x,x,x,rP,rP,1,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'b4-d4',
  'B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,2,rP,5,x,x,x/x,x,x,rP,rP,1,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'b7-c7',
  'R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x': 'f2-f3',
  'B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,rB,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,1,rN,rR,x,x,x': 'a8-b7',
  'G-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,1,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/1,bP,9,yB,1,gK/bK,bQ,rB,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,1,rN,rR,x,x,x': 'n7-m8',
}