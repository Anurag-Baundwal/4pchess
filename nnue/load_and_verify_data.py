import numpy as np
import argparse
import glob
import os

PIECE_MAP = {1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K'}
ENTRY_DTYPE = np.dtype([
    ('board_state', np.uint8, (784,)),
    ('score', np.int16),
    ('player_turn', np.uint8),
    ('game_result', np.int8)
])

def analyze_data(data_dir):
    files = glob.glob(os.path.join(data_dir, '*.bin'))
    if not files:
        print(f"No .bin files found in {data_dir}")
        return

    data = np.concatenate([np.fromfile(f, dtype=ENTRY_DTYPE) for f in files])
    print(f"Total positions: {len(data)}")
    
    if len(data) == 0: return

    # Print a single sample to verify
    sample = data[0]
    print(f"\n--- Sample 0 ---")
    print(f"Score (cp): {sample['score']}")
    print(f"Player Turn: {sample['player_turn']}")
    
    board = sample['board_state'].reshape(4, 14, 14)
    for c in range(14):
        print(" ".join([PIECE_MAP.get(p, '.') for p in board[0][c]]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing .bin files")
    args = parser.parse_args()
    analyze_data(args.data_dir)