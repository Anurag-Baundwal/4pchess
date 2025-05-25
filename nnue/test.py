import numpy as np
import tensorflow as tf
import os

# --- Configuration ---
MODEL_H5_PATH = "C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_33/nnue.h5" #_SET_THIS_
# Or if you want to test the latest model from the 'models' directory in train.py output
# MODEL_H5_PATH = "C:/Users/dell3/source/repos5/4pchess/data/models/nnue.h5"

# --- Enums (mirroring C++ types.h potentially) ---
class PlayerColor:
    RED = 0
    BLUE = 1
    YELLOW = 2
    GREEN = 3
    INVALID_COLOR = 4

class PieceType:
    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5
    INVALID_PIECE_TYPE = 6

# --- Helper Functions ---
def convert_probs_to_scores(probs):
    """Converts sigmoid probabilities back to centipawn scores."""
    epsilon = 1e-8 # To prevent log(0)
    prob = np.maximum(epsilon, np.minimum(1.0 - epsilon, probs))
    d_factor = -10.0 / np.log(1.0 / 0.9 - 1.0) # From your training script
    centipawns = 100.0 * d_factor * np.log(prob / (1.0 - prob))
    return centipawns

@tf.keras.utils.register_keras_serializable()
def one_hot_layer_fn(board_tensor):
    """
    Custom Keras layer for one-hot encoding.
    Matches the one in your training script.
    """
    one_hot = tf.one_hot(board_tensor, depth=7, axis=-1) # 0-6 for pieces (0-5) + empty/opponent
    s = tf.shape(one_hot)
    batch_dim = s[0]
    output = tf.reshape(one_hot, [batch_dim, 4, 14*14*7])
    return output

def get_initial_board_setup():
    """
    Returns a dictionary representing the standard 4-player chess setup.
    Key: (row, col), Value: (PlayerColor, PieceType)
    Coordinates: (0,0) top-left for "Red's Home Row Area" if board is oriented that way.
                 (13,0) bottom-left for "Red's Home Row Area".
    Adjust if your coordinate system or player starting positions differ.
    This setup assumes:
    - Red: rows 12, 13
    - Blue: cols 0, 1 (board rotated view)
    - Yellow: rows 0, 1 (board rotated view)
    - Green: cols 12, 13 (board rotated view)

    A more robust way would be to parse your engine's Board::CreateStandardSetup() output
    or have a direct FEN parser if available. This is a manual representation.
    """
    board_state = {}
    piece_row_order = [PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
                       PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK]

    # RED (rows 13-back, 12-pawns traditionally at 'bottom' if Red is 'White')
    # Assuming Red starts at the "bottom" of a typical 2-player board view.
    # (row, col)
    # Red back rank (row 13)
    for col, piece_type in enumerate(piece_row_order):
        board_state[(13, 3 + col)] = (PlayerColor.RED, piece_type)
    # Red pawns (row 12)
    for col in range(8):
        board_state[(12, 3 + col)] = (PlayerColor.RED, PieceType.PAWN)

    # BLUE (cols 0-back, 1-pawns traditionally at 'left')
    # Blue back rank (col 0)
    for row, piece_type in enumerate(piece_row_order):
        board_state[(10 - row, 0)] = (PlayerColor.BLUE, piece_type) # (10,0) to (3,0)
    # Blue pawns (col 1)
    for row in range(8):
        board_state[(10 - row, 1)] = (PlayerColor.BLUE, PieceType.PAWN)

    # YELLOW (rows 0-back, 1-pawns traditionally at 'top')
    # Yellow back rank (row 0)
    for col, piece_type in enumerate(piece_row_order):
        board_state[(0, 10 - col)] = (PlayerColor.YELLOW, piece_type) # (0,10) to (0,3)
    # Yellow pawns (row 1)
    for col in range(8):
        board_state[(1, 10 - col)] = (PlayerColor.YELLOW, PieceType.PAWN)

    # GREEN (cols 13-back, 12-pawns traditionally at 'right')
    # Green back rank (col 13)
    for row, piece_type in enumerate(piece_row_order):
        board_state[(3 + row, 13)] = (PlayerColor.GREEN, piece_type) # (3,13) to (10,13)
    # Green pawns (col 12)
    for row in range(8):
        board_state[(3 + row, 12)] = (PlayerColor.GREEN, PieceType.PAWN)

    return board_state


def board_state_to_nnue_input_tensor(board_state, turn_to_eval_color):
    """
    Converts a board_state dictionary to the (1, 4, 14*14) NNUE input tensor.
    The input to the NNUE is from the perspective of the *team* whose turn it is.
    The first of the 4 planes is for the player whose turn it is.
    The second plane is for the next player.
    The third plane is for the partner.
    The fourth plane is for the previous player.

    Input board_tensor (before one-hot in Keras model) has shape (4, 14*14).
    Each of the 4 elements is a 14x14 grid of piece IDs.
    piece_id = 0 for empty or opponent (from that view's perspective)
    piece_id = 1 + PieceType for own piece (from that view's perspective)
    """
    input_tensor_planes = np.zeros((4, 14 * 14), dtype=np.int32)

    for relative_player_view_idx in range(4):
        # Determine the actual color for this plane/view
        current_view_player_color_code = (turn_to_eval_color + relative_player_view_idx) % 4

        plane_data = np.zeros((14, 14), dtype=np.int32)
        for r in range(14):
            for c in range(14):
                piece_info = board_state.get((r, c))
                if piece_info:
                    piece_color, piece_type = piece_info
                    if piece_color == current_view_player_color_code:
                        plane_data[r, c] = 1 + piece_type # 1 to 6
                    else:
                        plane_data[r, c] = 0 # Opponent's piece or empty
                else:
                    plane_data[r, c] = 0 # Empty

        input_tensor_planes[relative_player_view_idx, :] = plane_data.flatten()

    return np.expand_dims(input_tensor_planes, axis=0) # Add batch dimension -> (1, 4, 14*14)


# --- Main Test Logic ---
def main():
    if not os.path.exists(MODEL_H5_PATH):
        print(f"ERROR: Model file not found at {MODEL_H5_PATH}")
        return

    # Load the Keras model
    # Need to provide custom_objects if your model uses custom layers/functions not standard in Keras
    custom_objects = {'one_hot_layer_fn': one_hot_layer_fn}
    try:
        model = tf.keras.models.load_model(MODEL_H5_PATH, custom_objects=custom_objects, compile=False)
        # We compile=False because we only need predict. If you need to inspect optimizer state, compile=True.
        # If you re-compile, ensure the optimizer and loss match your training.
        # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError()) # Example compile
        print(f"Successfully loaded model from {MODEL_H5_PATH}")
        model.summary()
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return

    # 1. Test Start Position (RED to move)
    print("\n--- Testing Start Position (RED to move) ---")
    initial_board = get_initial_board_setup()
    start_pos_tensor = board_state_to_nnue_input_tensor(initial_board, PlayerColor.RED)

    # print("Sample of start_pos_tensor (plane 0 - RED's view, first 14 values):")
    # print(start_pos_tensor[0, 0, :14])
    # print("Sample of start_pos_tensor (plane 1 - BLUE's view, first 14 values):")
    # print(start_pos_tensor[0, 1, :14])


    pred_prob_start = model.predict(start_pos_tensor)
    pred_score_start = convert_probs_to_scores(pred_prob_start[0,0]) # [0,0] because batch_size=1, output_dim=1
    print(f"Predicted probability (start pos, RED's turn): {pred_prob_start[0,0]:.6f}")
    print(f"Predicted score (start pos, RED's turn): {pred_score_start:.2f} cp")


    # 2. Test Position after 1. R:h2-h3 (BLUE to move)
    # Your C++ uses (12,7) -> (11,7) for h2-h3.
    # Assuming Red's pawns start on row 12, and 7 is the h-file (0-indexed, 3+col).
    # So (12, 3+4) = (12,7) is Red's h-pawn. Moves to (11,7).
    print("\n--- Testing After 1. R:h2-h3 (BLUE to move) ---")
    board_after_move = get_initial_board_setup()
    # Make the move: remove pawn from (12,7), add pawn to (11,7)
    if (12, 7) in board_after_move and board_after_move[(12,7)] == (PlayerColor.RED, PieceType.PAWN):
        del board_after_move[(12, 7)]
        board_after_move[(11, 7)] = (PlayerColor.RED, PieceType.PAWN)
        print("Applied move R: (12,7) -> (11,7)")
    else:
        print("ERROR: Could not find Red pawn at (12,7) to make the move h2-h3.")
        return

    pos_after_move_tensor = board_state_to_nnue_input_tensor(board_after_move, PlayerColor.BLUE)
    # print("Sample of pos_after_move_tensor (plane 0 - BLUE's view, first 14 values):")
    # print(pos_after_move_tensor[0, 0, :14])
    # print("Sample of pos_after_move_tensor (plane 2 - RED's view, values around row 11, col 7):")
    # red_plane_after_move = pos_after_move_tensor[0, 2, :].reshape(14,14)
    # print(red_plane_after_move[10:13, 5:10]) # Print a slice to check the moved pawn


    pred_prob_after_move = model.predict(pos_after_move_tensor)
    pred_score_after_move = convert_probs_to_scores(pred_prob_after_move[0,0])
    print(f"Predicted probability (after R:h2-h3, BLUE's turn): {pred_prob_after_move[0,0]:.6f}")
    print(f"Predicted score (after R:h2-h3, BLUE's turn): {pred_score_after_move:.2f} cp")

    print("\n--- C++ NNUE Test Values for Comparison ---")
    print(f"C++ NNUE StartPosRawEval (RED to move): 51 cp")
    print(f"C++ NNUE CorrectnessTest (after R:h2-h3, BLUE to move): -214 cp")

if __name__ == "__main__":
    main()