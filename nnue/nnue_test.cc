#include <chrono>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include "nnue.h"
#include "../board.h" 
#include "../utils.h"

namespace chess {
namespace { // Anonymous namespace for test-specific helpers

// Helper to print a few elements of a float array/vector
void PrintFloatArraySample(const std::string& name, const float* arr, int total_size, int sample_size = 8) {
    std::cout << name << " (first " << sample_size << " / last " << sample_size << " of " << total_size << "):" << std::endl;
    if (total_size == 0) {
        std::cout << "  (empty)" << std::endl;
        return;
    }
    for (int i = 0; i < std::min(sample_size, total_size); ++i) {
        std::cout << std::fixed << std::setprecision(6) << arr[i] << " ";
    }
    if (total_size > sample_size) {
        if (total_size > 2 * sample_size) std::cout << "... ";
        for (int i = std::max(sample_size, total_size - sample_size); i < total_size; ++i) {
            std::cout << std::fixed << std::setprecision(6) << arr[i] << " ";
        }
    }
    std::cout << std::endl;
}

#if defined __AVX2__ // Only compile if AVX2 is enabled in your build configuration
// Helper to print a few elements of an __m256 array
void PrintAVXArraySample(const std::string& name, const __m256* arr_m256, int num_m256_vectors, int total_float_size, int sample_m256_vectors = 1) {
    std::cout << name << " AVX (first " << sample_m256_vectors << " vec / last " << sample_m256_vectors << " vec of " << num_m256_vectors << " vecs; total floats " << total_float_size << "):" << std::endl;
    if (num_m256_vectors == 0) {
        std::cout << "  (empty)" << std::endl;
        return;
    }
    
    std::vector<float> temp_floats(8); // Temporary buffer to store AVX vector contents

    for (int i = 0; i < std::min(sample_m256_vectors, num_m256_vectors); ++i) {
        _mm256_storeu_ps(temp_floats.data(), arr_m256[i]);
        for(int j=0; j<8; ++j) std::cout << std::fixed << std::setprecision(6) << temp_floats[j] << " ";
        if (i < std::min(sample_m256_vectors, num_m256_vectors) - 1) std::cout << "| ";
    }
    
    if (num_m256_vectors > sample_m256_vectors) {
        if (num_m256_vectors > 2 * sample_m256_vectors) std::cout << "... ";
        for (int i = std::max(sample_m256_vectors, num_m256_vectors - sample_m256_vectors); i < num_m256_vectors; ++i) {
             _mm256_storeu_ps(temp_floats.data(), arr_m256[i]);
            for(int j=0; j<8; ++j) std::cout << std::fixed << std::setprecision(6) << temp_floats[j] << " ";
            if (i < num_m256_vectors -1) std::cout << "| ";
        }
    }
    std::cout << std::endl;
}
#endif // __AVX2__

// TEST(NNUE, SpeedTest) {
//   // IMPORTANT: Update this path to your model directory (e.g., C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11)
//   // For production models, you might copy the latest 'gen_XX' contents into a dedicated 'models' directory.
//   // Example if using a general 'models' folder with the latest trained weights:
//   // NNUE nnue("C:/Users/dell3/source/repos5/4pchess/models"); 
//   NNUE nnue("C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11"); // Updated path
//   auto start = std::chrono::system_clock::now();

//   constexpr int kNumEvals = 1000000;
//   for (int i = 0; i < kNumEvals; i++) {
//     // This test primarily measures the speed of NNUE::Evaluate in isolation,
//     // without board changes. For a true performance test, use a board
//     // that changes, requiring SetPiece/RemovePiece calls.
//     nnue.Evaluate(RED); 
//   }

//   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
//       std::chrono::system_clock::now() - start);
//   std::cout << "Duration (ms): " << duration.count() << std::endl;
//   std::cout << "Evals: " << kNumEvals << std::endl;
//   float evals_per_sec = 1000.0f * kNumEvals / duration.count();
//   std::cout << "Evals/sec: " << evals_per_sec << std::endl;
// }

TEST(NNUE, CorrectnessTest) {
  // IMPORTANT: Update this path to your model directory (e.g., C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11)
  NNUE nnue("C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11"); // Updated path

  auto board = Board::CreateStandardSetup();
  if (!board) {
      FAIL() << "Failed to create standard board setup.";
  }
  
  // Set NNUE pointer on board for evaluation
  // (though in Player, ThreadState will have its own NNUE copy)
  board->SetNNUE(&nnue); 

  std::vector<PlacedPiece> pp;
  // Get all pieces from the board (initial state)
  for (const auto& placed_pieces_for_color : board->GetPieceList()) {
    pp.insert(pp.end(), placed_pieces_for_color.begin(), placed_pieces_for_color.end());
  }
  // Initialize NNUE's internal state (linear_output_0_) with initial pieces
  nnue.InitializeWeights(pp);

  // Make a move to change the board state and test incremental updates
  // Assuming this move is valid for the standard setup
  // Move test_move(BoardLocation(12, 7), BoardLocation(11, 7)); // Example: Red Pawn moves h2->h3
  // board->MakeMove(test_move);

  // // Evaluate the new board state for Blue's turn (since Red moved)
  // int pred_score = nnue.Evaluate(BLUE);

  // std::cout << "CorrectnessTest: Predicted score for Blue after Red pawn move: " << pred_score << std::endl;

  Move test_move(BoardLocation(12, 5), BoardLocation(11, 5)); // Example: Red Pawn moves h2->h3
  board->MakeMove(test_move);
  Move test_move2(BoardLocation(9, 1), BoardLocation(9, 3)); // Example: Red Pawn moves h2->h3
  board->MakeMove(test_move2);
  int pred_score = nnue.Evaluate(YELLOW);

  std::cout << "CorrectnessTest: Predicted score for Yellow after f2-f3 and b5-d5: " << pred_score << std::endl;
  // Add an assertion based on an expected value if you have one
  // e.g., EXPECT_NEAR(pred_score, expected_value, tolerance);
}

TEST(NNUE, CopyWeightsTest) {
  // IMPORTANT: Update this path to your model directory (e.g., C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11)
  std::string weights_dir = "C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11"; // Updated path
  auto copy_from = std::make_shared<NNUE>(weights_dir);
  NNUE nnue(weights_dir, copy_from); // NNUE created with weights copied from 'copy_from'

  auto board = Board::CreateStandardSetup();
  if (!board) {
      FAIL() << "Failed to create standard board setup.";
  }

  board->SetNNUE(&nnue); 

  std::vector<PlacedPiece> pp;
  for (const auto& placed_pieces : board->GetPieceList()) {
    pp.insert(pp.end(), placed_pieces.begin(), placed_pieces.end());
  }
  nnue.InitializeWeights(pp);

  board->MakeMove(Move(BoardLocation(12, 7), BoardLocation(11, 7)));

  int pred_score = nnue.Evaluate(BLUE);

  std::cout << "CopyWeightsTest: Predicted score: " << pred_score << std::endl;
}

TEST(NNUE, StartPosRawEval) {
  // IMPORTANT: Replace this path with the correct absolute path to your gen_11 model directory
  // THIS IS THE MOST CRITICAL PATH TO GET RIGHT FOR YOUR DEBUGGING!
  std::string weights_dir = "C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11"; 

  std::cout << "\n--- NNUE StartPosRawEval Test ---" << std::endl;
  std::cout << "Loading NNUE weights from: " << weights_dir << std::endl;
  
  // Create NNUE instance. Pass nullptr for copy_weights_from as we are loading fresh.
  NNUE nnue(weights_dir, nullptr); 
  std::cout << "NNUE instance created." << std::endl;

  auto board = Board::CreateStandardSetup();
  if (!board) {
    FAIL() << "Failed to create standard board setup.";
  }
  std::cout << "Standard board created." << std::endl;

  std::vector<PlacedPiece> all_initial_pieces;
  const auto& piece_list_of_lists = board->GetPieceList();
  for (const auto& pieces_for_color : piece_list_of_lists) {
      all_initial_pieces.insert(all_initial_pieces.end(), pieces_for_color.begin(), pieces_for_color.end());
  }
  std::cout << "Collected " << all_initial_pieces.size() << " initial pieces for NNUE initialization." << std::endl;
  
  // This populates linear_output_0_ based on biases and all pieces on the board at start.
  // The debug prints for this step are within NNUE::InitializeWeights.
  nnue.InitializeWeights(all_initial_pieces);
  std::cout << "NNUE weights initialized with board state (biases + all initial pieces)." << std::endl;

  // Now, call Evaluate and have it print its internal states.
  // The debug prints for this step are within NNUE::Evaluate.
  std::cout << "\n--- Calling NNUE::Evaluate(RED) for startpos ---" << std::endl;
  int32_t eval_score = nnue.Evaluate(RED); // Evaluate for RED to move (standard start player)

  std::cout << "\n--- NNUE StartPosRawEval Test Finished ---" << std::endl;
  std::cout << "Final evaluation score for RED at startpos: " << eval_score << " cp" << std::endl;
  
  // Add an assertion if you have a known ground truth for the startpos.
  EXPECT_NE(eval_score, 0); 
  EXPECT_GT(eval_score, 0); // Given your HCE produces positive scores
  EXPECT_NEAR(eval_score, 0, 100); // Expect an eval of 0 cp with a tolerance of 100 cp
}

// See if loading the position from the fen yields the same evaluation as the incremental update from the startpos
TEST(NNUE, FENLoadAndEvalAfterMove) {
  // IMPORTANT: Update this path to your model directory
  std::string weights_dir = "C:/Users/dell3/source/repos5/4pchess/data/gen_models/gen_11"; 

  std::cout << "\n--- NNUE FENLoadAndEvalAfterMove Test ---" << std::endl;
  std::cout << "Loading NNUE weights from: " << weights_dir << std::endl;
  NNUE nnue(weights_dir, nullptr); // Load weights fresh
  std::cout << "NNUE instance created." << std::endl;

  // FEN string for the position after 1. R:h2-h3 (BLUE to move)
  // This is the FEN you provided, adjusted for line breaks:
  std::string fen_after_move = "B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x";

  // Parse the board directly from the FEN
  auto board_from_fen = ParseBoardFromFEN(fen_after_move);
  ASSERT_NE(board_from_fen, nullptr) << "Failed to parse FEN: " << fen_after_move;
  std::cout << "Board parsed from FEN." << std::endl;

  // Get all pieces from the newly parsed board
  std::vector<PlacedPiece> all_pieces_from_fen;
  const auto& piece_list_of_lists = board_from_fen->GetPieceList();
  for (const auto& pieces_for_color : piece_list_of_lists) {
      all_pieces_from_fen.insert(all_pieces_from_fen.end(), pieces_for_color.begin(), pieces_for_color.end());
  }
  std::cout << "Collected " << all_pieces_from_fen.size() << " pieces from FEN for NNUE initialization." << std::endl;
  
  // Initialize NNUE's internal state with pieces from the FEN
  nnue.InitializeWeights(all_pieces_from_fen);
  std::cout << "NNUE weights initialized with board state from FEN." << std::endl;

  // Evaluate the position. It's BLUE's turn after Red's move.
  std::cout << "\n--- Calling NNUE::Evaluate(BLUE) for FEN position ---" << std::endl;
  int32_t eval_score = nnue.Evaluate(BLUE); 

  std::cout << "\n--- NNUE FENLoadAndEvalAfterMove Test Finished ---" << std::endl;
  std::cout << "Final evaluation score for BLUE (from FEN init): " << eval_score << " cp" << std::endl;
  
  // Assertions:
  // Expected values for this position:
  // Python test: -197.90 cp
  // C++ CorrectnessTest (incremental update): -214 cp
  // We expect this test to be very close to the C++ CorrectnessTest value.
  EXPECT_NEAR(eval_score, -214, 15); // Tolerance of 15 cp (allows for some float differences)
}


// Simple naive tests for profiling float vs int operations
TEST(NNUE, NaiveFloatTest) {
  auto start = std::chrono::system_clock::now();

  constexpr size_t kNumEvals = 10000000000ULL; // Use ULL for large literals
  float r = 0;
  for (size_t i = 0; i < kNumEvals; i++) {
    r += i;
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start);
  std::cout << "Duration (ms): " << duration.count() << std::endl;
  std::cout << "Float ops: " << kNumEvals << std::endl;
  int64_t evals_per_sec = (int64_t) (1000.0 * kNumEvals / duration.count());
  std::cout << "Float ops/sec: " << evals_per_sec << std::endl;
  std::cout << "Res: " << r << std::endl;
}

TEST(NNUE, NaiveInt32Test) {
  auto start = std::chrono::system_clock::now();

  constexpr size_t kNumEvals = 10000000000ULL; // Use ULL for large literals
  int32_t r = 0;
  for (size_t i = 0; i < kNumEvals; i++) {
    r += (int32_t)i; // Cast to int32_t to prevent overflow if i gets large
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now() - start);
  std::cout << "Duration (ms): " << duration.count() << std::endl;
  std::cout << "Int32 ops: " << kNumEvals << std::endl;
  int64_t evals_per_sec = (int64_t) (1000.0 * kNumEvals / duration.count());
  std::cout << "Int32 ops/sec: " << evals_per_sec << std::endl;
  std::cout << "Res: " << r << std::endl;
}

}  // namespace
}  // namespace chess