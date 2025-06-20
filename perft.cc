#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "board.h"
#include "utils.h"

ABSL_FLAG(int, depth, 4, "The depth for the perft test.");
ABSL_FLAG(std::string, fen, "",
          "The FEN string for the starting position. Uses standard setup if "
          "empty.");
ABSL_FLAG(bool, divide, false,
          "Show perft results for each move from the root position.");

namespace chess {

// Perft function to count leaf nodes
uint64_t perft(Board& board, int depth) {
  if (depth == 0) {
    return 1;
  }

  uint64_t nodes = 0;
  Move move_buffer[300];
  Player player_to_move = board.GetTurn();

  size_t num_moves = board.GetPseudoLegalMoves2(move_buffer, 300);

  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = move_buffer[i];
    board.MakeMove(move);

    // After making a move, check if the king of the player who just moved is in check.
    // If so, the move was illegal.
    if (!board.IsKingInCheck(player_to_move)) {
      nodes += perft(board, depth - 1);
    }

    board.UndoMove();
  }

  return nodes;
}

// "Divide" function to show node counts for each root move
uint64_t divide(Board& board, int depth) {
  if (depth == 0) {
    std::cout << "Depth must be at least 1 for divide." << std::endl;
    return 0;
  }

  std::cout << "Divide for depth " << depth << ":" << std::endl;
  uint64_t total_nodes = 0;

  Move move_buffer[300];
  Player player_to_move = board.GetTurn();
  size_t num_moves = board.GetPseudoLegalMoves2(move_buffer, 300);

  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = move_buffer[i];
    board.MakeMove(move);

    if (!board.IsKingInCheck(player_to_move)) {
      uint64_t nodes = perft(board, depth - 1);
      total_nodes += nodes;
      std::cout << move.PrettyStr() << ": " << nodes << std::endl;
    }

    board.UndoMove();
  }
  std::cout << "\nTotal Nodes: " << total_nodes << std::endl;
  return total_nodes;
}

}  // namespace chess

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);

  std::shared_ptr<chess::Board> board;
  std::string fen = absl::GetFlag(FLAGS_fen);
  if (!fen.empty()) {
    board = chess::ParseBoardFromFEN(fen);
    if (board == nullptr) {
      std::cerr << "Failed to parse FEN: " << fen << std::endl;
      return 1;
    }
    std::cout << "Starting from FEN: " << fen << std::endl;
  } else {
    board = chess::Board::CreateStandardSetup();
    std::cout << "Starting from standard position." << std::endl;
  }

  int depth = absl::GetFlag(FLAGS_depth);
  bool use_divide = absl::GetFlag(FLAGS_divide);
  uint64_t total_nodes = 0;

  auto start = std::chrono::high_resolution_clock::now();

  if (use_divide) {
    total_nodes = chess::divide(*board, depth);
  } else {
    total_nodes = chess::perft(*board, depth);
    std::cout << "Perft(" << depth << ") = " << total_nodes << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  std::cout << "Time taken: " << diff.count() << " seconds" << std::endl;

  if (diff.count() > 0) {
    double nps = static_cast<double>(total_nodes) / diff.count();
    std::cout << "Nodes per second (NPS): " << static_cast<uint64_t>(nps)
              << std::endl;
  }

  return 0;
}