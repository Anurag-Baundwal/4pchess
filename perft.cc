#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "board.h"
#include "utils.h"

namespace chess {

// Perft function to count leaf nodes
uint64_t perft(Board& board, int depth) {
  if (depth == 0) return 1;

  // Calculate ONCE at the start of the node
  board.RefreshKingSafety(); 
  
  // Cache safety data for the filter
  const Player player = board.GetTurn();
  const Bitboard checkers = board.Checkers();
  const Bitboard pinned = board.PinnedPieces(player.GetColor());
  const bool in_check = !checkers.is_zero();

  // --- OPTIMIZATION: Bulk Counting at Depth 1 ---
  if (depth == 1) {
      uint64_t nodes = 0;
      
      // Use ExtMove stack buffer
      ExtMove move_buffer[300];
      // Get pointer to end of generated moves
      ExtMove* end_ptr = board.GetPseudoLegalMoves2(move_buffer);

      for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
          const auto& move = *m_ptr;
          int from_sq = BitboardImpl::LocationToIndex(move.From());

          // Lazy filter: Only check IsLegal if strictly necessary
          bool needs_legality_check = in_check 
              || board.GetPiece(from_sq).GetPieceType() == KING
              || move.GetEnpassantLocation().Present()
              || (BitboardImpl::IndexToBitboard(from_sq) & pinned).operator bool();

          if (needs_legality_check && !board.IsLegal(move)) {
              continue;
          }
          nodes++;
      }
      return nodes;
  }

  uint64_t nodes = 0;
  ExtMove move_buffer[300];
  ExtMove* end_ptr = board.GetPseudoLegalMoves2(move_buffer);

  for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
    const auto& move = *m_ptr;
    int from_sq = BitboardImpl::LocationToIndex(move.From());

    bool needs_legality_check = in_check 
        || board.GetPiece(from_sq).GetPieceType() == KING
        || move.GetEnpassantLocation().Present()
        || (BitboardImpl::IndexToBitboard(from_sq) & pinned).operator bool();

    if (needs_legality_check && !board.IsLegal(move)) {
        continue;
    }

    board.MakeMove(move);
    nodes += perft(board, depth - 1);
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

  board.RefreshKingSafety();
  
  const Player player = board.GetTurn();
  const Bitboard checkers = board.Checkers();
  const Bitboard pinned = board.PinnedPieces(player.GetColor());
  const bool in_check = !checkers.is_zero();

  ExtMove move_buffer[300];
  ExtMove* end_ptr = board.GetPseudoLegalMoves2(move_buffer);

  for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
    const auto& move = *m_ptr;
    int from_sq = BitboardImpl::LocationToIndex(move.From());

    bool needs_legality_check = in_check 
        || board.GetPiece(from_sq).GetPieceType() == KING
        || move.GetEnpassantLocation().Present()
        || (BitboardImpl::IndexToBitboard(from_sq) & pinned).operator bool();

    if (needs_legality_check && !board.IsLegal(move)) {
        continue;
    }

    board.MakeMove(move);
    
    // Recursive call will hit the bulk counting optimization if depth-1 == 1
    uint64_t nodes = perft(board, depth - 1);
    total_nodes += nodes;
    std::cout << move.PrettyStr() << ": " << nodes << std::endl;

    board.UndoMove();
  }
  std::cout << "\nTotal Nodes: " << total_nodes << std::endl;
  return total_nodes;
}

}  // namespace chess

void print_usage() {
    std::cerr << "Usage: ./perft.exe [--depth <d>] [--fen <fen_string>] [--divide]\n"
              << "  --depth <d>         : The depth for the perft test (default: 4).\n"
              << "  --fen <fen_string>  : The FEN for the starting position.\n"
              << "  --divide            : Show perft results for each root move.\n";
}

int main(int argc, char* argv[]) {
  int depth = 4;
  std::string fen = "";
  bool use_divide = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth") {
        if (i + 1 < argc) {
            try {
                depth = std::stoi(argv[++i]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid number for --depth." << std::endl;
                print_usage();
                return 1;
            }
        } else {
            std::cerr << "Error: --depth option requires one argument." << std::endl;
            print_usage();
            return 1;
        }
    } else if (arg == "--fen") {
        if (i + 1 < argc) {
            fen = argv[++i];
        } else {
            std::cerr << "Error: --fen option requires one argument." << std::endl;
            print_usage();
            return 1;
        }
    } else if (arg == "--divide") {
        use_divide = true;
    } else {
        std::cerr << "Error: Unknown option " << arg << std::endl;
        print_usage();
        return 1;
    }
  }

  std::shared_ptr<chess::Board> board;
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