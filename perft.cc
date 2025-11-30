#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "board.h"
#include "utils.h"

namespace chess {

// Templated driver to enable compile-time optimizations for specific colors
template<PlayerColor Us>
uint64_t perft_driver(Board& board, int depth) {
    if (depth == 0) return 1;

    // STEP 1: Calculate safety on the stack.
    // This avoids writing to Board members and the expensive memcpy in MakeMove.
    // 'safety' stays in CPU registers/L1 cache for this node.
    SafetyInfo safety = board.CalculateSafetyT<Us>(); 
    
    const bool in_check = !safety.checkers.is_zero();

    // --- OPTIMIZATION: Bulk Counting at Depth 1 ---
    if (depth == 1) {
        uint64_t nodes = 0;
        ExtMove move_buffer[300];
        
        // Pass safety info to move generation (needed for king moves/castling)
        ExtMove* end_ptr = board.GenerateMovesT<Us>(move_buffer, safety);

        for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
            const auto& move = *m_ptr;
            // OPTIMIZATION: Use direct index access avoids BoardLocation roundtrip
            int from_sq = move.FromIndex();

            // Check legality using local bitboards directly
            bool is_king_move = board.GetPiece(from_sq).GetPieceType() == KING;
            bool is_ep = move.GetEnpassantLocation().Present();
            
            // Check pinning using stack variable 'safety.pinned'
            bool is_pinned = safety.pinned.test(from_sq);

            // 1. If we are in check, pinned, moving king, or en-passant: perform full legality check.
            // 2. Otherwise, the move is guaranteed legal by GenerateMovesT logic (pseudo-legal).
            if ((in_check || is_king_move || is_ep || is_pinned) && !board.IsLegal(move, safety)) {
                continue;
            }
            nodes++;
        }
        return nodes;
    }

    uint64_t nodes = 0;
    ExtMove move_buffer[300];
    
    // Pass safety info to move generation
    ExtMove* end_ptr = board.GenerateMovesT<Us>(move_buffer, safety);

    for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
        const auto& move = *m_ptr;
        // OPTIMIZATION: Use direct index access
        int from_sq = move.FromIndex();

        bool is_king_move = board.GetPiece(from_sq).GetPieceType() == KING;
        bool is_ep = move.GetEnpassantLocation().Present();
        bool is_pinned = safety.pinned.test(from_sq);

        if ((in_check || is_king_move || is_ep || is_pinned) && !board.IsLegal(move, safety)) {
            continue;
        }

        // MakeMove is now cheaper (no safety backup) and inlined
        board.MakeMove(move);
        
        // Next player logic
        constexpr PlayerColor NextUs = static_cast<PlayerColor>((Us + 1) % 4);
        nodes += perft_driver<NextUs>(board, depth - 1);
        
        // UndoMove is now cheaper and inlined
        board.UndoMove(move); 
    }

    return nodes;
}

// Dispatcher
uint64_t perft(Board& board, int depth) {
    switch (board.GetTurn().GetColor()) {
        case RED:    return perft_driver<RED>(board, depth);
        case BLUE:   return perft_driver<BLUE>(board, depth);
        case YELLOW: return perft_driver<YELLOW>(board, depth);
        case GREEN:  return perft_driver<GREEN>(board, depth);
        default: return 0;
    }
}

// "Divide" function
uint64_t divide(Board& board, int depth) {
  if (depth == 0) return 0;

  std::cout << "Divide for depth " << depth << ":" << std::endl;
  uint64_t total_nodes = 0;

  auto run_root = [&](auto color_constant) -> uint64_t {
      constexpr PlayerColor Color = decltype(color_constant)::value;
      
      // Calculate safety locally
      SafetyInfo safety = board.CalculateSafetyT<Color>();
      const bool in_check = !safety.checkers.is_zero();

      ExtMove move_buffer[300];
      ExtMove* end_ptr = board.GenerateMovesT<Color>(move_buffer, safety);
      uint64_t sum = 0;

      for (ExtMove* m_ptr = move_buffer; m_ptr < end_ptr; ++m_ptr) {
        const auto& move = *m_ptr;
        int from_sq = move.FromIndex();

        bool is_king_move = board.GetPiece(from_sq).GetPieceType() == KING;
        bool is_ep = move.GetEnpassantLocation().Present();
        bool is_pinned = safety.pinned.test(from_sq);

        if ((in_check || is_king_move || is_ep || is_pinned) && !board.IsLegal(move, safety)) {
            continue;
        }

        board.MakeMove(move);
        
        // Next player
        constexpr PlayerColor NextColor = static_cast<PlayerColor>((Color + 1) % 4);
        uint64_t nodes = perft_driver<NextColor>(board, depth - 1);
        
        sum += nodes;
        std::cout << move.PrettyStr() << ": " << nodes << std::endl;

        board.UndoMove(move);
      }
      return sum;
  };

  switch (board.GetTurn().GetColor()) {
      case RED:    total_nodes = run_root(std::integral_constant<PlayerColor, RED>{}); break;
      case BLUE:   total_nodes = run_root(std::integral_constant<PlayerColor, BLUE>{}); break;
      case YELLOW: total_nodes = run_root(std::integral_constant<PlayerColor, YELLOW>{}); break;
      case GREEN:  total_nodes = run_root(std::integral_constant<PlayerColor, GREEN>{}); break;
      default: break;
  }

  std::cout << "\nTotal Nodes: " << total_nodes << std::endl;
  return total_nodes;
}

}  // namespace chess

void print_usage() {
    std::cerr << "Usage: ./perft.exe [--depth <d>] [--fen <fen_string>] [--divide]\n";
}

int main(int argc, char* argv[]) {
  int depth = 4;
  std::string fen = "";
  bool use_divide = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth") {
        if (i + 1 < argc) depth = std::stoi(argv[++i]);
    } else if (arg == "--fen") {
        if (i + 1 < argc) fen = argv[++i];
    } else if (arg == "--divide") {
        use_divide = true;
    }
  }

  std::shared_ptr<chess::Board> board;
  if (!fen.empty()) {
    board = chess::ParseBoardFromFEN(fen);
    if (!board) return 1;
  } else {
    board = chess::Board::CreateStandardSetup();
  }

  auto start = std::chrono::high_resolution_clock::now();
  uint64_t total_nodes = use_divide ? chess::divide(*board, depth) : chess::perft(*board, depth);
  auto end = std::chrono::high_resolution_clock::now();
  
  std::chrono::duration<double> diff = end - start;
  std::cout << "Perft(" << depth << ") = " << total_nodes << std::endl;
  std::cout << "Time taken: " << diff.count() << " seconds" << std::endl;
  if (diff.count() > 0) {
    std::cout << "Nodes per second (NPS): " << static_cast<uint64_t>(total_nodes / diff.count()) << std::endl;
  }

  return 0;
}