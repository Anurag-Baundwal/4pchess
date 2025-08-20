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

    // 1. Check for winning king capture first. This is a terminal node.
    if (board.CheckWasLastMoveKingCapture() != IN_PROGRESS) {
      nodes += 1;
      board.UndoMove();
      continue; // Stop recursing down this line
    }

    // 2. If not a king capture, then check for self-check legality.
    if (board.IsKingInCheck(player_to_move)) {
      board.UndoMove();
      continue; // Illegal move, discard it
    }

    // 3. If legal and not game-ending, recurse.
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

  Move move_buffer[300];
  Player player_to_move = board.GetTurn();
  size_t num_moves = board.GetPseudoLegalMoves2(move_buffer, 300);

  for (size_t i = 0; i < num_moves; i++) {
      const auto& move = move_buffer[i];

      board.MakeMove(move);

      // 1. Check for winning king capture first.
      if (board.CheckWasLastMoveKingCapture() != IN_PROGRESS) {
          uint64_t nodes = 1; // This move is a single leaf node
          total_nodes += nodes;
          std::cout << move.PrettyStr() << ": " << nodes << std::endl;
          board.UndoMove();
          continue;
      }
      
      // 2. If not a king capture, check for self-check.
      if (board.IsKingInCheck(player_to_move)) {
          board.UndoMove();
          continue; // Illegal move
      }

      // 3. If legal, recurse with perft.
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
    std::cerr << "Usage: ./perft [--depth <d>] [--fen <fen_string>] [--divide]\n"
              << "  --depth <d>         : The depth for the perft test (default: 4).\n"
              << "  --fen <fen_string>  : The FEN for the starting position.\n"
              << "  --divide            : Show perft results for each root move.\n";
}

int main(int argc, char* argv[]) {
  int depth = 4;
  std::string fen = "";
  bool use_divide = false;

  // Manual argument parsing
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

// Compile:
// g++ -std=c++17 -O3 -march=native -Wall -I . perft_standalone.cc board.cc utils.cc -o perft.exe

// Run:
// ./perft.exe --depth 1
// ./perft.exe --depth 4 --fen "G-0,0,0,0-0,0,0,0-0,0,1,0-0,0,0,0-0-x,x,x,1,bR,yB,yK,3,yR,x,x,x/x,x,x,2,yP,5,x,x,x/x,x,x,2,bB,yP,yP,1,yP,1,x,x,x/bR,7,yP,yP,4/5,bP,8/9,gK,1,gP,gP,yB/1,bP,1,bP,6,gP,gB,2/2,bP,8,gP,2/1,bP,bN,3,rB,4,gP,2/bK,bP,3,rP,5,rN,rB,yQ/3,bP,rP,4,rP,1,gP,2/x,x,x,rP,7,x,x,x/x,x,x,3,rP,4,x,x,x/x,x,x,3,rK,1,rR,2,x,x,x" --divide