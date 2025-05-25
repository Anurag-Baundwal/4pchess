// nnue/gen_data.cc
// this version uses std::endl instead of '\n' for line endings so it's probably slower
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>
#include <iomanip>
#include <vector> // Added for std::vector
#include <random>   // For std::mt19937
#include <algorithm> // For std::shuffle (optional, if more advanced random FEN selection is needed)


#include "../board.h"
#include "../player.h"
#include "../utils.h"
#include "nnue.h"

// --- MODIFICATION: Global/Static variables for FEN loading ---
namespace chess { // Encapsulate in chess namespace to avoid global pollution

std::vector<std::string> g_loaded_fens;
bool g_fens_loaded_successfully = false;
std::once_flag g_fens_load_flag;
// Assume FENs_4PC_balanced.txt is in a 'fens' subdirectory of your project root.
// Adjust this path if your file is located elsewhere.
const std::string kDefaultFenFilePath = "fens/FENs_4PC_balanced.txt";

// Function to load FENs from the specified file.
void TryLoadFENsFromFile() {
    std::ifstream fen_file(kDefaultFenFilePath);
    if (!fen_file.is_open()) {
        std::cerr << "Warning: Could not open FEN file: " << kDefaultFenFilePath
                  << ". Will use default board generation." << std::endl;
        g_fens_loaded_successfully = false;
        return;
    }
    std::string line;
    while (std::getline(fen_file, line)) {
        // Basic check: non-empty and contains a slash (common in FENs)
        if (!line.empty() && line.find('/') != std::string::npos) {
            g_loaded_fens.push_back(line);
        }
    }
    fen_file.close();
    if (g_loaded_fens.empty()) {
        std::cerr << "Warning: FEN file " << kDefaultFenFilePath
                  << " was empty or contained no FEN-like strings. Will use default board generation." << std::endl;
        g_fens_loaded_successfully = false;
    } else {
        std::cout << "Successfully loaded " << g_loaded_fens.size() << " FENs from " << kDefaultFenFilePath << std::endl;
        g_fens_loaded_successfully = true;
    }
}

} // namespace chess
// --- END MODIFICATION ---


namespace { // Original anonymous namespace for constants

constexpr int kNumThreads = 12; // Default, can be overridden by argv
constexpr int kNumSamples = 1'000'000; // Default, can be overridden by argv
constexpr int kDepth = 4; // Default, can be overridden by argv
constexpr float kRandomMoveRate = 0.05f; // MODIFIED from 0.30
constexpr float kStartPosRate = 0.15f; // Chance of starting the game from the standard starting position
constexpr float kRandomFENRate = 0.2f;// 0.2; // This is for the purely random FEN generation, not the file loading.
//constexpr float kRandomNoNNUERate = 0.5; // This was unused, can be removed or used if needed
constexpr int kMaxMovesPerGame = 300;

}  // namespace


namespace chess {


namespace { // Original anonymous namespace for helper functions

void SaveBoard(
    std::fstream& fs_board,
    std::fstream& fs_score,
    std::fstream& fs_per_depth_score,
    const Board& board,
    const Player& turn, // turn is the player whose perspective 'score' is from
    int score, // Score from the perspective of 'turn'
    const std::vector<int>& per_depth_score) { // per_depth_score is RY-relative
  for (int turn_addition = 0; turn_addition < 4; turn_addition++) {
    PlayerColor color_loop = static_cast<PlayerColor>((turn.GetColor() + turn_addition) % 4);
    for (int row = 0; row < 14; row++) {
      for (int col = 0; col < 14; col++) {
        Piece piece = board.GetPiece(row, col);
        int id = 0; // for missing
        if (piece.Present() && piece.GetColor() == color_loop) {
          id = 1 + (int)piece.GetPieceType();
        }
        fs_board << id;
        if (row == 13 && col == 13 && turn_addition == 3) {
          fs_board << std::endl;
        } else {
          fs_board << ",";
        }
      }
    }
  }
  fs_score << score << std::endl;
  for (size_t i = 0; i < per_depth_score.size(); i++) {
    fs_per_depth_score << per_depth_score[i];
    if (i < per_depth_score.size() - 1) {
      fs_per_depth_score << ",";
    } else {
      fs_per_depth_score << std::endl;
    }
  }
}

std::unique_ptr<std::fstream> OpenFile(const std::string& filepath) {
  auto fs = std::make_unique<std::fstream>();
  fs->open(filepath, std::fstream::out | std::fstream::app);
  if (fs->fail()) {
    std::cout << "Can't open file: " << filepath << std::endl;
    abort();
  }
  return fs;
}

// Random float in [0, 1]
float RandFloat() {
  return (float)rand() / RAND_MAX;
}

int RandInt(int max_val) { 
  if (max_val <= 0) return 0;
  return rand() % max_val; // Generates in [0, max_val - 1]
}

void SaveGameResults(
    GameResult game_result,
    PlayerColor last_player_to_move_color, 
    int num_total_moves_in_game,
    std::fstream& fs_result) {
  PlayerColor first_player_color = static_cast<PlayerColor>((last_player_to_move_color - (num_total_moves_in_game - 1) % 4 + 4) % 4);
  
  PlayerColor current_turn_color = first_player_color;
  for (int i = 0; i < num_total_moves_in_game; i++) {
    Team current_team = GetTeam(current_turn_color);
    switch (game_result) {
    case WIN_RY:
      fs_result << (current_team == RED_YELLOW ? 1 : -1) << std::endl;
      break;
    case WIN_BG:
      fs_result << (current_team == BLUE_GREEN ? 1 : -1) << std::endl; 
      break;
    case STALEMATE:
    case IN_PROGRESS: 
      fs_result << 0 << std::endl;
      break;
    }
    current_turn_color = static_cast<PlayerColor>((current_turn_color + 1) % 4);
  }
}


bool IsLegalLocation(int row, int col) {
  if (row < 0
      || row > 13
      || col < 0
      || col > 13
      || (row < 3 && (col < 3 || col > 10))
      || (row > 10 && (col < 3 || col > 10))) {
    return false;
  }
  return true;
}

BoardLocation GetCandidateLocation(PieceType piece_type, PlayerColor color) {
  if (piece_type != PAWN) {
    while (true) {
      int row = RandInt(14);
      int col = RandInt(14);
      if (IsLegalLocation(row, col)) {
        return BoardLocation(row, col);
      }
    }
  }

  int min_row;
  int max_row;
  int min_col;
  int max_col;
  switch (color) {
  case RED:
    min_row = 4; max_row = 12; min_col = 3; max_col = 10;
    break;
  case BLUE:
    min_row = 3; max_row = 10; min_col = 1; max_col = 9;
    break;
  case YELLOW:
    min_row = 1; max_row = 9; min_col = 3; max_col = 10;
    break;
  case GREEN:
    min_row = 3; max_row = 10; min_col = 4; max_col = 12;
    break;
  default:
    std::cerr << "GetCandidateLocation: Invalid color" << std::endl; abort();
    break;
  }

  int row = min_row + RandInt(max_row - min_row + 1); 
  int col = min_col + RandInt(max_col - min_col + 1); 
  return BoardLocation(row, col);
}

void AddPiece(std::unordered_map<BoardLocation, Piece>& location_to_piece,
              PieceType piece_type,
              PlayerColor color) {
  BoardLocation location;
  int attempts = 0;
  while (true) {
    location = GetCandidateLocation(piece_type, color);
    if (location_to_piece.find(location) == location_to_piece.end()) {
      break;
    }
    attempts++;
    if (attempts > 100) { 
        std::cerr << "Warning: Could not place piece after 100 attempts in AddPiece." << std::endl;
        for(int r=0; r<14; ++r) for(int c=0; c<14; ++c) {
            if (IsLegalLocation(r,c)) {
                BoardLocation fall_loc(r,c);
                if (location_to_piece.find(fall_loc) == location_to_piece.end()) {
                    location = fall_loc;
                    goto found_fallback;
                }
            }
        }
        found_fallback:; 
        if (location_to_piece.find(location) != location_to_piece.end()) {
             std::cerr << "Error: Still could not place piece in AddPiece. Aborting." << std::endl;
             abort(); 
        }
        break;
    }
  }
  location_to_piece[location] = Piece(color, piece_type);
}

std::shared_ptr<Board> CreateBoardFromRandomFEN() {
  std::unordered_map<BoardLocation, Piece> location_to_piece;
  for (int color_idx = 0; color_idx < 4; color_idx++) {
    PlayerColor col = static_cast<PlayerColor>(color_idx);
    AddPiece(location_to_piece, KING, col);
    AddPiece(location_to_piece, QUEEN, col);
    AddPiece(location_to_piece, ROOK, col);
    AddPiece(location_to_piece, ROOK, col);
    AddPiece(location_to_piece, BISHOP, col);
    AddPiece(location_to_piece, BISHOP, col);
    AddPiece(location_to_piece, KNIGHT, col);
    AddPiece(location_to_piece, KNIGHT, col);
    for (int i = 0; i < 8; i++) {
      AddPiece(location_to_piece, PAWN, col);
    }
  }

  PlayerColor turn_color = static_cast<PlayerColor>(RandInt(4));
  Player turn(turn_color);
  return std::make_shared<Board>(turn, std::move(location_to_piece));
}

}  // namespace


class GenData {
 public:

  GenData(int depth, int num_threads, int num_samples,
          std::string nnue_weights_filepath, float nnue_search_rate)
    : depth_(depth), num_threads_(num_threads), num_samples_(num_samples),
      nnue_weights_filepath_(std::move(nnue_weights_filepath)),
      nnue_search_rate_(nnue_search_rate) {
    
    // --- MODIFICATION: Load FENs once ---
    std::call_once(g_fens_load_flag, TryLoadFENsFromFile);
    // --- END MODIFICATION ---

    if (nnue_search_rate_ > 0.0f && !nnue_weights_filepath_.empty() && std::filesystem::exists(nnue_weights_filepath_)) {
      enable_nnue_ = true;
      copy_weights_from_ = std::make_shared<NNUE>(nnue_weights_filepath_);
    } else {
      enable_nnue_ = false; 
      copy_weights_from_ = nullptr; 
    }
  }

  void Run(const std::string& output_dir) {
    start_ = std::chrono::system_clock::now();
    std::vector<std::unique_ptr<std::thread>> threads;
    for (int i = 0; i < num_threads_; i++) {
      auto run_games = [this, i, &output_dir]() {
        CreateData(output_dir, i);
      };
      auto thread = std::make_unique<std::thread>(run_games);
      threads.push_back(std::move(thread));
    }
    for (int i = 0; i < num_threads_; i++) {
      threads[i]->join();
    }
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now() - start_);
    std::cout << "Duration (sec): " << duration.count() << std::endl;
  }

void CreateData(std::string output_dir, int thread_id) {
    std::filesystem::path thread_output_path(output_dir);
    try {
        std::filesystem::create_directories(thread_output_path);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory " << thread_output_path << ": " << e.what() << std::endl;
        abort(); 
    }

    std::string tid = std::to_string(thread_id);
    auto fs_board =           OpenFile((thread_output_path / ("board_" + tid + ".csv")).string());
    auto fs_score =           OpenFile((thread_output_path / ("score_" + tid + ".csv")).string());
    auto fs_per_depth_score = OpenFile((thread_output_path / ("per_depth_score_" + tid + ".csv")).string());
    auto fs_result =          OpenFile((thread_output_path / ("game_result_" + tid + ".csv")).string());

    PlayerOptions options_with_nnue;
    options_with_nnue.enable_nnue = enable_nnue_; 
    options_with_nnue.nnue_weights_filepath = nnue_weights_filepath_;
    options_with_nnue.num_threads = 1;
    
    PlayerOptions options_without_nnue;
    options_without_nnue.enable_nnue = false; 
    options_without_nnue.num_threads = 1;

    std::shared_ptr<Board> board;
    Move buffer[300];

    while (positions_calculated_ < num_samples_) {
      AlphaBetaPlayer player_with_nnue(options_with_nnue, copy_weights_from_);
      AlphaBetaPlayer player_without_nnue(options_without_nnue, nullptr); 

      // --- MODIFICATION: Board Initialization ---
      if (RandFloat() < kStartPosRate) { // e.g., 35% of games start from standard setup
          board = Board::CreateStandardSetup();
      } else if (g_fens_loaded_successfully && !g_loaded_fens.empty()) {
          std::string random_fen = g_loaded_fens[RandInt(g_loaded_fens.size())];
          board = ParseBoardFromFEN(random_fen);
          if (board == nullptr) {
              std::cerr << "Error parsing FEN from file: '" << random_fen 
                        << "' (Thread " << thread_id << "). Falling back to standard setup." << std::endl;
              board = Board::CreateStandardSetup();
          }
      } else {
          // Fallback if FEN file wasn't loaded or is empty
          if (RandFloat() < kRandomFENRate) { // kRandomFENRate is 0.0 by default
              board = CreateBoardFromRandomFEN();
          } else {
              board = Board::CreateStandardSetup();
          }
      }
      if (!board) {
           std::cerr << "Critical Error: Board pointer is null after initialization. (Thread " << thread_id << ")" << std::endl;
           continue; 
      }
      // --- END MODIFICATION ---


      int num_game_moves = 0;
      while (true) {
        if (num_game_moves >= kMaxMovesPerGame) {
          IncrGameResults(IN_PROGRESS);
          SaveGameResults(IN_PROGRESS, board->GetTurn().GetColor(), num_game_moves, *fs_result);
          break;
        }
        GameResult game_result_check = board->GetGameResult();
        if (game_result_check != IN_PROGRESS) {
          IncrGameResults(game_result_check);
          SaveGameResults(game_result_check, board->GetTurn().GetColor(), num_game_moves, *fs_result);
          break;
        }
        
        std::vector<int> per_depth_score_ry_relative; 
        per_depth_score_ry_relative.reserve(depth_ + 1);

        AlphaBetaPlayer* p_selected =
          (enable_nnue_ && RandFloat() <= nnue_search_rate_) ? &player_with_nnue : &player_without_nnue;

        Player current_player_obj = board->GetTurn();
        Team current_player_team = current_player_obj.GetTeam();

        int static_eval_current_player = p_selected->StaticEvaluation(*board);
        int static_eval_ry = (current_player_team == BLUE_GREEN) ? -static_eval_current_player : static_eval_current_player;
        per_depth_score_ry_relative.push_back(static_eval_ry);
        
        std::optional<Move> best_move_for_game;
        int final_search_score_ry_relative = static_eval_ry; 

        if (depth_ > 0) {
            for (int depth = 1; depth <= depth_; depth++) {
              auto res_tuple = p_selected->MakeMove(*board, std::nullopt, depth);
              if (!res_tuple.has_value()) {
                  std::cerr << "Error: MakeMove returned nullopt in gen_data for depth " << depth 
                            << " (Thread " << thread_id << ")" << std::endl;
                  if (!per_depth_score_ry_relative.empty()) {
                      per_depth_score_ry_relative.push_back(per_depth_score_ry_relative.back());
                  } else {
                      per_depth_score_ry_relative.push_back(0); 
                  }
                  continue; 
              }
              auto res_val = res_tuple.value();
              int current_depth_score_ry = std::get<0>(res_val); 
              
              per_depth_score_ry_relative.push_back(current_depth_score_ry);
              best_move_for_game = std::get<1>(res_val);
              final_search_score_ry_relative = current_depth_score_ry; 
            }
        }

        int score_for_file_current_player_perspective = (current_player_team == BLUE_GREEN) ? -final_search_score_ry_relative : final_search_score_ry_relative;

        SaveBoard(*fs_board, *fs_score, *fs_per_depth_score, *board,
            current_player_obj, score_for_file_current_player_perspective, per_depth_score_ry_relative);
        
        IncrementStats();
        num_game_moves++;

        std::optional<Move> move_to_play = best_move_for_game; 

        if (!move_to_play.has_value()) { 
            size_t n_pseudo_moves = board->GetPseudoLegalMoves2(buffer, 300);
            if (n_pseudo_moves > 0) {
                std::vector<Move> legal_options;
                for(size_t i=0; i < n_pseudo_moves; ++i) {
                    board->MakeMove(buffer[i]);
                    if (!board->IsKingInCheck(current_player_obj)) {
                        legal_options.push_back(buffer[i]);
                    }
                    board->UndoMove();
                }
                if (!legal_options.empty()) {
                    move_to_play = legal_options[RandInt(legal_options.size())];
                } else { 
                    GameResult gr_after_no_legal = board->GetGameResult();
                    if (gr_after_no_legal != IN_PROGRESS) {
                         IncrGameResults(gr_after_no_legal);
                         SaveGameResults(gr_after_no_legal, board->GetTurn().GetColor(), num_game_moves-1, *fs_result);
                    } else {
                        std::cerr << "Error: No legal moves but game IN_PROGRESS. (Thread " << thread_id << ")" << std::endl;
                        IncrGameResults(IN_PROGRESS); 
                        SaveGameResults(IN_PROGRESS, board->GetTurn().GetColor(), num_game_moves-1, *fs_result);
                    }
                    break; 
                }
            } else { 
                 GameResult gr_after_no_pseudo = board->GetGameResult();
                 IncrGameResults(gr_after_no_pseudo); 
                 SaveGameResults(gr_after_no_pseudo, board->GetTurn().GetColor(), num_game_moves-1, *fs_result);
                 break; 
            }
        }
        
        // --- MODIFICATION: kRandomMoveRate is now 0.5 ---
        // The existing logic below handles playing the random move correctly.
        if (RandFloat() < kRandomMoveRate) {
            size_t n_pseudo_moves = board->GetPseudoLegalMoves2(buffer, 300);
            if (n_pseudo_moves > 0) {
                std::vector<Move> legal_random_options;
                for(size_t i=0; i < n_pseudo_moves; ++i) {
                    board->MakeMove(buffer[i]);
                    if (!board->IsKingInCheck(current_player_obj)) {
                        legal_random_options.push_back(buffer[i]);
                    }
                    board->UndoMove();
                }
                 if (!legal_random_options.empty()) {
                    board->MakeMove(legal_random_options[RandInt(legal_random_options.size())]);
                } else if (n_pseudo_moves > 0) { 
                    board->MakeMove(buffer[RandInt(n_pseudo_moves)]);
                } else { break; } 
            } else { break; } 
        } else {
            board->MakeMove(move_to_play.value());
        }
      } 
    } 
  } 

  void IncrementStats() {
    std::lock_guard lock(mutex_);
    positions_calculated_++;
    if (positions_calculated_ % 100 == 0 || positions_calculated_ == 1 || positions_calculated_ == num_samples_) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::system_clock::now() - start_);
      float pos_per_sec = (duration.count() > 0) ? ((float)positions_calculated_ / duration.count()) : 0.0f;
      size_t total_games = games_ry_won + games_bg_won + games_stalemate + games_unfinished;
      float ry_win_pct = (total_games > 0) ? (100.0 * (float)games_ry_won / (float)total_games) : 0.0f;
      float bg_win_pct = (total_games > 0) ? (100.0 * (float)games_bg_won / (float)total_games) : 0.0f;
      float stalemate_pct = (total_games > 0) ? (100.0 * (float)games_stalemate / (float)total_games) : 0.0f;
      float unfinished_pct = (total_games > 0) ? (100.0 * (float)games_unfinished / (float)total_games) : 0.0f;
      std::cout
        << "Positions calculated: " << positions_calculated_
        << " Positions/sec: " << pos_per_sec
        << std::setprecision(4)
        << " #games: " << total_games
        << " RY-win: " << ry_win_pct << "%"
        << " BG-win: " << bg_win_pct << "%"
        << " Draw: " << stalemate_pct << "%"
        << " Unfin: " << unfinished_pct << "%"
        << std::endl;
    }
  }

  void IncrGameResults(GameResult game_result) {
    std::lock_guard lock(mutex_);
    switch (game_result) {
    case WIN_RY:
      games_ry_won++;
      break;
    case WIN_BG:
      games_bg_won++;
      break;
    case STALEMATE:
      games_stalemate++;
      break;
    case IN_PROGRESS: 
      games_unfinished++;
      break;
    }
  }

 private:
  int depth_ = 0;
  int num_threads_ = 0;
  size_t num_samples_ = 0;
  std::mutex mutex_;
  std::atomic<size_t> positions_calculated_ = 0;
  std::chrono::time_point<std::chrono::system_clock> start_;
  bool enable_nnue_ = false;
  std::string nnue_weights_filepath_;
  std::shared_ptr<NNUE> copy_weights_from_; 
  float nnue_search_rate_ = 0;

  size_t games_ry_won = 0;
  size_t games_bg_won = 0;
  size_t games_stalemate = 0;
  size_t games_unfinished = 0;
};

}  // namespace chess

namespace { // Original anonymous namespace for main's PrintUsage

void PrintUsage() {
  std::cout << "Usage: <prog> /absolute/output_dir [depth] [num_threads]"
    << " [num_samples] [nnue_search_rate (0.0-1.0)] [nnue_weights_filepath (optional)]"
    << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  // Initialize random seed (important for rand() to be different each run)
  std::srand(static_cast<unsigned int>(std::time(nullptr)));

  if (argc < 2) {
    std::cout << "Missing output_dir" << std::endl;
    PrintUsage();
    return 1;
  }
  std::string output_dir(argv[1]);

  int depth = kDepth;
  if (argc >= 3) {
    depth = std::atoi(argv[2]);
  }
  if (depth < 0) { 
    std::cout << "Invalid depth: " << std::string(argv[2]) << ". Must be >= 0." << std::endl;
    PrintUsage();
    return 1;
  }

  int num_threads = kNumThreads;
  if (argc >= 4) {
    num_threads = std::atoi(argv[3]);
  }
  if (num_threads <= 0) {
    std::cout << "Invalid num_threads: " << std::string(argv[3]) << std::endl;
    PrintUsage();
    return 1;
  }

  int num_samples = kNumSamples;
  if (argc >= 5) {
    num_samples = std::atoi(argv[4]);
  }
  if (num_samples <= 0) {
    std::cout << "Invalid num_samples: " << std::string(argv[4]) << std::endl;
    PrintUsage();
    return 1;
  }

  float nnue_search_rate = 0.5; 
  if (argc >= 6) {
    nnue_search_rate = std::atof(argv[5]);
    if (nnue_search_rate < 0.0f || nnue_search_rate > 1.0f) {
        std::cout << "Invalid nnue_search_rate: " << argv[5] << ". Must be between 0.0 and 1.0." << std::endl;
        PrintUsage();
        return 1;
    }
  }
  
  std::string nnue_weights_filepath;
  if (argc >= 7) { 
    nnue_weights_filepath = std::string(argv[6]);
  }


  chess::GenData gen_data(depth, num_threads, num_samples, nnue_weights_filepath,
      nnue_search_rate);
  gen_data.Run(output_dir);
  return 0;
}