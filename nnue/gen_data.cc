#include <mutex>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <cmath> // Required for std::exp (Softmax)

#include "../board.h"
#include "../player.h"
#include "../utils.h"
#include "nnue.h"

// --- Binary Data Struct ---
#pragma pack(push, 1)
struct TrainingDataEntry {
    uint8_t board_state[784]; // 4 channels (Views) x 14x14
    int16_t score;            // Score from the perspective of player_turn
    uint8_t player_turn;      // 0=R, 1=B, 2=Y, 3=G
    int8_t game_result;       // 1=Win, -1=Loss, 0=Draw, -128=In Progress
};
#pragma pack(pop)

namespace chess {

// --- Global FEN Setup ---
std::vector<std::string> g_loaded_fens;
bool g_fens_loaded_successfully = false;
std::once_flag g_fens_load_flag;
const std::string kDefaultFenFilePath = "fens/FENs_4PC_balanced.txt";

void TryLoadFENsFromFile() {
    std::ifstream fen_file(kDefaultFenFilePath);
    if (!fen_file.is_open()) {
        std::cerr << "Warning: Could not open FEN file: " << kDefaultFenFilePath << std::endl;
        g_fens_loaded_successfully = false;
        return;
    }
    std::string line;
    while (std::getline(fen_file, line)) {
        if (!line.empty() && line.find('/') != std::string::npos) {
            g_loaded_fens.push_back(line);
        }
    }
    fen_file.close();
    g_fens_loaded_successfully = !g_loaded_fens.empty();
}

} // namespace chess

namespace {
// --- Generation Constants ---
constexpr int kNumThreads = 12;
constexpr int kNumSamples = 100'000;
constexpr int kDepth = 4;
constexpr float kRandomMoveRate = 0.05f;
constexpr float kStartPosRate = 0.15f;
constexpr float kRandomFENRate = 0.2f;
constexpr int kMaxMovesPerGame = 300;

// --- Quality Filter Parameters ---
constexpr int kMaxMaterialImbalance = 1200;
constexpr int kMinPiecesForTraining = 10;
constexpr int kMaxPiecesForTraining = 64;
constexpr int kMaxScoreMagnitude = 1500;
constexpr int kAdjudicationScoreThreshold = 1750;

// --- Thread-Safe Random Number Generation ---
thread_local std::mt19937 rng(std::random_device{}());

float RandFloat() { 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng); 
}

int RandInt(int max_val) { 
    if (max_val <= 0) return 0;
    std::uniform_int_distribution<int> dist(0, max_val - 1);
    return dist(rng); 
}

bool IsPositionGoodForTraining(chess::Board& board, int search_score, const std::optional<chess::Move>& best_move) {
    if (best_move.has_value() && best_move->IsCapture()) return false;
    if (std::abs(search_score) > kMaxScoreMagnitude || std::abs(search_score) > chess::kMateValue - kMaxMovesPerGame) return false;
    for (int i = 0; i < 4; ++i) {
        if (board.IsKingInCheck(chess::Player(static_cast<chess::PlayerColor>(i)))) return false;
    }
    int total_pieces = 0;
    for (const auto& piece_list : board.GetPieceList()) total_pieces += piece_list.size();
    if (total_pieces < kMinPiecesForTraining || total_pieces > kMaxPiecesForTraining) return false;
    if (std::abs(board.PieceEvaluation()) > kMaxMaterialImbalance) return false;
    return true;
}

void SerializeBoardToArray(const chess::Board& board, chess::Player current_turn, uint8_t (&board_state_array)[784]) {
    std::memset(board_state_array, 0, sizeof(board_state_array));
    for (int relative_view_idx = 0; relative_view_idx < 4; ++relative_view_idx) {
        chess::PlayerColor view_color = static_cast<chess::PlayerColor>((current_turn.GetColor() + relative_view_idx) % 4);
        int channel_offset = relative_view_idx * 196;
        for (int row = 0; row < 14; ++row) {
            for (int col = 0; col < 14; ++col) {
                chess::Piece piece = board.GetPiece(row, col);
                if (piece.Present() && piece.GetColor() == view_color) {
                    board_state_array[channel_offset + row * 14 + col] = static_cast<uint8_t>(piece.GetPieceType()) + 1;
                }
            }
        }
    }
}

}  // namespace

namespace chess {

class GenData {
 public:
  GenData(int depth, int num_threads, int num_samples, std::string nnue_weights_filepath, float nnue_search_rate)
    : depth_(depth), num_threads_(num_threads), num_samples_(num_samples),
      nnue_weights_filepath_(std::move(nnue_weights_filepath)), nnue_search_rate_(nnue_search_rate) {
    std::call_once(g_fens_load_flag, TryLoadFENsFromFile);
    if (nnue_search_rate_ > 0.0f && !nnue_weights_filepath_.empty() && std::filesystem::exists(nnue_weights_filepath_)) {
      enable_nnue_ = true;
      copy_weights_from_ = std::make_shared<NNUE>(nnue_weights_filepath_);
    }
  }

  void Run(const std::string& output_dir) {
    start_ = std::chrono::system_clock::now();
    std::vector<std::unique_ptr<std::thread>> threads;
    for (int i = 0; i < num_threads_; i++) {
      threads.push_back(std::make_unique<std::thread>([this, i, output_dir]() { CreateData(output_dir, i); }));
    }
    for (int i = 0; i < num_threads_; i++) threads[i]->join();
  }

 private:
  void CreateData(std::string output_dir, int thread_id) {
    std::filesystem::path thread_output_path(output_dir);
    std::filesystem::create_directories(thread_output_path);

    std::string bin_filename = (thread_output_path / ("data_" + std::to_string(thread_id) + ".bin")).string();
    std::ofstream fs_data(bin_filename, std::ios::binary | std::ios::out);

    PlayerOptions options_with_nnue;
    options_with_nnue.num_threads = 1;
    options_with_nnue.enable_nnue = enable_nnue_;
    options_with_nnue.nnue_weights_filepath = nnue_weights_filepath_;

    PlayerOptions options_without_nnue;
    options_without_nnue.num_threads = 1;
    options_without_nnue.enable_nnue = false;
    options_without_nnue.nnue_weights_filepath = "";

    Move buffer[300];

    while (positions_calculated_ < num_samples_) {
      AlphaBetaPlayer player_with_nnue(options_with_nnue, copy_weights_from_);
      AlphaBetaPlayer player_without_nnue(options_without_nnue, nullptr); 
      std::shared_ptr<Board> board;

      if (RandFloat() < kStartPosRate) board = Board::CreateStandardSetup();
      else if (g_fens_loaded_successfully) board = ParseBoardFromFEN(g_loaded_fens[RandInt(g_loaded_fens.size())]);
      else board = Board::CreateStandardSetup();

      if (!board) continue;

      // 1. Determine NNUE vs HCE *once* at the start of the game
      bool is_nnue_game = enable_nnue_ && (RandFloat() <= nnue_search_rate_);
      AlphaBetaPlayer* p_selected = is_nnue_game ? &player_with_nnue : &player_without_nnue;

      int num_game_moves = 0;
      std::vector<TrainingDataEntry> game_history;
      GameResult adjudicated_result = IN_PROGRESS;

      while (true) {
        if (num_game_moves >= kMaxMovesPerGame || positions_calculated_ >= num_samples_) break;
        
        GameResult result = board->GetGameResult();
        if (result != IN_PROGRESS) break;

        Player current_turn = board->GetTurn();

        auto res_tuple = p_selected->MakeMove(*board, std::nullopt, depth_);
        if (!res_tuple.has_value()) break;
        
        int score = std::get<0>(res_tuple.value()); // Score / Eval from the perspective of the side to move (not RY relative)

        std::optional<Move> best_move = std::get<1>(res_tuple.value());
        if (!best_move.has_value()) break; // No legal moves (Mate / Stalemate)

        // Adjudication
        if (num_game_moves > 20 && std::abs(score) > kAdjudicationScoreThreshold) {
            if (score > 0) {
                adjudicated_result = (current_turn.GetTeam() == RED_YELLOW) ? WIN_RY : WIN_BG;
            } else {
                adjudicated_result = (current_turn.GetTeam() == RED_YELLOW) ? WIN_BG : WIN_RY;
            }
            break;
        }

        // Save normalized score to training data
        if (IsPositionGoodForTraining(*board, score, best_move)) {
            TrainingDataEntry entry;
            SerializeBoardToArray(*board, current_turn, entry.board_state);
            entry.score = static_cast<int16_t>(std::clamp(score, -32000, 32000));
            entry.player_turn = static_cast<uint8_t>(current_turn.GetColor());
            entry.game_result = -128; // To be backfilled
            game_history.push_back(entry);
        }

        // 2. Data Diversification: Top-5 Softmax Picker with Shallow-Then-Deep Search
        if (RandFloat() < kRandomMoveRate) {
            size_t n_pseudo = board->GetPseudoLegalMoves2(buffer, 300);
            std::vector<Move> legal_moves;
            for (size_t i = 0; i < n_pseudo; ++i) {
                board->MakeMove(buffer[i]);
                if (!board->IsKingInCheck(current_turn)) legal_moves.push_back(buffer[i]);
                board->UndoMove();
            }

            if (!legal_moves.empty()) {
                
                // Helper lambda: safely evaluates candidates, preventing crashes if a move finishes the game.
                auto evaluate_candidate = [&](const Move& m, int eval_depth) -> int {
                    board->MakeMove(m);
                    int score = -chess::kMateValue;
                    if (board->GetGameResult() != IN_PROGRESS) {
                        GameResult res = board->GetGameResult();
                        if ((current_turn.GetTeam() == RED_YELLOW && res == WIN_RY) ||
                            (current_turn.GetTeam() == BLUE_GREEN && res == WIN_BG)) {
                            score = chess::kMateValue;
                        } else if (res == STALEMATE) {
                            score = 0;
                        } else {
                            score = -chess::kMateValue;
                        }
                    } else {
                        auto move_res = p_selected->MakeMove(*board, std::nullopt, eval_depth); 
                        if (move_res.has_value()) {
                            // Since we made a move, move_res is from the OPPONENT's perspective.
                            // We negate it to get the score from current_turn's perspective.
                            score = -std::get<0>(*move_res); 
                        }
                    }
                    board->UndoMove();
                    return score;
                };

                std::vector<std::pair<Move, int>> shallow_scores;
                
                // Stage 1: Shallow Pre-sort (Fast, depth=2)
                for (const auto& m : legal_moves) {
                    shallow_scores.push_back({m, evaluate_candidate(m, 2)});
                }

                if (!shallow_scores.empty()) {
                    std::sort(shallow_scores.begin(), shallow_scores.end(), 
                        [](const auto& a, const auto& b) { return a.second > b.second; });
                    
                    // Stage 2: Deep Evaluation on the Top 7 candidates (Accurate, TT-assisted)
                    int num_candidates = std::min<int>(7, static_cast<int>(shallow_scores.size()));
                    std::vector<std::pair<Move, int>> deep_scores;

                    for (int i = 0; i < num_candidates; ++i) {
                        Move m = shallow_scores[i].first;
                        deep_scores.push_back({m, evaluate_candidate(m, std::max(1, depth_ - 1))});
                    }

                    if (!deep_scores.empty()) {
                        std::sort(deep_scores.begin(), deep_scores.end(), 
                            [](const auto& a, const auto& b) { return a.second > b.second; });

                        // Stage 3: Softmax probability distribution over the true Top 5
                        int num_top = std::min<int>(5, static_cast<int>(deep_scores.size()));
                        std::vector<float> probs(num_top);
                        float sum_probs = 0.0f;
                        
                        float temperature = 0.3f; 
                        int max_score = deep_scores[0].second; 
                        
                        for (int i = 0; i < num_top; ++i) {
                            float scaled_score = (deep_scores[i].second - max_score) / (100.0f * temperature);
                            probs[i] = std::exp(scaled_score); // Stable as it stays <= 1.0
                            sum_probs += probs[i];
                        }
                        
                        float rand_val = RandFloat() * sum_probs;
                        float cumulative = 0.0f;
                        std::optional<Move> chosen_move = std::nullopt;
                        for (int i = 0; i < num_top; ++i) {
                            cumulative += probs[i];
                            if (rand_val <= cumulative) {
                                chosen_move = deep_scores[i].first;
                                break;
                            }
                        }
                        if (!chosen_move.has_value()) chosen_move = deep_scores[0].first; // safety fallback
                        
                        board->MakeMove(*chosen_move);
                    } else {
                        board->MakeMove(*best_move);
                    }
                } else {
                    board->MakeMove(*best_move);
                }
            } else {
                break; // No legal moves 
            }
        } else {
            board->MakeMove(*best_move);
        }
        
        num_game_moves++;
      }

      GameResult final_result = board->GetGameResult();
      if (final_result == IN_PROGRESS) final_result = adjudicated_result; // Catch adjudications

      for (auto& entry : game_history) {
          if (final_result == WIN_RY) entry.game_result = (entry.player_turn == RED || entry.player_turn == YELLOW) ? 1 : -1;
          else if (final_result == WIN_BG) entry.game_result = (entry.player_turn == BLUE || entry.player_turn == GREEN) ? 1 : -1;
          else if (final_result == STALEMATE) entry.game_result = 0;
          
          fs_data.write(reinterpret_cast<const char*>(&entry), sizeof(TrainingDataEntry));
          IncrementStats();
      }
    }
    fs_data.close();
  } 

  void IncrementStats() {
    std::lock_guard lock(mutex_);
    positions_calculated_++;
    if (positions_calculated_ % 5000 == 0) {
      std::cout << "Positions calculated: " << positions_calculated_ << " / " << num_samples_ << std::endl;
    }
  }

  int depth_, num_threads_;
  size_t num_samples_;
  std::mutex mutex_;
  std::atomic<size_t> positions_calculated_ = 0;
  std::chrono::time_point<std::chrono::system_clock> start_;
  bool enable_nnue_ = false;
  std::string nnue_weights_filepath_;
  std::shared_ptr<NNUE> copy_weights_from_; 
  float nnue_search_rate_ = 0;
};

}  // namespace chess

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  std::string output_dir(argv[1]);
  int depth = (argc >= 3) ? std::atoi(argv[2]) : 4;
  int num_threads = (argc >= 4) ? std::atoi(argv[3]) : 12;
  int num_samples = (argc >= 5) ? std::atoi(argv[4]) : 100000;
  float nnue_search_rate = (argc >= 6) ? std::atof(argv[5]) : 0.5f;
  std::string nnue_weights = (argc >= 7) ? std::string(argv[6]) : "";

  chess::GenData gen_data(depth, num_threads, num_samples, nnue_weights, nnue_search_rate);
  gen_data.Run(output_dir);
  return 0;
}