#include <tuple>
#include <algorithm>
#include <cmath>
#include <utility>
#include <cassert>
#include <functional>
#include <optional>
#include <iostream>
#include <tuple>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <mutex>
#include <atomic>

#include "board.h"
#include "player.h"
#include "move_picker.h"
#include "transposition_table.h"

namespace chess {

// This namespace must be accessible here for the conversion helpers.
// The definitions are in board.cc, but we need the declarations.
namespace BitboardImpl {
    extern int LocationToIndex(const BoardLocation& loc);
    extern BoardLocation IndexToLocation(int index);
    extern void InitBitboards();
    extern Bitboard kLegalSquares;
    extern Bitboard kRayAttacks[240][8];
    extern Bitboard kKingAttacks[240];
    enum RayDirection { D_NE, D_NW, D_SE, D_SW, D_N, D_E, D_S, D_W };
}

AlphaBetaPlayer::AlphaBetaPlayer(std::optional<PlayerOptions> options) {
  if (options.has_value()) {
    options_ = *options;
  }

  // This must be called once globally to initialize bitboard attack tables.
  BitboardImpl::InitBitboards();

  piece_move_order_scores_[PAWN] = 1;
  piece_move_order_scores_[KNIGHT] = 2;
  piece_move_order_scores_[BISHOP] = 3;
  piece_move_order_scores_[ROOK] = 4;
  piece_move_order_scores_[QUEEN] = 5;
  piece_move_order_scores_[KING] = 0;

  king_attacker_values_[PAWN] = 25;
  king_attacker_values_[KNIGHT] = 30;
  king_attacker_values_[BISHOP] = 30;
  king_attacker_values_[ROOK] = 40;
  king_attacker_values_[QUEEN] = 50;
  king_attacker_values_[KING] = 0;

  if (options_.enable_transposition_table) {
    transposition_table_ = std::make_unique<TranspositionTable>(
        options_.transposition_table_size);
  }

  heuristic_mutexes_ = std::make_unique<std::mutex[]>(kHeuristicMutexes);
  counter_moves = new Move[256][256];
  continuation_history = new ContinuationHistory*[2];
  for (int i = 0; i < 2; i++) {
    continuation_history[i] = new ContinuationHistory[2];
  }
  ResetHistoryHeuristics();

  king_attack_weight_[0] = 0;
  king_attack_weight_[1] = 50;
  king_attack_weight_[2] = 100;
  king_attack_weight_[3] = 120;
  king_attack_weight_[4] = 150;
  king_attack_weight_[5] = 200;
  king_attack_weight_[6] = 250;
  king_attack_weight_[7] = 300;
  for (int i = 8; i < 30; i++) {
    king_attack_weight_[i] = 400;
  }

  if (options_.enable_piece_square_table) {
    for (int sq = 0; sq < 256; ++sq) {
        BoardLocation loc = BitboardImpl::IndexToLocation(sq);
        if (!loc.Present()) continue;

        int row = loc.GetRow();
        int col = loc.GetCol();

        for (int cl = 0; cl < 4; cl++) {
            PlayerColor color = static_cast<PlayerColor>(cl);
            for (int pt = 0; pt < 6; pt++) {
                PieceType piece_type = static_cast<PieceType>(pt);
                bool is_piece = (piece_type == QUEEN || piece_type == ROOK || piece_type == BISHOP || piece_type == KNIGHT);
                int table_value = 0;
                if (is_piece) {
                    float center_dist = std::sqrt((row - 6.5) * (row - 6.5) + (col - 6.5) * (col - 6.5));
                    table_value -= (int)(10 * center_dist);
                    if (color == RED || color == YELLOW) {
                        if (col < 3 || col > 10) table_value += 10;
                    } else {
                        if (row < 3 || row > 10) table_value += 10;
                    }
                }
                piece_square_table_[color][piece_type][sq] = table_value;
            }
        }
    }
  }

  if (options_.enable_piece_activation) {
    piece_activation_threshold_[KING] = 999;
    piece_activation_threshold_[PAWN] = 999;
    piece_activation_threshold_[NO_PIECE] = 999;
    piece_activation_threshold_[QUEEN] = 5;
    piece_activation_threshold_[BISHOP] = 5;
    piece_activation_threshold_[KNIGHT] = 3;
    piece_activation_threshold_[ROOK] = 5;
  }

  if (options_.enable_knight_bonus) {
    std::memset(knight_to_king_, 0, sizeof(knight_to_king_));
    for (int from_sq = 0; from_sq < 256; ++from_sq) {
        BoardLocation from_loc = BitboardImpl::IndexToLocation(from_sq);
        if (!from_loc.Present()) continue;

        for (int dr1 : {-2, -1, 1, 2}) {
            int dc1_abs = std::abs(dr1) == 1 ? 2 : 1;
            for (int dc1_sign : {-1, 1}) {
                int dc1 = dc1_sign * dc1_abs;
                BoardLocation loc1 = from_loc.Relative(dr1, dc1);
                if (!loc1.Present()) continue;

                for (int dr2 : {-2, -1, 1, 2}) {
                    int dc2_abs = std::abs(dr2) == 1 ? 2 : 1;
                    for (int dc2_sign : {-1, 1}) {
                        int dc2 = dc2_sign * dc2_abs;
                        BoardLocation to_loc = loc1.Relative(dr2, dc2);
                        if (!to_loc.Present()) continue;

                        int to_sq = BitboardImpl::LocationToIndex(to_loc);
                        if(to_sq >= 0) knight_to_king_[from_sq][to_sq] = true;
                    }
                }
            }
        }
    }
  }
}

AlphaBetaPlayer::~AlphaBetaPlayer() {
    delete[] counter_moves;
    for (int i = 0; i < 2; i++) {
        delete[] continuation_history[i];
    }
    delete[] continuation_history;
}

ThreadState::ThreadState(
    PlayerOptions options, const Board& board, const PVInfo& pv_info)
  : options_(options), root_board_(board), pv_info_(pv_info) {
  move_buffer_ = new Move[kBufferPartitionSize * kBufferNumPartitions];
}

ThreadState::~ThreadState() {
  delete[] move_buffer_;
}

Move* ThreadState::GetNextMoveBufferPartition() {
  if (buffer_id_ >= kBufferNumPartitions) {
    std::cout << "ThreadState move buffer overflow" << std::endl;
    abort();
  }
  return &move_buffer_[buffer_id_++ * kBufferPartitionSize];
}

void ThreadState::ReleaseMoveBufferPartition() {
  assert(buffer_id_ > 0);
  buffer_id_--;
}

int AlphaBetaPlayer::GetNumLegalMoves(Board& board) {
  constexpr int kLimit = 300;
  Move moves[kLimit];
  Player player = board.GetTurn();
  size_t num_moves = board.GetPseudoLegalMoves2(moves, kLimit);
  int n_legal = 0;
  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = moves[i];
    board.MakeMove(move);
    if (!board.IsKingInCheck(player)) {
      n_legal++;
    }
    board.UndoMove();
  }

  return n_legal;
}

std::optional<std::tuple<int, std::optional<Move>>> AlphaBetaPlayer::Search(
    Stack* ss,
    NodeType node_type,
    ThreadState& thread_state,
    Board& board,
    int ply,
    int depth,
    int alpha,
    int beta,
    bool maximizing_player,
    int expanded,
    const std::optional<
        std::chrono::time_point<std::chrono::system_clock>>& deadline,
    PVInfo& pvinfo,
    int null_moves,
    bool is_cut_node) {
  depth = std::max(depth, 0);
  if (canceled_
      || (deadline.has_value()
        && std::chrono::system_clock::now() >= *deadline)) {
    return std::nullopt;
  }
  num_nodes_++;
  bool is_root_node = ply == 1;

  bool is_pv_node = node_type != NonPV;
  bool is_tt_pv = false;

  std::optional<Move> tt_move;
  const HashTableEntry* tte = nullptr;
  bool tt_hit = false;
  if (options_.enable_transposition_table) {
    int64_t key = board.HashKey();

    tte = transposition_table_->Get(key);
    if (tte != nullptr) {
      if (tte->key == key) { // valid entry
        tt_hit = true;
        if (tte->depth >= depth) {
          num_cache_hits_++;
          if (!is_root_node
              && !is_pv_node
              && (tte->bound == EXACT
                || (tte->bound == LOWER_BOUND && tte->score >= beta)
                || (tte->bound == UPPER_BOUND && tte->score <= alpha))
             ) {

            if (tte->move.Present()) {
                return std::make_tuple(
                    std::min(beta, std::max(alpha, tte->score)), tte->move);
            }
            return std::make_tuple(
                std::min(beta, std::max(alpha, tte->score)), std::nullopt);
          }
        }
        if (tte->move.Present()) {
          tt_move = tte->move;
        }
        is_tt_pv = tte->is_pv;
      }
    }

  }
  Player player = board.GetTurn();

  if (depth <= 0) {
    if (options_.enable_qsearch) {
      return QSearch(ss, is_pv_node ? PV : NonPV, thread_state, board, 0, alpha, beta,
          maximizing_player, deadline, pvinfo);
    }

    int eval = Evaluate(thread_state, board, maximizing_player, alpha, beta);
    if (options_.enable_transposition_table) {
      transposition_table_->Save(board.HashKey(), 0, std::nullopt, eval, eval, EXACT, is_pv_node);
    }

    return std::make_tuple(eval, std::nullopt);
  }

  int eval;
  if (tt_hit && tte->eval != value_none_tt) {
    eval = tte->eval;
  } else {
    eval = Evaluate(thread_state, board, maximizing_player, alpha, beta);
  }

  (ss+2)->killers[0] = (ss+2)->killers[1] = Move();
  ss->move_count = 0;
  if (ply == 1) {
    ss->root_depth = depth;
  }
  (ss+1)->root_depth = ss->root_depth;
  ss->static_eval = eval;
  bool improving = ply > 2
    && (ss-2)->current_move.Present()
    && (ss-2)->static_eval + 150 < ss->static_eval;
  bool declining = ply > 2
    && (ss-2)->current_move.Present()
    && ss->static_eval + 150 < (ss-2)->static_eval;

  bool in_check = board.IsKingInCheck(player);
  ss->in_check = in_check;

  if (options_.enable_futility_pruning
      && !in_check
      && !is_pv_node
      && !is_tt_pv
      && depth <= 1
      && eval - 150 * depth >= beta
      && eval < kMateValue) {
    return std::make_tuple(beta, std::nullopt);
  }

  bool partner_checked = board.IsKingInCheck(GetPartner(player));

  if (options_.enable_null_move_pruning
      && !is_root_node
      && !is_pv_node
      && null_moves == 0
      && !in_check
      && eval >= beta + 50
      && !partner_checked
      ) {
    num_null_moves_tried_++;
    ss->continuation_history = &continuation_history[0][0][NO_PIECE][0];
    ss->current_move = Move();
    board.MakeNullMove();

    PVInfo null_pvinfo;
    int r = std::min(depth / 3 + 2, depth);

    auto value_and_move_or = Search(
        ss+1, NonPV, thread_state, board, ply + 1, depth - r,
        -beta, -beta + 1, !maximizing_player, expanded, deadline, null_pvinfo,
        null_moves + 1);

    board.UndoNullMove();

    if (value_and_move_or.has_value()) {
      int nmp_score = -std::get<0>(*value_and_move_or);
      if (nmp_score >= beta
          && nmp_score < kMateValue) {
        num_null_moves_pruned_++;
        return std::make_tuple(beta, std::nullopt);
      }
    }
  }

  std::optional<Move> best_move;
  int player_color = static_cast<int>(player.GetColor());

  int curr_n_activated = thread_state.NActivated()[player_color];
  int curr_total_moves = thread_state.TotalMoves()[player_color];

  const PieceToHistory* cont_hist[] = {
    (ss - 1)->continuation_history,
    (ss - 2)->continuation_history,
    (ss - 3)->continuation_history,
    (ss - 4)->continuation_history,
    (ss - 5)->continuation_history,
  };

  std::optional<Move> pv_move = pvinfo.GetBestMove();
  Move* moves = thread_state.GetNextMoveBufferPartition();
  MovePicker move_picker(
    board,
    pv_move.has_value() ? pv_move : tt_move,
    ss->killers,
    kPieceEvaluations,
    history_heuristic,
    capture_heuristic,
    piece_move_order_scores_,
    options_.enable_move_order_checks,
    moves,
    kBufferPartitionSize,
    counter_moves,
    true,
    cont_hist
    );

  bool has_legal_moves = false;
  int move_count = 0;
  int quiets = 0;
  bool fail_low = true;
  bool fail_high = false;
  std::vector<Move> searched_moves;

  while (true) {
    Move* move_ptr = move_picker.GetNextMove();
    if (move_ptr == nullptr) {
      break;
    }

    Move& move = *move_ptr;

    if (ss->excludedMove.Present() && move == ss->excludedMove) {
      continue;
    }

    Piece piece = board.GetPiece(move.From());
    PieceType piece_type = piece.GetPieceType();
    int from_idx = BitboardImpl::LocationToIndex(move.From());
    int to_idx = BitboardImpl::LocationToIndex(move.To());

    std::optional<std::tuple<int, std::optional<Move>>> value_and_move_or;
    bool delivers_check = move.DeliversCheck(board);
    bool lmr =
      options_.enable_late_move_reduction
      && depth > 1
      && move_count > 1 + is_root_node
      && (!is_tt_pv
          || !move.IsCapture()
          || (is_cut_node && (ss-1)->move_count > 1))
         ;

    bool quiet = !in_check && !move.IsCapture() && !delivers_check;
    int q = 1 + depth*depth/(declining?10:5);
    if (is_pv_node) {
      q = 5 + depth*depth/(declining?2:1);
      if (improving) q *= 2;
    }

    if (options_.enable_late_move_pruning
        && alpha > -kMateValue
        && quiet
        && quiets >= q
        ) {
      num_lm_pruned_++;
      continue;
    }

    int r = 1 + std::max(0,(depth-5)/3) + move_count/30;
    if (quiet) r++;
    r += depth / 8;
    r += declining - improving;
    r -= in_check;
    r -= delivers_check;
    r -= is_pv_node;
    r -= move.IsCapture() && move.ApproxSEE(board, kPieceEvaluations) > 0;
    if (!move.IsCapture()) {
      int history_score = history_heuristic[piece.GetPieceType()][from_idx][to_idx];
      r -= std::clamp((history_score - 4000) / 10000, -3, 3);
    } else {
      Piece captured = move.GetCapturePiece();
      int history_score = capture_heuristic[piece.GetPieceType()][piece.GetColor()]
        [captured.GetPieceType()][captured.GetColor()]
        [to_idx];
      r -= std::clamp((history_score - 4000) / 10000, -3, 3);
    }

    r = std::max(ply >= ss->root_depth * 1.0 ? 0 : -1, r);

    int new_depth = depth - 1;
    int lmr_depth = new_depth;
    if (lmr) {
      lmr_depth = std::max(new_depth - r, 0);
    }

    if (!is_root_node
        && !is_pv_node
        && alpha > -kMateValue
        && lmr
        && move.IsCapture()
        && lmr_depth < 10
        && !in_check) {
      Piece capture_piece = move.GetCapturePiece();
      PieceType capture_piece_type = capture_piece.GetPieceType();
      int futility_eval = eval + 400 + 291 * lmr_depth + kPieceEvaluations[capture_piece_type];
      if (futility_eval < alpha) {
        continue;
      }
    }

    ss->current_move = move;
    ss->continuation_history = &continuation_history[ss->in_check][move.IsCapture()][piece_type][to_idx];

    board.MakeMove(move);

    if (board.CheckWasLastMoveKingCapture() != IN_PROGRESS) {
      board.UndoMove();
      alpha = beta;
      best_move = move;
      pvinfo.SetBestMove(move);
      break;
    }

    if (board.IsKingInCheck(player)) {
      board.UndoMove();
      continue;
    }

    has_legal_moves = true;
    ss->move_count = move_count++;
    if (quiet) quiets++;

    if (options_.enable_mobility_evaluation || options_.enable_piece_activation) {
      UpdateMobilityEvaluation(thread_state, board, player);
    }

    bool is_pv_move = pv_move.has_value() && *pv_move == move;
    std::shared_ptr<PVInfo> child_pvinfo = is_pv_move && pvinfo.GetChild() != nullptr ? pvinfo.GetChild() : std::make_shared<PVInfo>();

    int e = 0;

    if (options_.enable_singular_extensions && !is_root_node && tt_move.has_value() && move == *tt_move && !ss->excludedMove.Present() && depth >= 9 && tte != nullptr && tte->score != value_none_tt && std::abs(tte->score) < kMateValue && tte->bound == LOWER_BOUND && tte->depth >= depth - 3)
    {
      num_singular_extension_searches_.fetch_add(1, std::memory_order_relaxed);
      int singular_beta = tte->score - (58 + 76 * (ss->tt_pv && node_type == NonPV)) * depth / 57;
      int singular_depth = (depth - 1) / 2;
      ss->excludedMove = move;
      PVInfo singular_pvinfo;
      auto singular_res = Search(ss, NonPV, thread_state, board, ply, singular_depth, singular_beta - 1, singular_beta, maximizing_player, expanded, deadline, singular_pvinfo, null_moves, is_cut_node);
      ss->excludedMove = Move();
      if (singular_res.has_value()) {
        int singular_score = std::get<0>(*singular_res);
        if (singular_score < singular_beta) {
          num_singular_extensions_.fetch_add(1, std::memory_order_relaxed);
          e = 1;
        }
      }
    }

    if (options_.enable_check_extensions && delivers_check && move_count < 6 && expanded < 3) {
      num_check_extensions_++;
      e = 1;
    }

    if (lmr) {
      num_lmr_searches_++;
      r = std::clamp(r, 0, depth - 1);
      value_and_move_or = Search(
          ss+1, NonPV, thread_state, board, ply + 1, depth - 1 - r + e,
          -alpha-1, -alpha, !maximizing_player, expanded + e,
          deadline, *child_pvinfo, 0, true);
      if (value_and_move_or.has_value() && r > 0) {
        int score = -std::get<0>(*value_and_move_or);
        if (score > alpha) {
          num_lmr_researches_++;
          value_and_move_or = Search(
              ss+1, NonPV, thread_state, board, ply + 1, depth - 1 + e,
              -alpha-1, -alpha, !maximizing_player, expanded + e,
              deadline, *child_pvinfo, 0, !is_cut_node);
        }
      }
    } else if (!is_pv_node || move_count > 1) {
      if (!tt_move.has_value() && is_cut_node) r += 2;
      value_and_move_or = Search(
          ss+1, NonPV, thread_state, board, ply + 1, depth - 1 + e - (r > 3),
          -alpha-1, -alpha, !maximizing_player, expanded + e,
          deadline, *child_pvinfo, 0, !is_cut_node);
    }

    bool full_search = is_pv_node && (move_count == 1 || (value_and_move_or.has_value() && -std::get<0>(*value_and_move_or) > alpha && (is_root_node || -std::get<0>(*value_and_move_or) < beta)));
    if (full_search) {
      value_and_move_or = Search(
          ss+1, PV, thread_state, board, ply + 1, depth - 1 + e,
          -beta, -alpha, !maximizing_player, expanded + e,
          deadline, *child_pvinfo, 0, false);
    }

    board.UndoMove();

    if (options_.enable_mobility_evaluation || options_.enable_piece_activation) {
      thread_state.NActivated()[player_color] = curr_n_activated;
      thread_state.TotalMoves()[player_color] = curr_total_moves;
    }

    if (!value_and_move_or.has_value()) {
      thread_state.ReleaseMoveBufferPartition();
      return std::nullopt;
    }
    int score = -std::get<0>(*value_and_move_or);
    searched_moves.push_back(move);

    if (score >= beta) {
      alpha = beta;
      best_move = move;
      pvinfo.SetChild(child_pvinfo);
      pvinfo.SetBestMove(move);
      fail_low = false;
      fail_high = true;
      break;
    }
    if (score > alpha) {
      fail_low = false;
      alpha = score;
      best_move = move;
      pvinfo.SetChild(child_pvinfo);
      pvinfo.SetBestMove(move);
    }

    if (!best_move.has_value()) {
      best_move = move;
      pvinfo.SetChild(child_pvinfo);
      pvinfo.SetBestMove(move);
    }
  }

  if (!fail_low) {
    UpdateStats(ss, thread_state, board, *best_move, depth, fail_high,
                searched_moves);
  }

  int score = alpha;
  if (!has_legal_moves) {
    if (!in_check) score = std::min(beta, std::max(alpha, 0));
    else score = std::min(beta, std::max(alpha, -kMateValue));
  }

  if (options_.enable_transposition_table) {
    ScoreBound bound = beta <= alpha ? LOWER_BOUND : is_pv_node &&
      best_move.has_value() ? EXACT : UPPER_BOUND;
    transposition_table_->Save(board.HashKey(), depth, best_move, score, ss->static_eval, bound, is_pv_node);
  }

  if (best_move.has_value() && !best_move->IsCapture()) {
    UpdateQuietStats(ss, *best_move);
  }

  if (score <= alpha) {
    ss->tt_pv = ss->tt_pv || ((ss-1)->tt_pv && depth > 3);
  }

  thread_state.ReleaseMoveBufferPartition();
  return std::make_tuple(score, best_move);
}

std::optional<std::tuple<int, std::optional<Move>>>
AlphaBetaPlayer::QSearch(
    Stack* ss,
    NodeType node_type,
    ThreadState& thread_state,
    Board& board,
    int depth,
    int alpha,
    int beta,
    bool maximizing_player,
    const std::optional<std::chrono::time_point<std::chrono::system_clock>>& deadline,
    PVInfo& pv_info) {
  if (canceled_
      || (deadline.has_value()
        && std::chrono::system_clock::now() >= *deadline)) {
    return std::nullopt;
  }
  if (depth < 0) {
    num_nodes_++;
  }

  bool is_pv_node = node_type != NonPV;
  int tt_depth = 0;

  std::optional<Move> tt_move;
  bool tt_hit = false;

  const HashTableEntry* tte = nullptr;
  if (options_.enable_transposition_table) {
    int64_t key = board.HashKey();

    tte = transposition_table_->Get(key);
    if (tte != nullptr) {
      if (tte->key == key) {
        tt_hit = true;
        if (tte->depth >= tt_depth) {
          num_cache_hits_++;
          if (!is_pv_node
              && (tte->bound == EXACT
                || (tte->bound == LOWER_BOUND && tte->score >= beta)
                || (tte->bound == UPPER_BOUND && tte->score <= alpha))
             ) {
            
            if (tte->move.Present()) {
                return std::make_tuple(
                    std::min(beta, std::max(alpha, tte->score)), tte->move);
            }
            return std::make_tuple(
                std::min(beta, std::max(alpha, tte->score)), std::nullopt);

          }
        }
        if (tte->move.Present()) tt_move = tte->move;
      }
    }
  }

  Player player = board.GetTurn();
  bool in_check = board.IsKingInCheck(player);
  ss->in_check = in_check;

  int best_value;
  int futility_base = -kMateValue;
  int static_eval_q = value_none_tt;
  if (in_check) {
    best_value = -kMateValue;
  } else {
    if (tt_hit && tte->eval != value_none_tt) best_value = tte->eval;
    else best_value = Evaluate(thread_state, board, maximizing_player, alpha, beta);
    static_eval_q = best_value;
    if (best_value >= beta) {
      if (options_.enable_transposition_table) {
        transposition_table_->Save(board.HashKey(), 0, std::nullopt, best_value, static_eval_q, LOWER_BOUND, is_pv_node);
      }
      return std::make_tuple(best_value, std::nullopt);
    }
    if (best_value + kPieceEvaluations[QUEEN] < alpha) {
      return std::make_tuple(alpha, std::nullopt);
    }
    futility_base = best_value;
  }

  std::optional<Move> best_move;
  int player_color = static_cast<int>(player.GetColor());
  int curr_n_activated = thread_state.NActivated()[player_color];
  int curr_total_moves = thread_state.TotalMoves()[player_color];

  const PieceToHistory* cont_hist[] = {
    (ss - 1)->continuation_history, (ss - 2)->continuation_history,
    (ss - 3)->continuation_history, (ss - 4)->continuation_history,
    (ss - 5)->continuation_history,
  };

  std::optional<Move> pv_move = pv_info.GetBestMove();
  Move* moves = thread_state.GetNextMoveBufferPartition();
  MovePicker move_picker(
    board, pv_move, ss->killers, kPieceEvaluations,
    history_heuristic, capture_heuristic, piece_move_order_scores_,
    options_.enable_move_order_checks, moves, kBufferPartitionSize,
    counter_moves, in_check, cont_hist
  );

  int move_count = 0;
  int quiet_check_evasions = 0;
  bool fail_low = true;
  bool fail_high = false;
  std::vector<Move> searched_moves;

  while (true) {
    Move* move_ptr = move_picker.GetNextMove();
    if (move_ptr == nullptr) break;
    Move& move = *move_ptr;
    bool capture = move.IsCapture();
    if (!in_check) {
      if (capture) {
        if (move.GetStandardCapture().Present()) {
          if (move.GetCapturePiece().GetPieceType() != QUEEN
              && board.GetPiece(move.From()).GetPieceType() != PAWN) {
            if (StaticExchangeEvaluationCapture(kPieceEvaluations, board, move) < 0) {
              continue;
            }
          }
        }
      } else {
        continue;
      }
    }

    std::optional<std::tuple<int, std::optional<Move>>> value_and_move_or;
    PieceType piece_type = board.GetPiece(move.From()).GetPieceType();
    int to_idx = BitboardImpl::LocationToIndex(move.To());

    ss->current_move = move;
    ss->continuation_history = &continuation_history[ss->in_check][move.IsCapture()][piece_type][to_idx];
    
    bool delivers_check = move.DeliversCheck(board);
    board.MakeMove(move);
    if (board.CheckWasLastMoveKingCapture() != IN_PROGRESS) {
      board.UndoMove();
      best_value = beta;
      best_move = move;
      pv_info.SetBestMove(move);
      break;
    }

    if (board.IsKingInCheck(player)) {
      board.UndoMove();
      continue;
    }

    move_count++;
    if (best_value > -kMateValue) {
      if ((!delivers_check && move_count > 2) || quiet_check_evasions > 1) {
        board.UndoMove();
        continue;
      }
      if (move.IsCapture() && !delivers_check && futility_base + kPieceEvaluations[move.GetCapturePiece().GetPieceType()] < alpha) {
        board.UndoMove();
        continue;
      }
    }
    quiet_check_evasions += !capture && in_check;

    if (options_.enable_mobility_evaluation || options_.enable_piece_activation) {
      UpdateMobilityEvaluation(thread_state, board, player);
    }
    
    bool is_pv_move = pv_move.has_value() && *pv_move == move;
    std::shared_ptr<PVInfo> child_pvinfo = is_pv_move && pv_info.GetChild() != nullptr ? pv_info.GetChild() : std::make_shared<PVInfo>();

    value_and_move_or = QSearch(
        ss+1, node_type, thread_state, board, depth - 1, -beta, -alpha, !maximizing_player,
        deadline, *child_pvinfo);

    board.UndoMove();

    if (options_.enable_mobility_evaluation || options_.enable_piece_activation) {
      thread_state.NActivated()[player_color] = curr_n_activated;
      thread_state.TotalMoves()[player_color] = curr_total_moves;
    }

    if (!value_and_move_or.has_value()) {
      thread_state.ReleaseMoveBufferPartition();
      return std::nullopt;
    }
    int score = -std::get<0>(*value_and_move_or);
    searched_moves.push_back(move);

    if (!best_move.has_value()) {
      best_move = move;
      pv_info.SetChild(child_pvinfo);
      pv_info.SetBestMove(move);
    }
    if (score > best_value) {
      best_value = score;
      if (score > alpha) {
        fail_low = false;
        best_move = move;
        if (is_pv_node) {
          pv_info.SetChild(child_pvinfo);
          pv_info.SetBestMove(move);
        }
        if (score < beta) alpha = score;
        else {
          fail_high = true;
          break;
        }
      }
    }
  }

  if (!fail_low) {
    UpdateStats(ss, thread_state, board, *best_move, 0, fail_high,
                searched_moves);
  }

  int score = best_value;
  if (in_check && best_value == -kMateValue) {
    score = std::min(beta, std::max(alpha, -kMateValue));
  } else {
    score = std::min(beta, std::max(alpha, best_value));
  }

  if (options_.enable_transposition_table) {
    ScoreBound bound = fail_high ? LOWER_BOUND : (fail_low && is_pv_node ? EXACT : UPPER_BOUND);
    transposition_table_->Save(board.HashKey(), tt_depth, best_move, score, static_eval_q, bound, is_pv_node);
  }

  thread_state.ReleaseMoveBufferPartition();
  return std::make_tuple(score, best_move);
}

void AlphaBetaPlayer::UpdateStats(
    Stack* ss, ThreadState& thread_state, const Board& board,
    const Move& move, int depth, bool fail_high,
    const std::vector<Move>& searched_moves) {
  
  int from_idx = BitboardImpl::LocationToIndex(move.From());
  int to_idx = BitboardImpl::LocationToIndex(move.To());
  Piece piece = board.GetPiece(move.From());

  int bonus = 1 << (fail_high ? depth + 1: depth);

  if (move.IsCapture()) {
    Piece captured = move.GetCapturePiece();
    size_t lock_key = to_idx;
    std::lock_guard<std::mutex> lock(heuristic_mutexes_[lock_key % kHeuristicMutexes]);
    capture_heuristic[piece.GetPieceType()][piece.GetColor()]
      [captured.GetPieceType()][captured.GetColor()]
      [to_idx] += bonus;
  } else {
    size_t lock_key = (from_idx * 256 + to_idx);
    std::lock_guard<std::mutex> lock(heuristic_mutexes_[lock_key % kHeuristicMutexes]);
    if (options_.enable_history_heuristic) {
      history_heuristic[piece.GetPieceType()][from_idx][to_idx] += bonus;
    }
    if (options_.enable_counter_move_heuristic) {
      counter_moves[from_idx][to_idx] = move;
    }
    UpdateQuietStats(ss, move);
    UpdateContinuationHistories(ss, move, piece.GetPieceType(), bonus);
  }
  for (const auto& other_move : searched_moves) {
    if (other_move != move) {
      int other_from_idx = BitboardImpl::LocationToIndex(other_move.From());
      int other_to_idx = BitboardImpl::LocationToIndex(other_move.To());
      Piece other_piece = board.GetPiece(other_from_idx);

      if (other_move.IsCapture()) {
        Piece other_captured = other_move.GetCapturePiece();
        size_t lock_key = other_to_idx;
        std::lock_guard<std::mutex> lock(heuristic_mutexes_[lock_key % kHeuristicMutexes]);
        capture_heuristic[other_piece.GetPieceType()][other_piece.GetColor()]
          [other_captured.GetPieceType()][other_captured.GetColor()]
          [other_to_idx] -= bonus;
      } else {
        size_t lock_key = (other_from_idx * 256 + other_to_idx);
        std::lock_guard<std::mutex> lock(heuristic_mutexes_[lock_key % kHeuristicMutexes]);
        history_heuristic[other_piece.GetPieceType()][other_from_idx][other_to_idx] -= bonus;
      }
    }
  }
}

void AlphaBetaPlayer::UpdateQuietStats(Stack* ss, const Move& move) {
  if (options_.enable_killers) {
    if (ss->killers[0] != move) {
      ss->killers[1] = ss->killers[0];
      ss->killers[0] = move;
    }
  }
}

void AlphaBetaPlayer::UpdateContinuationHistories(Stack* ss, const Move& move, PieceType piece_type, int bonus) {
  const int to_idx = BitboardImpl::LocationToIndex(move.To());
  for (int i : {1, 2, 3, 4, 5, 6}) {
    if (ss->in_check && i > 2) break;
    if ((ss-i)->current_move.Present()) {
      (*(ss-i)->continuation_history)[piece_type][to_idx] << bonus;
    }
  }
}

namespace {
constexpr int kPieceImbalanceTable[16] = {
  0, -25, -50, -150, -300, -350, -400, -400,
  -400, -400, -400, -400, -400, -400, -400, -400,
};
}

int AlphaBetaPlayer::Evaluate(
    ThreadState& thread_state, Board& board, bool maximizing_player, int alpha, int beta) {
  int eval;
  GameResult game_result = board.CheckWasLastMoveKingCapture();
  if (game_result != IN_PROGRESS) {
    if (game_result == WIN_RY) eval = kMateValue;
    else if (game_result == WIN_BG) eval = -kMateValue;
    else eval = 0;
  } else {
    eval = board.PieceEvaluation();

    auto threat_value = [](int t1, int t2) { return 120 * (t1 + t2); };
    eval += threat_value(thread_state.n_threats[RED], thread_state.n_threats[YELLOW]);
    eval -= threat_value(thread_state.n_threats[BLUE], thread_state.n_threats[GREEN]);

    int n_queen_ry = 0;
    int n_queen_bg = 0;
    int n_major_ry[4] = {0,0,0,0}; // Count major pieces for each color
    
    Bitboard all_pawns = board.piece_bitboards_[0][PAWN] | board.piece_bitboards_[1][PAWN] | board.piece_bitboards_[2][PAWN] | board.piece_bitboards_[3][PAWN];

    for (int color_idx = 0; color_idx < 4; ++color_idx) {
        PlayerColor color = static_cast<PlayerColor>(color_idx);
        Team team = GetTeam(color);
        int team_sign = (team == RED_YELLOW) ? 1 : -1;

        for (int pt_idx = 0; pt_idx < 6; ++pt_idx) {
            PieceType piece_type = static_cast<PieceType>(pt_idx);
            Bitboard bb = board.piece_bitboards_[color][piece_type];

            if (piece_type != PAWN && piece_type != KING) {
                n_major_ry[color] = bb.popcount();
            }

            while(!bb.is_zero()) {
                int sq = bb.ctz();
                bb &= (bb - 1);
                
                if (piece_type == QUEEN) {
                    if (team == RED_YELLOW) n_queen_ry++; else n_queen_bg++;
                }

                if (options_.enable_piece_square_table) {
                    eval += team_sign * piece_square_table_[color][piece_type][sq];
                }
                
                if (piece_type == PAWN) {
                    BoardLocation loc = BitboardImpl::IndexToLocation(sq);
                    int row = loc.GetRow(); int col = loc.GetCol();
                    int advancement = 0;
                    switch (color) {
                        case RED:    advancement = 12 - row; break;
                        case YELLOW: advancement = row - 1;  break;
                        case BLUE:   advancement = col - 1;  break;
                        case GREEN:  advancement = 12 - col; break;
                        default: break;
                    }
                    int bonus = 2 * advancement * advancement + std::max(0, 150 * (advancement - 5));
                    eval += team_sign * bonus;
                }

                // RESTORED: Rook on open/semi-open file bonus
                if (piece_type == ROOK) {
                    Bitboard file_mask;
                    if (color == RED || color == YELLOW) { // Vertical rooks
                        file_mask = BitboardImpl::kRayAttacks[sq][BitboardImpl::D_N] | BitboardImpl::kRayAttacks[sq][BitboardImpl::D_S];
                    } else { // Horizontal rooks
                        file_mask = BitboardImpl::kRayAttacks[sq][BitboardImpl::D_E] | BitboardImpl::kRayAttacks[sq][BitboardImpl::D_W];
                    }

                    if ((file_mask & all_pawns).is_zero()) { // Open file
                        eval += team_sign * 25;
                    } else if ((file_mask & board.piece_bitboards_[color][PAWN]).is_zero()) { // Semi-open file
                        eval += team_sign * 15;
                    }
                }

                if (options_.enable_knight_bonus && piece_type == KNIGHT) {
                    int knight_bonus = 0;
                    for (int i = 0; i < 2; i++) {
                        PlayerColor other_color = static_cast<PlayerColor>((color + 2 * i + 1) % 4);
                        BoardLocation king_loc = board.GetKingLocation(other_color);
                        if (king_loc.Present()) {
                            int king_sq = BitboardImpl::LocationToIndex(king_loc);
                            if (king_sq >= 0 && knight_to_king_[sq][king_sq]) {
                                knight_bonus += 100;
                            }
                        }
                    }
                    eval += team_sign * knight_bonus;
                }
            }
        }
    }
    
    int activation_ry = 0, activation_bg = 0;
    if (options_.enable_piece_activation) {
      auto team_activation_score = [](int n1, int n2) { return 35 * (n1 + n2) + 20 * n1 * n2; };
      int* n_activated = thread_state.NActivated();
      activation_ry = team_activation_score(n_activated[RED], n_activated[YELLOW]);
      activation_bg = team_activation_score(n_activated[BLUE], n_activated[GREEN]);
      eval += activation_ry - activation_bg;
    }
    
    int* total_moves = thread_state.TotalMoves();
    if (options_.enable_mobility_evaluation) {
        eval += 2 * (total_moves[RED] + total_moves[YELLOW] - total_moves[BLUE] - total_moves[GREEN]);
    }
    
    // FIXED: Piece imbalance logic now uses major piece counts
    if (options_.enable_piece_imbalance) {
      int n_major_red = n_major_ry[RED];
      int n_major_yellow = n_major_ry[YELLOW];
      int n_major_blue = n_major_ry[BLUE];
      int n_major_green = n_major_ry[GREEN];

      int diff_ry = std::abs(n_major_red - n_major_yellow);
      int diff_bg = std::abs(n_major_blue - n_major_green);
      
      eval += kPieceImbalanceTable[diff_ry] - kPieceImbalanceTable[diff_bg];
    }

    constexpr int kKingSafetyMargin = 600;
    if (options_.enable_lazy_eval) {
        int re = maximizing_player ? eval : -eval;
        if (re + kKingSafetyMargin <= alpha || re >= beta + kKingSafetyMargin) {
            num_lazy_eval_++;
            return re;
        }
    }

    if (options_.enable_king_safety) {
      for (int color_idx = 0; color_idx < 4; ++color_idx) {
        PlayerColor color = static_cast<PlayerColor>(color_idx);
        Team team = GetTeam(color);
        BoardLocation king_location = board.GetKingLocation(color);
        if (king_location.Present()) {
          int king_sq = BitboardImpl::LocationToIndex(king_location);
          int team_sign = (team == RED_YELLOW) ? 1 : -1;
          int king_safety = 0;
          bool opponent_has_queen = (team == RED_YELLOW && n_queen_bg > 0) || (team == BLUE_GREEN && n_queen_ry > 0);

          if (options_.enable_pawn_shield && opponent_has_queen) {
            if (!HasShield(board, color, king_sq)) king_safety -= 75;
            if (!OnBackRank(king_sq)) king_safety -= 50;
          }

          // RESTORED: Full king attack zone logic
          if (options_.enable_attacking_king_zone) {
            Bitboard king_zone = BitboardImpl::kKingAttacks[king_sq];
            int safety = 0;
            while(!king_zone.is_zero()) {
                int zone_sq = king_zone.ctz();
                king_zone &= (king_zone - 1);
                
                Bitboard attackers = board.GetAttackersBB(zone_sq, NO_TEAM);
                int value_of_attacks = 0, num_attackers = 0;
                int value_of_protection = 0, num_protectors = 0;

                while(!attackers.is_zero()) {
                    int attacker_sq = attackers.ctz();
                    attackers &= (attackers - 1);
                    Piece p = board.GetPiece(attacker_sq);
                    if(p.GetPieceType() == KING) continue;

                    int val = king_attacker_values_[p.GetPieceType()];
                    if(p.GetTeam() == team) {
                        num_protectors++;
                        value_of_protection += val;
                    } else {
                        num_attackers++;
                        value_of_attacks += val;
                    }
                }
                int attack_zone_val = value_of_attacks * king_attack_weight_[num_attackers] / 100;
                attack_zone_val -= value_of_protection * king_attack_weight_[num_protectors] / 200;
                safety -= std::max(0, attack_zone_val);
            }
            if (!opponent_has_queen) safety /= 2;
            king_safety += std::min(0, safety);
          }
          eval += team_sign * king_safety;
        }
      }
    }
  }
  return maximizing_player ? eval : -eval;
}

void AlphaBetaPlayer::ResetHistoryHeuristics() {
  std::memset(history_heuristic, 0, sizeof(history_heuristic));
  std::memset(capture_heuristic, 0, sizeof(capture_heuristic));
  std::memset(counter_moves, 0, sizeof(Move) * 256 * 256);
  // FIXED: Use . instead of ->
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
    // We must use memset on the object, not fill on the collection of objects
    std::memset(&continuation_history[i][j], 0, sizeof(continuation_history[i][j]));
  }
}

void AlphaBetaPlayer::AgeHistoryHeuristics() {
  auto age_table = [](auto* table, size_t size_bytes) {
      int* p = reinterpret_cast<int*>(table);
      for(size_t i = 0; i < size_bytes / sizeof(int); ++i) p[i] /= 2;
  };
  age_table(history_heuristic, sizeof(history_heuristic));
  age_table(capture_heuristic, sizeof(capture_heuristic));
  std::memset(counter_moves, 0, sizeof(Move) * 256 * 256);
  // FIXED: Use . instead of -> and memset instead of fill
  for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) {
      std::memset(&continuation_history[i][j], 0, sizeof(continuation_history[i][j]));
  }
}

void AlphaBetaPlayer::ResetMobilityScores(ThreadState& thread_state, Board& board) {
  if (options_.enable_mobility_evaluation || options_.enable_piece_activation) {
    for (int i = 0; i < 4; i++) {
      Player player(static_cast<PlayerColor>(i));
      UpdateMobilityEvaluation(thread_state, board, player);
    }
  }
}

int AlphaBetaPlayer::StaticEvaluation(Board& board) {
  auto pv_copy = pv_info_.Copy();
  ThreadState thread_state(options_, board, *pv_copy);
  ResetMobilityScores(thread_state, board);
  return Evaluate(thread_state, board, true, -kMateValue, kMateValue);
}

std::optional<std::tuple<int, std::optional<Move>, int>>
AlphaBetaPlayer::MakeMove(
    Board& board,
    std::optional<std::chrono::milliseconds> time_limit,
    int max_depth) {
  root_team_ = board.GetTurn().GetTeam();
  if (board.HashKey() != last_board_key_) {
    average_root_eval_ = 0;
    asp_nobs_ = 0;
    asp_sum_ = 0;
    asp_sum_sq_ = 0;
  }
  last_board_key_ = board.HashKey();

  SetCanceled(false);
  std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline;
  if (time_limit.has_value()) {
    deadline = std::chrono::system_clock::now() + *time_limit;
  }
  if (options_.max_search_depth.has_value()) {
    max_depth = std::min(max_depth, *options_.max_search_depth);
  }

  AgeHistoryHeuristics();

  int num_threads = options_.enable_multithreading ? options_.num_threads : 1;
  std::vector<ThreadState> thread_states;
  thread_states.reserve(num_threads);
  for (int i = 0; i < num_threads; i++) {
    auto pv_copy = pv_info_.Copy();
    thread_states.emplace_back(options_, board, *pv_copy);
    ResetMobilityScores(thread_states.back(), board);
  }

  std::vector<std::unique_ptr<std::thread>> threads;
  for (size_t i = 1; i < num_threads; i++) {
    threads.push_back(std::make_unique<std::thread>([
      this, i, &thread_states, deadline, max_depth] {
      MakeMoveSingleThread(i, thread_states[i], deadline,
          max_depth);
    }));
  }

  auto res = MakeMoveSingleThread(0, thread_states[0], deadline, max_depth);

  SetCanceled(true);
  for (auto& thread : threads) {
    thread->join();
  }

  if (res.has_value()) {
      pv_info_ = thread_states[0].GetPVInfo();
  }

  SetCanceled(false);
  return res;
}

std::optional<std::tuple<int, std::optional<Move>, int>>
AlphaBetaPlayer::MakeMoveSingleThread(
    size_t thread_id,
    ThreadState& thread_state,
    std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline,
    int max_depth) {
  Board board = thread_state.GetRootBoard();
  PVInfo& pv_info = thread_state.GetPVInfo();

  int next_depth = std::min(1 + pv_info.GetDepth(), max_depth);
  std::optional<std::tuple<int, std::optional<Move>>> res;
  int alpha = -kMateValue;
  int beta = kMateValue;
  bool maximizing_player = board.TeamToPlay() == RED_YELLOW;
  int searched_depth = 0;
  Stack stack[kMaxPly + 10];
  Stack* ss = stack + 7;
  for (int i = 7; i > 0; i--) {
    (ss-i)->continuation_history = &continuation_history[0][0][NO_PIECE][0];
  }

  if (options_.enable_aspiration_window) {
    while (next_depth <= max_depth) {
      std::optional<std::tuple<int, std::optional<Move>>> move_and_value;
      if (thread_id == 0) {
          int prev = average_root_eval_;
          int delta = 50;
          if (asp_nobs_ > 0) delta = 50 + std::sqrt((asp_sum_sq_ - asp_sum_*asp_sum_/(double)asp_nobs_)/asp_nobs_);
          alpha = std::max(prev - delta, -kMateValue);
          beta = std::min(prev + delta, kMateValue);
          int fail_cnt = 0;

          while (true) {
            move_and_value = Search(ss, Root, thread_state, board, 1, next_depth, alpha, beta, maximizing_player, 0, deadline, pv_info);
            if (!move_and_value.has_value()) break;
            int evaluation = std::get<0>(*move_and_value);
            if (asp_nobs_ == 0) average_root_eval_ = evaluation;
            else average_root_eval_ = (2 * evaluation + average_root_eval_) / 3;
            asp_nobs_++;
            asp_sum_ += evaluation;
            asp_sum_sq_ += evaluation * evaluation;

            if (std::abs(evaluation) == kMateValue) break;
            if (evaluation <= alpha) {
              beta = (alpha + beta) / 2;
              alpha = std::max(evaluation - delta, -kMateValue);
              ++fail_cnt;
            } else if (evaluation >= beta) {
              beta = std::min(evaluation + delta, kMateValue);
              ++fail_cnt;
            } else break;
            if (fail_cnt >= 5) { alpha = -kMateValue; beta = kMateValue; }
            delta += delta / 3;
          }
      } else {
          move_and_value = Search(ss, Root, thread_state, board, 1, next_depth, -kMateValue, kMateValue, maximizing_player, 0, deadline, pv_info);
      }
      if (!move_and_value.has_value()) break;
      res = move_and_value;
      searched_depth = next_depth;
      next_depth++;
      if (std::abs(std::get<0>(*move_and_value)) == kMateValue) break;
    }
  } else {
    while (next_depth <= max_depth) {
      auto move_and_value = Search(ss, Root, thread_state, board, 1, next_depth, alpha, beta, maximizing_player, 0, deadline, pv_info);
      if (!move_and_value.has_value()) break;
      res = move_and_value;
      searched_depth = next_depth;
      next_depth++;
      if (std::abs(std::get<0>(*move_and_value)) == kMateValue) break;
    }
  }

  if (res.has_value()) {
    int eval = std::get<0>(*res);
    if (!maximizing_player) eval = -eval;
    return std::make_tuple(eval, std::get<1>(*res), searched_depth);
  }
  return std::nullopt;
}

int PVInfo::GetDepth() const {
  if (best_move_.has_value()) {
    return 1 + (child_ == nullptr ? 0 : child_->GetDepth());
  }
  return 0;
}

void AlphaBetaPlayer::UpdateMobilityEvaluation(
    ThreadState& thread_state, Board& board, Player player) {
  Move* moves = thread_state.GetNextMoveBufferPartition();
  Player curr_player = board.GetTurn();
  board.SetPlayer(player);
  size_t num_moves = board.GetPseudoLegalMoves2(moves, kBufferPartitionSize);
  int color = player.GetColor();
  thread_state.TotalMoves()[color] = num_moves;
  
  int n_threats = 0;
  for(size_t i = 0; i < num_moves; ++i) {
      if (moves[i].IsCapture() && moves[i].ApproxSEE(board, kPieceEvaluations) >= 100) {
          n_threats++;
      }
  }
  thread_state.n_threats[color] = n_threats;
  
  if (options_.enable_piece_activation) {
    int n_pieces_activated = 0;
    for (int pt_idx = KNIGHT; pt_idx <= QUEEN; ++pt_idx) {
        Bitboard bb = board.piece_bitboards_[color][static_cast<PieceType>(pt_idx)];
        while(!bb.is_zero()) {
            int sq = bb.ctz();
            bb &= (bb - 1);
            if(!OnBackRank(sq)) n_pieces_activated++;
        }
    }
    thread_state.NActivated()[color] = n_pieces_activated;
  }

  board.SetPlayer(curr_player);
  thread_state.ReleaseMoveBufferPartition();
}

bool AlphaBetaPlayer::OnBackRank(int sq) {
  if (sq < 0) return false;
  BoardLocation loc = BitboardImpl::IndexToLocation(sq);
  return !loc.Present() || loc.GetRow() == 0 || loc.GetRow() == 13 || loc.GetCol() == 0 || loc.GetCol() == 13;
}

bool AlphaBetaPlayer::HasShield(const Board& board, PlayerColor color, int king_sq) {
  Bitboard shield_rays;
  switch (color) {
    case RED:    shield_rays = BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_NW] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_N] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_NE]; break;
    case BLUE:   shield_rays = BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_NE] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_E] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_SE]; break;
    case YELLOW: shield_rays = BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_SE] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_S] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_SW]; break;
    case GREEN:  shield_rays = BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_SW] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_W] | BitboardImpl::kRayAttacks[king_sq][BitboardImpl::D_NW]; break;
    default:     return false;
  }
  return !( (shield_rays & board.piece_bitboards_[color][PAWN]).is_zero() );
}

std::shared_ptr<PVInfo> PVInfo::Copy() const {
  std::shared_ptr<PVInfo> copy = std::make_shared<PVInfo>();
  if (best_move_.has_value()) {
    copy->SetBestMove(*best_move_);
  }
  if (child_ != nullptr) {
    copy->SetChild(child_->Copy());
  }
  return copy;
}

}  // namespace chess