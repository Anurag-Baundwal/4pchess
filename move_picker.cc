#include "move_picker.h"

#include <algorithm>
#include <iostream>

namespace chess {

enum Stage {
  PV_MOVE = 0,
  GOOD_CAPTURE = 1,
  KILLER = 2,
  BAD_CAPTURE = 3,
  QUIET = 4,
};

static inline Piece GetCapturePiece(const Board& board, const Move& move) {
    if (move.IsEnPassant()) {
        return board.GetPiece(move.GetEnpassantLocation());
    }
    return board.GetPiece(move.To());
}

MovePicker::MovePicker(
    Board& board,
    const std::optional<Move>& pvmove,
    Move* killers,
    const int piece_evaluations[6],
    int history_heuristic[6][14][14][14][14],
    int capture_heuristic[6][4][6][4][14][14],
    int piece_move_order_scores[6],
    bool enable_move_order_checks,
    ExtMove* buffer,
    size_t buffer_size,
    Move* counter_moves,
    const SafetyInfo& safety,
    bool include_quiets,
    const PieceToHistory** piece_to_history
    ) {
  enable_move_order_checks_ = enable_move_order_checks;
  moves_ = buffer;
  
  ExtMove* end_ptr = buffer;
  switch (board.GetTurn().GetColor()) {
      case RED:    end_ptr = board.GenerateMovesT<RED>(buffer, safety); break;
      case BLUE:   end_ptr = board.GenerateMovesT<BLUE>(buffer, safety); break;
      case YELLOW: end_ptr = board.GenerateMovesT<YELLOW>(buffer, safety); break;
      case GREEN:  end_ptr = board.GenerateMovesT<GREEN>(buffer, safety); break;
      default:     break;
  }

  num_moves_ = end_ptr - buffer;
  board_ = &board;

  for (size_t i = 0; i < num_moves_; i++) {
    auto& move = moves_[i];

    const auto capture = GetCapturePiece(board, move);
    bool is_capture = capture.Present();
    
    const auto piece = board.GetPiece(move.From());
    const auto piece_type = piece.GetPieceType();
    const auto& from = move.From();
    const auto& to = move.To();

    int score = piece_move_order_scores[piece.GetPieceType()];
    
    int stage_idx = -1;

    if (pvmove.has_value() && move == *pvmove) {
      stage_idx = PV_MOVE;
    } else if (killers != nullptr
               && (killers[0] == move || killers[1] == move)
               && include_quiets) {
      stage_idx = KILLER;
      score += (move == killers[0] ? 1 : 0);
    } else if (is_capture) { 
      int captured_val = piece_evaluations[capture.GetPieceType()];
      int attacker_val = piece_evaluations[piece.GetPieceType()];
      int incr_score = captured_val - attacker_val/100;
      score += incr_score;
      int history_score = capture_heuristic[piece.GetPieceType()][piece.GetColor()]
        [capture.GetPieceType()][capture.GetColor()]
        [to.GetRow()][to.GetCol()];
      score += history_score;
      if (attacker_val <= captured_val) {
        stage_idx = GOOD_CAPTURE;
      } else {
        stage_idx = BAD_CAPTURE;
      }
    } else if (include_quiets) {
      score += history_heuristic[piece.GetPieceType()][from.GetRow()][from.GetCol()][to.GetRow()][to.GetCol()] / 2;
      int cm_idx = from.GetRow()*14*14*14 + from.GetCol()*14*14 + to.GetRow()*14 + to.GetCol();
      if (move == counter_moves[cm_idx]) {
        score += 50;
      }
      if (piece_to_history) {
          score += (*piece_to_history[0])[piece_type][to.GetRow()][to.GetCol()] / 2;
          score += (*piece_to_history[1])[piece_type][to.GetRow()][to.GetCol()] / 4;
          score += (*piece_to_history[2])[piece_type][to.GetRow()][to.GetCol()] / 4;
          score += (*piece_to_history[3])[piece_type][to.GetRow()][to.GetCol()] / 4;
          score += (*piece_to_history[4])[piece_type][to.GetRow()][to.GetCol()] / 4;
      }
      stage_idx = QUIET;
    }

    if (stage_idx != -1) {
        auto& stage = stages_[stage_idx];
        if (stage.count < kMaxStageMoves) {
            stage.items[stage.count++] = Item(static_cast<short>(i), static_cast<float>(score));
        }
    }
  }
}

Move* MovePicker::GetNextMove() {
  while (stage_ < 5 && stage_idx_ >= stages_[stage_].count) {
    stage_++;
    stage_idx_ = 0;
  }
  if (stage_ >= 5) {
    return nullptr;
  }

  auto& stage_buf = stages_[stage_];
  
  if (!init_stages_[stage_]) {
    if (stage_buf.count > 1) {
      if (enable_move_order_checks_) {
        for (int i = 0; i < stage_buf.count; ++i) {
          // DeliversCheck uses the board to determine checks
          if (moves_[stage_buf.items[i].index].DeliversCheck(*board_)) {
            stage_buf.items[i].score += (stage_ == QUIET ? 100'000.0f : 1000.0f);
          }
        }
      }

      // Sort the array portion
      std::sort(stage_buf.items, stage_buf.items + stage_buf.count, 
        [](const Item& a, const Item& b) {
          return a.score > b.score;
        });
    }
    init_stages_[stage_] = true;
  }

  Move* move = &moves_[stage_buf.items[stage_idx_].index];
  stage_idx_++;

  return move;
}

}  // namespace chess