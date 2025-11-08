#include "move_picker.h"

#include <algorithm>
#include <iostream>

namespace chess {

// Corrected constructor
MovePicker::MovePicker(
    Board& board,
    const std::optional<Move>& pvmove,
    Move* killers,
    const int* piece_evaluations,
    int (*history_heuristic)[14][14][14][14],
    int (*capture_heuristic)[4][6][4][14][14],
    const int* piece_move_order_scores,
    bool enable_move_order_checks,
    Move* counter_moves,
    bool include_quiets,
    const PieceToHistory** piece_to_history
) : board_(board), pv_move_(pvmove), killers_(killers), 
    piece_evals_(piece_evaluations), history_heuristic_(history_heuristic),
    capture_heuristic_(capture_heuristic), piece_scores_(piece_move_order_scores),
    counter_moves_(counter_moves), piece_to_history_(piece_to_history),
    include_quiets_(include_quiets), move_order_checks_(enable_move_order_checks)
{
    // Initialization is now minimal. The work is done in GetNextMove().
}

void MovePicker::score_captures() {
    num_scored_moves_ = 0;
    for (int i = 0; i < num_moves_; ++i) {
        const Move& move = move_buffer_[i];
        const Piece captured = move.GetCapturePiece();
        const Piece piece = board_.GetPiece(move.From());
        
        int score = piece_scores_[piece.GetPieceType()];
        int captured_val = piece_evals_[captured.GetPieceType()];
        int attacker_val = piece_evals_[piece.GetPieceType()];
        
        score += captured_val - attacker_val / 100;
        // CORRECTED: Removed incorrect dereference of pointer
        score += capture_heuristic_[piece.GetPieceType()][piece.GetColor()]
                                    [captured.GetPieceType()][captured.GetColor()]
                                    [move.To().GetRow()][move.To().GetCol()];
        
        scored_moves_[num_scored_moves_++] = {move, score};
    }
}

void MovePicker::score_quiets() {
    num_scored_moves_ = 0;
    for (int i = 0; i < num_moves_; ++i) {
        const Move& move = move_buffer_[i];
        const Piece piece = board_.GetPiece(move.From());
        const auto& from = move.From();
        const auto& to = move.To();

        int score = piece_scores_[piece.GetPieceType()];
        // CORRECTED: Removed incorrect dereference of pointer
        score += history_heuristic_[piece.GetPieceType()][from.GetRow()][from.GetCol()][to.GetRow()][to.GetCol()] / 2;

        if (counter_moves_ && move == counter_moves_[from.GetRow()*14*14*14 + from.GetCol()*14*14 + to.GetRow()*14 + to.GetCol()]) {
            score += 50;
        }
        
        if (piece_to_history_) {
            // CORRECTED: Removed incorrect dereference of pointer
            score += (*piece_to_history_[0])[piece.GetPieceType()][to.GetRow()][to.GetCol()] / 2;
            score += (*piece_to_history_[1])[piece.GetPieceType()][to.GetRow()][to.GetCol()] / 4;
            score += (*piece_to_history_[2])[piece.GetPieceType()][to.GetRow()][to.GetCol()] / 4;
            score += (*piece_to_history_[3])[piece.GetPieceType()][to.GetRow()][to.GetCol()] / 4;
            score += (*piece_to_history_[4])[piece.GetPieceType()][to.GetRow()][to.GetCol()] / 4;
        }

        scored_moves_[num_scored_moves_++] = {move, score};
    }
}


Move* MovePicker::GetNextMove() {
    while (true) {
        switch (stage_) {
            case PV_MOVE:
                stage_ = GENERATE_CAPTURES;
                if (pv_move_.has_value() && pv_move_->Present()) {
                    // CORRECTED: pv_move_ is now a member copy, safe to return pointer to
                    return &(*pv_move_);
                }
                // Fallthrough to next stage

            case GENERATE_CAPTURES:
            {
                Move* end = board_.generate<CAPTURES>(move_buffer_);
                num_moves_ = end - move_buffer_;
                score_captures();

                // Partition into good and bad captures
                MoveScore* good_captures_end = std::partition(scored_moves_, scored_moves_ + num_scored_moves_, 
                    [&](const MoveScore& ms){ return ms.move.SEE(board_, piece_evals_) >= 0; });
                
                good_captures_end_idx_ = good_captures_end - scored_moves_;

                // Sort good captures
                std::sort(scored_moves_, good_captures_end, 
                    [](const MoveScore& a, const MoveScore& b){ return a.score > b.score; });
                
                // Sort bad captures separately
                 std::sort(good_captures_end, scored_moves_ + num_scored_moves_, 
                    [](const MoveScore& a, const MoveScore& b){ return a.score > b.score; });

                stage_ = GOOD_CAPTURES;
                current_move_idx_ = 0;
            }
            // Fallthrough

            case GOOD_CAPTURES:
                if (current_move_idx_ < good_captures_end_idx_) {
                    Move& move = scored_moves_[current_move_idx_++].move;
                    if (pv_move_.has_value() && move == *pv_move_) continue;
                    return &scored_moves_[current_move_idx_ - 1].move;
                }
                stage_ = include_quiets_ ? KILLERS : BAD_CAPTURES;
                break;
            
            case KILLERS:
                stage_ = GENERATE_QUIETS; // Prepare for next stage
                if (killers_) {
                    for (; killer_idx_ < 2; ++killer_idx_) {
                        Move& killer = killers_[killer_idx_];
                        // A killer must be a quiet move and not the PV move
                        if (killer.Present() && !killer.IsCapture() &&
                            (!pv_move_.has_value() || killer != *pv_move_)) {
                           return &killer;
                        }
                    }
                }
                // Fallthrough
            
            case GENERATE_QUIETS:
            {
                if (!include_quiets_) {
                    stage_ = BAD_CAPTURES;
                    continue;
                }
                Move* end = board_.generate<QUIETS>(move_buffer_);
                num_moves_ = end - move_buffer_;
                score_quiets();
                
                if (move_order_checks_) {
                    for(int i=0; i<num_scored_moves_; ++i) {
                        if (scored_moves_[i].move.DeliversCheck(board_)) {
                            scored_moves_[i].score += 100000;
                        }
                    }
                }
                
                std::sort(scored_moves_, scored_moves_ + num_scored_moves_, 
                    [](const MoveScore& a, const MoveScore& b){ return a.score > b.score; });
                
                stage_ = QUIET_MOVES;
                current_move_idx_ = 0;
            }
            // Fallthrough

            case QUIET_MOVES:
                if (current_move_idx_ < num_scored_moves_) {
                    Move& move = scored_moves_[current_move_idx_++].move;
                    bool is_pv = pv_move_.has_value() && move == *pv_move_;
                    bool is_killer = killers_ && (move == killers_[0] || move == killers_[1]);
                    if (!is_pv && !is_killer) {
                       return &scored_moves_[current_move_idx_ - 1].move;
                    }
                    continue; // Skip if it was already tried as PV or Killer
                }
                stage_ = BAD_CAPTURES;
                current_move_idx_ = good_captures_end_idx_; // Start from where bad captures begin
                break;

            case BAD_CAPTURES:
                if (current_move_idx_ < num_scored_moves_) {
                     Move& move = scored_moves_[current_move_idx_++].move;
                     if (pv_move_.has_value() && move == *pv_move_) continue;
                     return &scored_moves_[current_move_idx_ - 1].move;
                }
                stage_ = DONE;
                break;

            case DONE:
                return nullptr;
        }
    }
}


}  // namespace chess