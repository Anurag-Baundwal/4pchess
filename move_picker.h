#ifndef _MOVE_PICKER_H_
#define _MOVE_PICKER_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "board.h"


namespace chess {

////////////////////////////////////////////////////////////////////////////////
/// Stats
////////////////////////////////////////////////////////////////////////////////

// StatsEntry stores the stat table value. It is usually a number but could
// be a move or even a nested history. We use a class instead of a naked value
// to directly call history update operator<<() on the entry so to use stats
// tables at caller sites as simple multi-dim arrays.
template<typename T, int D>
class StatsEntry {

    T entry;

   public:
    void operator=(const T& v) { entry = v; }
    T*   operator&() { return &entry; }
    T*   operator->() { return &entry; }
    operator const T&() const { return entry; }

    void operator<<(int bonus) {
        assert(abs(bonus) <= D);  // Ensure range is [-D, D]
        static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");
        entry += std::min(D - entry, bonus);
        assert(abs(entry) <= D);
    }
};

// Stats is a generic N-dimensional array used to store various statistics.
// The first template parameter T is the base type of the array, and the second
// template parameter D limits the range of updates in [-D, D] when we update
// values with the << operator, while the last parameters (Size and Sizes)
// encode the dimensions of the array.
template<typename T, int D, int Size, int... Sizes>
struct Stats: public std::array<Stats<T, D, Sizes...>, Size> {
    using stats = Stats<T, D, Size, Sizes...>;

    void fill(const T& v) {

        // For standard-layout 'this' points to the first struct member
        assert(std::is_standard_layout_v<stats>);

        using entry = StatsEntry<T, D>;
        entry* p    = reinterpret_cast<entry*>(this);
        std::fill(p, p + sizeof(*this) / sizeof(entry), v);
    }
};

template<typename T, int D, int Size>
struct Stats<T, D, Size>: public std::array<StatsEntry<T, D>, Size> {};

// In stats table, D=0 means that the template parameter is not used
enum StatsParams {
    NOT_USED = 0
};
enum StatsType {
    NoCaptures,
    Captures
};

// Addressed by [piece][to]
using PieceToHistory = Stats<int32_t, 2147483647, 7, 14, 14>;

// Addressed by [piece_1][to_1][piece_2][to_2]
using ContinuationHistory = Stats<PieceToHistory, NOT_USED, 7, 14, 14>;


////////////////////////////////////////////////////////////////////////////////

class MovePicker {
 public:
  // CORRECTED CONSTRUCTOR
  MovePicker(
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
    );

  // If this returns nullptr then there are no more moves
  Move* GetNextMove();
  int GetNumMoves() const { return num_moves_; };

 private:
  enum Stage {
    PV_MOVE,
    GENERATE_CAPTURES,
    GOOD_CAPTURES,
    KILLERS,
    GENERATE_QUIETS,
    QUIET_MOVES,
    BAD_CAPTURES,
    DONE
  };

  struct MoveScore {
      Move move;
      int score;
  };
  
  void score_captures();
  void score_quiets();

  Board& board_;
  std::optional<Move> pv_move_; // Store a copy
  Move* killers_;
  const int* piece_evals_;
  int (*history_heuristic_)[14][14][14][14];
  int (*capture_heuristic_)[4][6][4][14][14];
  const int* piece_scores_;
  Move* counter_moves_;
  const PieceToHistory** piece_to_history_;


  Stage stage_ = PV_MOVE;
  int num_moves_ = 0; // Moves in current stage's buffer
  bool include_quiets_;
  bool move_order_checks_;

  static constexpr int kMaxMoves = 256;
  Move move_buffer_[kMaxMoves];
  MoveScore scored_moves_[kMaxMoves];
  int current_move_idx_ = 0;
  int num_scored_moves_ = 0;
  int good_captures_end_idx_ = 0; // New member to track end of good captures
  int killer_idx_ = 0;
};

}  // namespace chess

#endif  // _MOVE_PICKER_H_