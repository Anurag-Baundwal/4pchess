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

template<typename T, int D>
class StatsEntry {
    T entry;
   public:
    void operator=(const T& v) { entry = v; }
    T*   operator&() { return &entry; }
    T*   operator->() { return &entry; }
    operator const T&() const { return entry; }

    void operator<<(int bonus) {
        assert(abs(bonus) <= D);
        static_assert(D <= std::numeric_limits<T>::max(), "D overflows T");
        entry += std::min(D - entry, bonus);
        assert(abs(entry) <= D);
    }
};

template<typename T, int D, int Size, int... Sizes>
struct Stats: public std::array<Stats<T, D, Sizes...>, Size> {
    using stats = Stats<T, D, Size, Sizes...>;
    void fill(const T& v) {
        assert(std::is_standard_layout_v<stats>);
        using entry = StatsEntry<T, D>;
        entry* p = reinterpret_cast<entry*>(this);
        std::fill(p, p + sizeof(*this) / sizeof(entry), v);
    }
};

template<typename T, int D, int Size>
struct Stats<T, D, Size>: public std::array<StatsEntry<T, D>, Size> {};

enum StatsParams { NOT_USED = 0 };
enum StatsType { NoCaptures, Captures };

using PieceToHistory = Stats<int32_t, 2147483647, 7, 14, 14>;
using ContinuationHistory = Stats<PieceToHistory, NOT_USED, 7, 14, 14>;

////////////////////////////////////////////////////////////////////////////////

class MovePicker {
 public:
  MovePicker(
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
    bool include_quiets = true,
    const PieceToHistory** piece_to_history = nullptr
    );

  Move* GetNextMove();
  int GetNumMoves() const { return num_moves_; };

 private:
  struct Item {
    unsigned short index;
    float score;
    Item() = default;
    Item(short idx, float sco) : index(idx), score(sco) { }
  };

  // Fixed size buffers to avoid heap allocation
  static constexpr int kMaxStageMoves = 256;
  
  struct StageBuffer {
      Item items[kMaxStageMoves];
      int count = 0;
  };

  Board* board_ = nullptr;
  ExtMove* moves_ = nullptr; 
  size_t num_moves_ = 0;
  
  uint8_t stage_ = 0;
  uint8_t stage_idx_ = 0;
  
  // Replaces std::vector<std::vector<Item>>
  StageBuffer stages_[5];
  
  bool init_stages_[5] = {false, false, false, false, false};
  bool enable_move_order_checks_;
};

}  // namespace chess

#endif  // _MOVE_PICKER_H_