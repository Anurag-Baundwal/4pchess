#ifndef _TYPES_H_
#define _TYPES_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "board.h" // Needed for Move type

namespace chess {

////////////////////////////////////////////////////////////////////////////////
/// Shared Data Structures
////////////////////////////////////////////////////////////////////////////////

// A struct to hold a killer move and its associated score for aging.
struct KillerEntry {
  Move move = Move();
  int score = 0;
};

// --- Stats templates and related definitions (moved from move_picker.h) ---

// StatsEntry stores the stat table value.
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

// A generic N-dimensional array for statistics.
template<typename T, int D, int Size, int... Sizes>
struct Stats: public std::array<Stats<T, D, Sizes...>, Size> {
    using stats = Stats<T, D, Size, Sizes...>;
    void fill(const T& v) {
        assert(std::is_standard_layout_v<stats>);
        using entry = StatsEntry<T, D>;
        entry* p    = reinterpret_cast<entry*>(this);
        std::fill(p, p + sizeof(*this) / sizeof(entry), v);
    }
};

template<typename T, int D, int Size>
struct Stats<T, D, Size>: public std::array<StatsEntry<T, D>, Size> {};

enum StatsParams { NOT_USED = 0 };
enum StatsType { NoCaptures, Captures };

// Addressed by [piece][to]
using PieceToHistory = Stats<int32_t, 2147483647, 7, 14, 14>;

// Addressed by [piece_1][to_1][piece_2][to_2]
using ContinuationHistory = Stats<PieceToHistory, NOT_USED, 7, 14, 14>;

}  // namespace chess

#endif // _TYPES_H_