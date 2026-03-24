#include <cassert>
#include <optional>
#include <iostream>

#include "transposition_table.h"

namespace chess {

TranspositionTable::TranspositionTable(size_t table_size) {
  assert((table_size > 0) && "transposition table_size = 0");
  table_size_ = table_size;
  hash_table_ = (HashTableEntry*) calloc(table_size, sizeof(HashTableEntry));
  assert(
      (hash_table_ != nullptr) && 
      "Can't create transposition table. Try using a smaller size.");
  a_mutexes_ = std::make_unique<std::mutex[]>(kNumMutexes);
  // Initialize all eval fields to the 'none' value
  for (size_t i = 0; i < table_size_; ++i) {
    hash_table_[i].eval = value_none_tt;
    hash_table_[i].age = 0; // Initialize age
  }
}


const HashTableEntry* TranspositionTable::Get(int64_t key) {
  size_t n = key % table_size_;
  std::lock_guard<std::mutex> lock(a_mutexes_[n % kNumMutexes]);
  HashTableEntry* entry = hash_table_ + n;
  if (entry->key == key) {
    return entry;
  }
  return nullptr;
}

void TranspositionTable::Save(
    int64_t key, int depth, std::optional<Move> move, int score, int eval,
    ScoreBound bound, bool is_pv) {
  size_t n = key % table_size_;
  std::lock_guard<std::mutex> lock(a_mutexes_[n % kNumMutexes]);
  HashTableEntry& entry = hash_table_[n];

  bool replace = false;

  if (entry.key != key) {
    // 1. HASH COLLISION (Different position mapping to the same index)
    // Replace if the existing entry is from an older search (stale), 
    // OR if the current search is deeper.
    if (entry.age != generation_ || depth >= entry.depth) {
      replace = true;
    }
  } else {
    // 2. SAME POSITION
    // Replace if the new search is at least as deep, or gives an EXACT bound.
    if (depth >= entry.depth || bound == EXACT) {
      replace = true;
    }
    // Very important: Refresh the age! This prevents our deep, valuable 
    // entries from being evicted by random hash collisions during this turn.
    entry.age = generation_;
  }

  if (replace) {
    entry.key = key;
    entry.depth = depth;
    if (move.has_value()) {
      entry.move = *move;
    } else {
      entry.move = Move();
    }
    entry.score = score;
    entry.eval = eval;
    entry.bound = bound;
    entry.is_pv = is_pv;
    entry.age = generation_;
  }
}


}  // namespace chess