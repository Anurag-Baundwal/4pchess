#ifndef __NNUE_H__
#define __NNUE_H__

#include <immintrin.h>
#include <memory>
#include <vector>

#include "../types.h"

namespace chess {

// Layer 0 feature size (must match layer_sizes_[0] in nnue.cc)
constexpr int kNNUE_L0_Size = 32;

// The Accumulator holds the state of Layer 0 for all 4 players.
// In Option B, this lives inside the Stack struct in player.h, not in the NNUE class.
struct alignas(32) Accumulator {
  float values[4][kNNUE_L0_Size];
};

class NNUE {
 public:
  NNUE(std::string weights_dir,
       std::shared_ptr<NNUE> copy_weights_from = nullptr);
  ~NNUE();

  // Initializes an accumulator from scratch (called at the root of the search)
  void InitializeAccumulator(Accumulator& acc, const std::vector<PlacedPiece>& placed_pieces) const;
  
  // Fast incremental updates (called during MakeMove)
  void AddPiece(Accumulator& acc, Piece piece, BoardLocation location) const;
  void RemovePiece(Accumulator& acc, Piece piece, BoardLocation location) const;
  
  // Outputs the eval score in centipawns. Now fully stateless and thread-safe!
  int32_t Evaluate(PlayerColor turn, const Accumulator& acc) const;

 private:
  void CopyWeights(const NNUE& copy_from);
  void LoadWeightsFromFile(const std::string& weights_dir);
  void CopyWeightsToAvxVectors();

  int num_layers_ = 0;
  int* layer_sizes_ = nullptr;
  int* input_sizes_ = nullptr;

  // note that the first kernel input size is 4*layer_sizes[0]
  float*** kernel_ = nullptr; // [num_layers][layer_sizes[i-1]][layer_sizes[i]]
  float** bias_ = nullptr;    // [num_layers][layer_sizes[i]]

#ifdef __AVX2__

  // __m256 == 8 floats
  __m256*** avx2_kernel_rowwise_ = nullptr; // [num_layers][layer_sizes[i-1]][layer_sizes[i]]
  __m256*** avx2_kernel_colwise_ = nullptr; // [num_layers][layer_sizes[i]][layer_sizes[i-1]]
  __m256** avx2_bias_ = nullptr;            // [num_layers][layer_sizes[i]]
  __m256 avx2_zero_ = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

#endif

};

}  // namespace chess

#endif // __NNUE_H__