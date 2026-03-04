// nnue.cc

// Warn if AVX2 has not been enabled by the compiler
#ifndef __AVX2__
#if defined(_MSC_VER)
#pragma message("Warning: AVX2 is not enabled by the compiler! Performance will be degraded. Compile with /arch:AVX2")
#else
#warning "AVX2 is not enabled by the compiler! Performance will be degraded. Compile with -mavx2 -mfma"
#endif
#endif

#include "nnue.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#define USE_AVX2 true 

namespace chess {

namespace { 
int ceil_div(int x, int y) {
  return x / y + (x % y != 0);
}

#if defined __AVX2__ && USE_AVX2
float sum8(__m256 x) {
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    const __m128 loQuad = _mm256_castps256_ps128(x);
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    const __m128 loDual = sumQuad;
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    const __m128 lo = sumDual;
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}
#endif

}  // namespace

NNUE::NNUE(std::string weights_dir,
           std::shared_ptr<NNUE> copy_weights_from) {
  num_layers_ = 4;
  layer_sizes_ = new int[4] { kNNUE_L0_Size, 32, 32, 1 };
  input_sizes_ = new int[4] { 14*14*7, 4*layer_sizes_[0], layer_sizes_[1], layer_sizes_[2] };

  kernel_ = new float**[num_layers_];
  bias_ = new float*[num_layers_];
  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    int in_size = input_sizes_[layer_id];
    int out_size = layer_sizes_[layer_id];
    kernel_[layer_id] = new float*[in_size];
    for (int id = 0; id < in_size; id++) {
      kernel_[layer_id][id] = new float[out_size];
    }
    bias_[layer_id] = new float[out_size];
  }

  if (copy_weights_from != nullptr) {
    CopyWeights(*copy_weights_from.get());
  } else {
    LoadWeightsFromFile(weights_dir);
  }
  CopyWeightsToAvxVectors();
}

void NNUE::CopyWeights(const NNUE& copy_from) {
  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    int in_size = input_sizes_[layer_id];
    int out_size = layer_sizes_[layer_id];
    for (int id = 0; id < in_size; id++) {
      std::memcpy(
          kernel_[layer_id][id],
          copy_from.kernel_[layer_id][id],
          out_size * sizeof(float));
    }

    std::memcpy(
        bias_[layer_id],
        copy_from.bias_[layer_id],
        out_size * sizeof(float));
  }
}

void NNUE::LoadWeightsFromFile(const std::string& weights_dir) {
  std::filesystem::path wpath(weights_dir);

  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    std::string kernel_filename = "layer_" + std::to_string(layer_id) + ".kernel";
    std::ifstream kernel_infile(wpath / kernel_filename);
    if (!kernel_infile.good()) {
      std::cerr << "Can't open kernel file: " << (wpath / kernel_filename).string() << std::endl;
      abort();
    }
    for (int input_id = 0; input_id < input_sizes_[layer_id]; input_id++) {
        for (int output_id = 0; output_id < layer_sizes_[layer_id]; output_id++) {
            if (!(kernel_infile >> kernel_[layer_id][input_id][output_id])) {
                std::cerr << "Error reading kernel value for layer " << layer_id 
                          << ", input_id " << input_id << ", output_id " << output_id << std::endl;
                abort();
            }
            if (kernel_infile.peek() == ',') kernel_infile.ignore(); 
        }
    }
    kernel_infile.close();

    std::string bias_filename = "layer_" + std::to_string(layer_id) + ".bias";
    std::ifstream bias_infile(wpath / bias_filename);
    if (!bias_infile.good()) {
      std::cerr << "Can't open bias file: " << (wpath / bias_filename).string() << std::endl;
      abort();
    }
    for (int output_id = 0; output_id < layer_sizes_[layer_id]; output_id++) {
      if (!(bias_infile >> bias_[layer_id][output_id])) {
          std::cerr << "Error reading bias value for layer " << layer_id << ", output_id " << output_id << std::endl;
          abort();
      }
      if (bias_infile.peek() == ',') bias_infile.ignore();  
    }
    bias_infile.close();

    // CRITICAL FIX: Add the "Empty Square" weights (Channel 0) into Layer 0's biases.
    // The Python model learns weights for empty squares because of one-hot encoding.
    if (layer_id == 0) {
      for (int sq = 0; sq < 196; sq++) {
        int empty_idx = sq * 7 + 0;
        for (int out_id = 0; out_id < layer_sizes_[0]; out_id++) {
          bias_[0][out_id] += kernel_[0][empty_idx][out_id];
        }
      }
    }
  }
}

void NNUE::CopyWeightsToAvxVectors() {
#if defined __AVX2__ && USE_AVX2
  avx2_kernel_rowwise_ = new __m256**[num_layers_];
  avx2_kernel_colwise_ = new __m256**[num_layers_];
  avx2_bias_ = new __m256*[num_layers_];
  
  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    int in_size = input_sizes_[layer_id];
    int out_size = layer_sizes_[layer_id];
    int avx2_in_size_ceil = ceil_div(in_size, 8); 
    int avx2_out_size_ceil = ceil_div(out_size, 8); 

    // FIX (Optimization): avx2_kernel_rowwise_ is only used by AddPiece/RemovePiece,
    // which strictly operate on layer 0. Skip allocation for all other layers.
    if (layer_id == 0) {
      avx2_kernel_rowwise_[layer_id] = new __m256*[in_size];
      for (int id = 0; id < in_size; id++) {
        avx2_kernel_rowwise_[layer_id][id] = (__m256*)_mm_malloc(avx2_out_size_ceil * sizeof(__m256), 32);
        for (int avx2_id = 0; avx2_id < avx2_out_size_ceil; avx2_id++) {
          // FIX (Critical): Use a zero-padded local buffer to avoid reading past the
          // end of the allocation when out_size is not a multiple of 8 (e.g. layer 3
          // has out_size=1, causing a 7-float over-read with a raw _mm256_loadu_ps).
          float padded[8] = {0.0f};
          for (int i = 0; i < 8; ++i) {
            if (8 * avx2_id + i < out_size) {
              padded[i] = kernel_[layer_id][id][8 * avx2_id + i];
            }
          }
          avx2_kernel_rowwise_[layer_id][id][avx2_id] = _mm256_loadu_ps(padded);
        }
      }
    } else {
      avx2_kernel_rowwise_[layer_id] = nullptr;
    }
    
    if (layer_id > 0) { 
      avx2_kernel_colwise_[layer_id] = new __m256*[out_size];
      for (int id = 0; id < out_size; id++) { 
        avx2_kernel_colwise_[layer_id][id] = (__m256*)_mm_malloc(avx2_in_size_ceil * sizeof(__m256), 32);
        for (int avx2_id = 0; avx2_id < avx2_in_size_ceil; avx2_id++) { 
          float colwise_values[8];
          for (int i = 0; i < 8; i++) { 
            if (8 * avx2_id + i < in_size) {
              colwise_values[i] = kernel_[layer_id][8 * avx2_id + i][id];
            } else {
              colwise_values[i] = 0.0f; 
            }
          }
          avx2_kernel_colwise_[layer_id][id][avx2_id] = _mm256_loadu_ps(colwise_values);
        }
      }
    } else { 
        avx2_kernel_colwise_[layer_id] = nullptr; 
    }

    // FIX (Critical): Use a zero-padded local buffer for avx2_bias_ for the same
    // reason as avx2_kernel_rowwise_ above -- layer 3 has out_size=1.
    avx2_bias_[layer_id] = (__m256*)_mm_malloc(avx2_out_size_ceil * sizeof(__m256), 32);
    for (int avx2_id = 0; avx2_id < avx2_out_size_ceil; avx2_id++) {
      float padded[8] = {0.0f};
      for (int i = 0; i < 8; ++i) {
        if (8 * avx2_id + i < out_size) {
          padded[i] = bias_[layer_id][8 * avx2_id + i];
        }
      }
      avx2_bias_[layer_id][avx2_id] = _mm256_loadu_ps(padded);
    }
  }
#endif
}

NNUE::~NNUE() {
#if defined __AVX2__ && USE_AVX2
  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    if (avx2_bias_[layer_id]) _mm_free(avx2_bias_[layer_id]);

    if (avx2_kernel_rowwise_[layer_id]) {
      // avx2_kernel_rowwise_ is only allocated for layer 0
      for (int id = 0; id < input_sizes_[layer_id]; id++) {
        _mm_free(avx2_kernel_rowwise_[layer_id][id]);
      }
      delete[] avx2_kernel_rowwise_[layer_id];
    }
    
    if (avx2_kernel_colwise_[layer_id]) {
        if (layer_id > 0) { 
            for (int id = 0; id < layer_sizes_[layer_id]; id++) {
                _mm_free(avx2_kernel_colwise_[layer_id][id]);
            }
            delete[] avx2_kernel_colwise_[layer_id];
        }
    }
  }
  delete[] avx2_bias_;
  delete[] avx2_kernel_rowwise_;
  delete[] avx2_kernel_colwise_;
#endif

  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    delete[] bias_[layer_id];
    
    if (kernel_[layer_id] != nullptr) {
      for (int id = 0; id < input_sizes_[layer_id]; id++) {
        delete[] kernel_[layer_id][id];
      }
      delete[] kernel_[layer_id];
    }
  }
  delete[] bias_;
  delete[] kernel_;
  delete[] layer_sizes_;
  delete[] input_sizes_;
}

void NNUE::InitializeAccumulator(Accumulator& acc, const std::vector<PlacedPiece>& placed_pieces) const {
#if defined __AVX2__ && USE_AVX2
  for (int player_id = 0; player_id < 4; player_id++) {
    __m256* acc_v = (__m256*)acc.values[player_id];
    for (int id = 0; id < ceil_div(layer_sizes_[0], 8); id++) {
      acc_v[id] = avx2_bias_[0][id];
    }
  }
#else
  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < layer_sizes_[0]; id++) {
      acc.values[player_id][id] = bias_[0][id];
    }
  }
#endif

  for (const auto& placed_piece : placed_pieces) {
    AddPiece(acc, placed_piece.GetPiece(), placed_piece.GetLocation());
  }
}

void NNUE::AddPiece(Accumulator& acc, Piece piece, BoardLocation location) const {
  int color = piece.GetColor();
  int piece_idx_for_kernel = 1 + (int)piece.GetPieceType(); 
  int sq_base = location.GetRow() * 14 * 7 + location.GetCol() * 7;
  int s = layer_sizes_[0];

#if defined __AVX2__ && USE_AVX2
  __m256* k_piece = avx2_kernel_rowwise_[0][sq_base + piece_idx_for_kernel];
  __m256* k_empty = avx2_kernel_rowwise_[0][sq_base + 0];
  __m256* acc_v = (__m256*)acc.values[color];
  
  for (int i = 0; i < ceil_div(s, 8); i++) {
    __m256 delta = _mm256_sub_ps(k_piece[i], k_empty[i]); // Add piece, remove empty
    acc_v[i] = _mm256_add_ps(acc_v[i], delta);
  }
#else
  float* k_piece = kernel_[0][sq_base + piece_idx_for_kernel];
  float* k_empty = kernel_[0][sq_base + 0];
  for (int i = 0; i < s; i++) {
    acc.values[color][i] += (k_piece[i] - k_empty[i]);
  }
#endif
}

void NNUE::RemovePiece(Accumulator& acc, Piece piece, BoardLocation location) const {
  int color = piece.GetColor();
  int piece_idx_for_kernel = 1 + (int)piece.GetPieceType();
  int sq_base = location.GetRow() * 14 * 7 + location.GetCol() * 7;
  int s = layer_sizes_[0]; 

#if defined __AVX2__ && USE_AVX2
  __m256* k_piece = avx2_kernel_rowwise_[0][sq_base + piece_idx_for_kernel];
  __m256* k_empty = avx2_kernel_rowwise_[0][sq_base + 0];
  __m256* acc_v = (__m256*)acc.values[color];

  for (int i = 0; i < ceil_div(s, 8); i++) {
    __m256 delta = _mm256_sub_ps(k_piece[i], k_empty[i]); // We subtract the same delta
    acc_v[i] = _mm256_sub_ps(acc_v[i], delta);
  }
#else
  float* k_piece = kernel_[0][sq_base + piece_idx_for_kernel];
  float* k_empty = kernel_[0][sq_base + 0];
  for (int i = 0; i < s; i++) {
    acc.values[color][i] -= (k_piece[i] - k_empty[i]);
  }
#endif
}

int32_t NNUE::Evaluate(PlayerColor turn, const Accumulator& acc) const {
  // We allocate layer buffers locally so that this function is 100% thread safe.
  alignas(32) float l0_activated[4][kNNUE_L0_Size];
  
  // Create buffers to ping-pong the layers. Size 32 handles layers 1, 2, 3 safely.
  constexpr int kMaxLayerNeurons = 32;
  alignas(32) float l_out[4][kMaxLayerNeurons]; 

#if defined __AVX2__ && USE_AVX2
  
  // 1. Layer 0 Activation (ReLU)
  for (int player_id = 0; player_id < 4; player_id++) {
    __m256* acc_v = (__m256*)acc.values[player_id];
    __m256* act_v = (__m256*)l0_activated[player_id];
    for (int id = 0; id < ceil_div(kNNUE_L0_Size, 8); id++) {
      act_v[id] = _mm256_max_ps(avx2_zero_, acc_v[id]);
    }
  }

  // 2. Layer 1 (The stitched 4-player view layer)
  int l1_output_size = layer_sizes_[1]; 
  for (int out_idx = 0; out_idx < l1_output_size; ++out_idx) {
    __m256 dot_product_sum_vec = avx2_zero_;
    
    for (int relative_view_idx = 0; relative_view_idx < 4; ++relative_view_idx) {
      PlayerColor actual_player_color_for_l0 = static_cast<PlayerColor>((turn + relative_view_idx) % 4);
      __m256* act_v = (__m256*)l0_activated[actual_player_color_for_l0];
      
      for (int l0_chunk_idx = 0; l0_chunk_idx < ceil_div(kNNUE_L0_Size, 8); ++l0_chunk_idx) {
        int kernel_colwise_flat_input_chunk_idx = relative_view_idx * ceil_div(kNNUE_L0_Size, 8) + l0_chunk_idx;
        dot_product_sum_vec = _mm256_add_ps(
            dot_product_sum_vec,
            _mm256_mul_ps(
                act_v[l0_chunk_idx], 
                avx2_kernel_colwise_[1][out_idx][kernel_colwise_flat_input_chunk_idx] 
            )
        );
      }
    }
    l_out[1][out_idx] = std::max(0.0f, sum8(dot_product_sum_vec) + bias_[1][out_idx]);
  }

  // 3. Layers 2 & 3 (Standard hidden layers)
  for (int layer_id = 2; layer_id < num_layers_; layer_id++) {
    int in_size = layer_sizes_[layer_id - 1]; 
    int avx2_in_size_ceil = ceil_div(in_size, 8);
    int out_size = layer_sizes_[layer_id];
    
    // Load the previous layer's output as AVX vectors
    __m256* prev_out_v = (__m256*)l_out[layer_id - 1];

    for (int out_id = 0; out_id < out_size; out_id++) { 
      __m256 result_sum_vector = avx2_zero_; 
      for (int in_chunk_id = 0; in_chunk_id < avx2_in_size_ceil; in_chunk_id++) {
        result_sum_vector = _mm256_add_ps(
            result_sum_vector,
            _mm256_mul_ps(prev_out_v[in_chunk_id], avx2_kernel_colwise_[layer_id][out_id][in_chunk_id])); 
      }
      float f = sum8(result_sum_vector) + bias_[layer_id][out_id]; 
      
      if (layer_id < num_layers_ - 1) { 
        f = std::max(0.0f, f);
      }
      l_out[layer_id][out_id] = f;
    }
  }

  float logit = l_out[num_layers_ - 1][0];

#else // Scalar (non-AVX2) path

  // 1. Layer 0 Activation
  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < kNNUE_L0_Size; id++) {
      l0_activated[player_id][id] = std::max(0.0f, acc.values[player_id][id]);
    }
  }

  // 2. Layer 1
  int l1_output_size = layer_sizes_[1];             
  std::vector<float> l1_input_buffer(4 * kNNUE_L0_Size); 

  for (int relative_view_idx = 0; relative_view_idx < 4; ++relative_view_idx) {
    PlayerColor actual_player_color_for_l0 = static_cast<PlayerColor>((turn + relative_view_idx) % 4);
    std::memcpy(l1_input_buffer.data() + (relative_view_idx * kNNUE_L0_Size), 
                l0_activated[actual_player_color_for_l0], 
                kNNUE_L0_Size * sizeof(float));
  }

  for (int l1_out_neuron_idx = 0; l1_out_neuron_idx < l1_output_size; ++l1_out_neuron_idx) {
    l_out[1][l1_out_neuron_idx] = 0;
    for (int flat_input_idx = 0; flat_input_idx < 4 * kNNUE_L0_Size; ++flat_input_idx) {
      l_out[1][l1_out_neuron_idx] += l1_input_buffer[flat_input_idx] * kernel_[1][flat_input_idx][l1_out_neuron_idx];
    }
    l_out[1][l1_out_neuron_idx] = std::max(0.0f, l_out[1][l1_out_neuron_idx] + bias_[1][l1_out_neuron_idx]);
  }

  // 3. Layers 2 & 3
  for (int layer_id = 2; layer_id < num_layers_; layer_id++) {
    int in_size = layer_sizes_[layer_id - 1]; 
    int out_size = layer_sizes_[layer_id]; 

    for (int out_id = 0; out_id < out_size; out_id++) { 
      l_out[layer_id][out_id] = 0; 
      for (int in_id = 0; in_id < in_size; in_id++) { 
        l_out[layer_id][out_id] += l_out[layer_id - 1][in_id] * kernel_[layer_id][in_id][out_id]; 
      }
      l_out[layer_id][out_id] += bias_[layer_id][out_id]; 
      
      if (layer_id < num_layers_ - 1) { 
        l_out[layer_id][out_id] = std::max(0.0f, l_out[layer_id][out_id]);
      }
    }
  }

  float logit = l_out[num_layers_ - 1][0];

#endif

  // Using the log-odds mathematically collapses the Sigmoid and inverse equations
  // d_factor = -10.0f / log(1.0f/.9f - 1.0f) = 10.0f / log(9) ≈ 4.5511961f
  // centipawns = 100.0f * d_factor * logit
  return static_cast<int32_t>(455.11961f * logit);
}

}  // namespace chess