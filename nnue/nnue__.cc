// nnue.cc
// this one has debug prints for internal states
#include "nnue.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>   // For std::cout, std::endl, std::cerr
#include <sstream>
#include <string>
#include <vector>     // For std::vector
#include <iomanip>    // For std::fixed, std::setprecision
#include <algorithm>  // For std::min, std::max

// Keep USE_AVX2 defined true if you intend to use it and compile with AVX2 flags.
// If you don't compile with AVX2 flags, this block won't be used anyway.
#define USE_AVX2 true 

namespace chess {

namespace { // Anonymous namespace for test-specific helpers (from nnue_test.cc)

// Helper to print a few elements of a float array/vector
void PrintFloatArraySample(const std::string& name, const float* arr, int total_size, int sample_size = 8) {
    std::cout << "  " << name << " (first " << sample_size << " / last " << sample_size << " of " << total_size << "):" << std::endl;
    if (total_size == 0) {
        std::cout << "    (empty)" << std::endl;
        return;
    }
    std::cout << "    ";
    for (int i = 0; i < std::min(sample_size, total_size); ++i) {
        std::cout << std::fixed << std::setprecision(6) << arr[i] << " ";
    }
    if (total_size > sample_size) {
        if (total_size > 2 * sample_size) std::cout << "... ";
        for (int i = std::max(sample_size, total_size - sample_size); i < total_size; ++i) {
            std::cout << std::fixed << std::setprecision(6) << arr[i] << " ";
        }
    }
    std::cout << std::endl;
}

#if defined __AVX2__ && USE_AVX2 // Only compile if AVX2 is enabled in your build configuration
// Helper to print a few elements of an __m256 array
void PrintAVXArraySample(const std::string& name, const __m256* arr_m256, int num_m256_vectors, int total_float_size, int sample_m256_vectors = 1) {
    std::cout << "  " << name << " AVX (first " << sample_m256_vectors << " vec / last " << sample_m256_vectors << " vec of " << num_m256_vectors << " vecs; total floats " << total_float_size << "):" << std::endl;
    if (num_m256_vectors == 0) {
        std::cout << "    (empty)" << std::endl;
        return;
    }
    
    std::vector<float> temp_floats(8); // Temporary buffer to store AVX vector contents

    std::cout << "    ";
    for (int i = 0; i < std::min(sample_m256_vectors, num_m256_vectors); ++i) {
        _mm256_storeu_ps(temp_floats.data(), arr_m256[i]);
        for(int j=0; j<8; ++j) std::cout << std::fixed << std::setprecision(6) << temp_floats[j] << " ";
        if (i < std::min(sample_m256_vectors, num_m256_vectors) - 1) std::cout << "| ";
    }
    
    if (num_m256_vectors > sample_m256_vectors) {
        if (num_m256_vectors > 2 * sample_m256_vectors) std::cout << "... ";
        for (int i = std::max(sample_m256_vectors, num_m256_vectors - sample_m256_vectors); i < num_m256_vectors; ++i) {
             _mm256_storeu_ps(temp_floats.data(), arr_m256[i]);
            for(int j=0; j<8; ++j) std::cout << std::fixed << std::setprecision(6) << temp_floats[j] << " ";
            if (i < num_m256_vectors -1) std::cout << "| ";
        }
    }
    std::cout << std::endl;
}
#endif // __AVX2__


int ceil_div(int x, int y) {
  return x / y + (x % y != 0);
}

}  // namespace

NNUE::NNUE(std::string weights_dir,
           std::shared_ptr<NNUE> copy_weights_from) {
  num_layers_ = 4;
  layer_sizes_ = new int[4] { 32, 32, 32, 1 };
  input_sizes_ = new int[4] { 14*14*7, 4*layer_sizes_[0], layer_sizes_[1], layer_sizes_[2] };

  linear_output_0_ = new float*[4];
  l0_output_ = new float*[4];
  for (int player_id = 0; player_id < 4; player_id++) {
    linear_output_0_[player_id] = new float[layer_sizes_[0]];
    l0_output_[player_id] = new float[layer_sizes_[0]];
  }
  l_output_ = new float*[num_layers_];
  l_output_[0] = nullptr; // Layer 0 output is handled by l0_output_
  for (int layer_id = 1; layer_id < num_layers_; layer_id++) {
    l_output_[layer_id] = new float[layer_sizes_[layer_id]];
  }
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
  // load weights from files
  std::filesystem::path wpath(weights_dir);
  std::string line;

  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    // kernel
    std::string kernel_filename = "layer_" + std::to_string(layer_id) + ".kernel";
    std::ifstream kernel_infile(wpath / kernel_filename);
    if (!kernel_infile.good()) {
      std::cerr << "Can't open kernel file: " << (wpath / kernel_filename).string() << std::endl;
      abort();
    }
    for (int input_id = 0; input_id < input_sizes_[layer_id]; input_id++) {
        if (!std::getline(kernel_infile, line)) {
            std::cerr << "Error reading kernel line for layer " << layer_id << ", input_id " << input_id << std::endl;
            abort();
        }
        std::istringstream ss(line);
        for (int output_id = 0; output_id < layer_sizes_[layer_id]; output_id++) {
            ss >> kernel_[layer_id][input_id][output_id];
            if (ss.peek() == ',') {
                ss.ignore(); // ignore commas
            }
        }
    }
    kernel_infile.close();

    // bias
    std::string bias_filename = "layer_" + std::to_string(layer_id) + ".bias";
    std::ifstream bias_infile(wpath / bias_filename);
    if (!bias_infile.good()) {
      std::cerr << "Can't open bias file: " << (wpath / bias_filename).string() << std::endl;
      abort();
    }
    if (!std::getline(bias_infile, line)) { // Bias is typically one line
        std::cerr << "Error reading bias line for layer " << layer_id << std::endl;
        abort();
    }
    std::istringstream ss(line);
    for (int output_id = 0; output_id < layer_sizes_[layer_id]; output_id++) {
      ss >> bias_[layer_id][output_id];
      if (ss.peek() == ',') {
        ss.ignore();  // ignore commas
      }
    }
    bias_infile.close();
  }
}

void NNUE::CopyWeightsToAvxVectors() {

#if defined __AVX2__ && USE_AVX2
  // Initialize AVX2 arrays for linear outputs and activation outputs
  avx2_linear_output_0_ = new __m256*[4];
  avx2_l0_output_ = new __m256*[4];
  for (int player_id = 0; player_id < 4; player_id++) {
    avx2_linear_output_0_[player_id] = new __m256[ceil_div(layer_sizes_[0], 8)];
    avx2_l0_output_[player_id] = new __m256[ceil_div(layer_sizes_[0], 8)];
  }
  avx2_l_output_ = new __m256*[num_layers_];
  avx2_l_output_[0] = nullptr; // Layer 0 output is handled by avx2_l0_output_
  for (int layer_id = 1; layer_id < num_layers_; layer_id++) {
    avx2_l_output_[layer_id] = new __m256[ceil_div(layer_sizes_[layer_id], 8)];
  }

  // Initialize AVX2 arrays for kernels and biases
  avx2_kernel_rowwise_ = new __m256**[num_layers_];
  avx2_kernel_colwise_ = new __m256**[num_layers_];
  avx2_bias_ = new __m256*[num_layers_];
  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    int in_size = input_sizes_[layer_id];
    int out_size = layer_sizes_[layer_id];
    int avx2_in_size_ceil = ceil_div(in_size, 8); // Number of __m256 vectors for input dim
    int avx2_out_size_ceil = ceil_div(out_size, 8); // Number of __m256 vectors for output dim

    // Row-wise kernel storage (for input-sparse layers like L0)
    avx2_kernel_rowwise_[layer_id] = new __m256*[in_size];
    for (int id = 0; id < in_size; id++) {
      avx2_kernel_rowwise_[layer_id][id] = new __m256[avx2_out_size_ceil];
      // Load weights into row-wise vectors
      for (int avx2_id = 0; avx2_id < avx2_out_size_ceil; avx2_id++) {
        // Use _mm256_loadu_ps for unaligned loads if necessary, or ensure alignment.
        // It's safer to use _mm256_loadu_ps for `float*` to `__m256` conversions.
        avx2_kernel_rowwise_[layer_id][id][avx2_id] = _mm256_loadu_ps(&kernel_[layer_id][id][8 * avx2_id]);
      }
    }
    
    // Column-wise kernel storage (for efficient dense layer multiplication)
    // Layer 0 is special (sparse input), so we don't use colwise for it.
    if (layer_id > 0) { 
      avx2_kernel_colwise_[layer_id] = new __m256*[out_size];
      for (int id = 0; id < out_size; id++) { // Iterates through output neurons
        avx2_kernel_colwise_[layer_id][id] = new __m256[avx2_in_size_ceil];
        // Load weights by columns
        for (int avx2_id = 0; avx2_id < avx2_in_size_ceil; avx2_id++) { // Iterate through input chunks
          float colwise_values[8];
          for (int i = 0; i < 8; i++) { // Fill 8 floats for the __m256 vector
            // Transpose: kernel_[layer_id][input_neuron][output_neuron]
            // We are collecting values for `input_id` from `8 * avx2_id` to `8 * avx2_id + 7`
            // and `output_id` is fixed as `id`.
            // Ensure 8*avx2_id + i is within bounds of input_size
            if (8 * avx2_id + i < in_size) {
              colwise_values[i] = kernel_[layer_id][8 * avx2_id + i][id];
            } else {
              colwise_values[i] = 0.0f; // Pad with zeros if input_size is not a multiple of 8
            }
          }
          avx2_kernel_colwise_[layer_id][id][avx2_id] = _mm256_loadu_ps(colwise_values);
        }
      }
    } else { 
        avx2_kernel_colwise_[layer_id] = nullptr; // Explicitly set to nullptr for layer 0
    }

    // Bias vectors
    avx2_bias_[layer_id] = new __m256[avx2_out_size_ceil];
    for (int avx2_id = 0; avx2_id < avx2_out_size_ceil; avx2_id++) {
      avx2_bias_[layer_id][avx2_id] = _mm256_loadu_ps(&bias_[layer_id][8 * avx2_id]);
    }
  }

#endif
}

NNUE::~NNUE() {
#if defined __AVX2__ && USE_AVX2
  for (int player_id = 0; player_id < 4; player_id++) {
    delete[] avx2_linear_output_0_[player_id];
    delete[] avx2_l0_output_[player_id];
  }
  delete[] avx2_linear_output_0_;
  delete[] avx2_l0_output_;

  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    delete[] avx2_bias_[layer_id];
    if (avx2_l_output_[layer_id] != nullptr) {
      delete[] avx2_l_output_[layer_id];
    }
    if (avx2_kernel_rowwise_[layer_id] != nullptr) {
      int in_size = input_sizes_[layer_id];
      for (int id = 0; id < in_size; id++) {
        delete[] avx2_kernel_rowwise_[layer_id][id];
      }
      delete[] avx2_kernel_rowwise_[layer_id];
    }
    if (avx2_kernel_colwise_[layer_id] != nullptr) {
        if (layer_id > 0) { // Only delete if actually allocated
            for (int id = 0; id < layer_sizes_[layer_id]; id++) {
                delete[] avx2_kernel_colwise_[layer_id][id];
            }
            delete[] avx2_kernel_colwise_[layer_id];
        }
    }
  }
  delete[] avx2_bias_;
  delete[] avx2_l_output_;
  delete[] avx2_kernel_rowwise_;
  delete[] avx2_kernel_colwise_;
#endif

  for (int player_id = 0; player_id < 4; player_id++) {
    delete[] linear_output_0_[player_id];
    delete[] l0_output_[player_id];
  }
  delete[] linear_output_0_;
  delete[] l0_output_;

  for (int layer_id = 0; layer_id < num_layers_; layer_id++) {
    delete[] bias_[layer_id];
    if (l_output_[layer_id] != nullptr) {
      delete[] l_output_[layer_id];
    }
    if (kernel_[layer_id] != nullptr) {
      int in_size = input_sizes_[layer_id];
      for (int id = 0; id < in_size; id++) {
        delete[] kernel_[layer_id][id];
      }
      delete[] kernel_[layer_id];
    }
  }
  delete[] bias_;
  delete[] l_output_;
  delete[] kernel_;
  delete[] layer_sizes_;
  delete[] input_sizes_;

}

void NNUE::InitializeWeights(const std::vector<PlacedPiece>& placed_pieces) {
// This function needs to initialize `linear_output_0_` to the biases first,
// then add contributions from all pieces on the board.

#if defined __AVX2__ && USE_AVX2
  // 1. Initialize layer-0 linear outputs with biases
  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < ceil_div(layer_sizes_[0], 8); id++) {
      avx2_linear_output_0_[player_id][id] = avx2_bias_[0][id];
    }
  }
  std::cout << "[DEBUG NNUE Init] AVX2 linear_output_0_[RED] (after bias, before pieces):" << std::endl;
  PrintAVXArraySample("  Bias0_RED", avx2_linear_output_0_[RED], ceil_div(layer_sizes_[0], 8), layer_sizes_[0]);

#else
  // 1. Initialize layer-0 linear outputs with biases
  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < layer_sizes_[0]; id++) {
      linear_output_0_[player_id][id] = bias_[0][id];
    }
  }
  std::cout << "[DEBUG NNUE Init] Scalar linear_output_0_[RED] (after bias, before pieces):" << std::endl;
  PrintFloatArraySample("  Bias0_RED", linear_output_0_[RED], layer_sizes_[0]);
#endif

  // 2. Call SetPiece for each piece to add its contribution.
  // SetPiece correctly adds to the bias-initialized linear_output_0_
  for (const auto& placed_piece : placed_pieces) {
    SetPiece(placed_piece.GetPiece(), placed_piece.GetLocation());
  }

#if defined __AVX2__ && USE_AVX2
  std::cout << "[DEBUG NNUE Init] AVX2 linear_output_0_[RED] (after all pieces):" << std::endl;
  PrintAVXArraySample("  Accum0_RED", avx2_linear_output_0_[RED], ceil_div(layer_sizes_[0], 8), layer_sizes_[0]);
#else
  std::cout << "[DEBUG NNUE Init] Scalar linear_output_0_[RED] (after all pieces):" << std::endl;
  PrintFloatArraySample("  Accum0_RED", linear_output_0_[RED], layer_sizes_[0]);
#endif
}

void NNUE::SetPiece(Piece piece, BoardLocation location) {
  int color = piece.GetColor();
  // piece_type here refers to the one-hot index used in the NNUE input features.
  // gen_data.cc encodes empty/opponent as 0, and player pieces as 1 + actual PieceType.
  // So, PieceType PAWN (0) becomes index 1, KNIGHT (1) becomes index 2, etc.
  int piece_idx_for_kernel = 1 + (int)piece.GetPieceType(); 
  int row = location.GetRow();
  int col = location.GetCol();
  int s = layer_sizes_[0]; // Output size of Layer 0

#if defined __AVX2__ && USE_AVX2
  // kernel_[0] mapping: [row*14*7 + col*7 + piece_idx_for_kernel][output_feature_id]
  __m256* k_row = avx2_kernel_rowwise_[0][row*14*7 + col*7 + piece_idx_for_kernel];
  for (int i = 0; i < ceil_div(s, 8); i++) {
    avx2_linear_output_0_[color][i] =
      _mm256_add_ps(avx2_linear_output_0_[color][i], k_row[i]);
  }

#else
  // kernel_[0] mapping: [row*14*7 + col*7 + piece_idx_for_kernel][output_feature_id]
  float* k_row = kernel_[0][row*14*7 + col*7 + piece_idx_for_kernel];
  for (int i = 0; i < s; i++) {
    linear_output_0_[color][i] += k_row[i];
  }

#endif
}

void NNUE::RemovePiece(Piece piece, BoardLocation location) {
  int color = piece.GetColor();
  // Same indexing convention as SetPiece
  int piece_idx_for_kernel = 1 + (int)piece.GetPieceType();
  int row = location.GetRow();
  int col = location.GetCol();
  int s = layer_sizes_[0]; // Output size of Layer 0

#if defined __AVX2__ && USE_AVX2
  // kernel_[0] mapping: [row*14*7 + col*7 + piece_idx_for_kernel][output_feature_id]
  __m256* k_row = avx2_kernel_rowwise_[0][row*14*7 + col*7 + piece_idx_for_kernel];
  for (int i = 0; i < ceil_div(s, 8); i++) {
    avx2_linear_output_0_[color][i] =
      _mm256_sub_ps(avx2_linear_output_0_[color][i], k_row[i]);
  }

#else
  // kernel_[0] mapping: [row*14*7 + col*7 + piece_idx_for_kernel][output_feature_id]
  float* k_row = kernel_[0][row*14*7 + col*7 + piece_idx_for_kernel];
  for (int i = 0; i < s; i++) {
    linear_output_0_[color][i] -= k_row[i];
  }

#endif
}

void NNUE::ComputeLayer0Activation() {
#if defined __AVX2__ && USE_AVX2

  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < ceil_div(layer_sizes_[0], 8); id++) {
      // ReLU activation: max(0, x)
      avx2_l0_output_[player_id][id] =
        _mm256_max_ps(avx2_zero_, avx2_linear_output_0_[player_id][id]);
    }
    // Debug print for RED player's L0 output
    if (player_id == RED) { 
        std::cout << "[DEBUG NNUE L0Act] AVX2 l0_output_[RED]:" << std::endl;
        PrintAVXArraySample("  L0_Act_RED", avx2_l0_output_[RED], ceil_div(layer_sizes_[0], 8), layer_sizes_[0]);
    }
  }

#else

  for (int player_id = 0; player_id < 4; player_id++) {
    for (int id = 0; id < layer_sizes_[0]; id++) {
      // ReLU activation: max(0, x)
      l0_output_[player_id][id] = std::max(0.0f, linear_output_0_[player_id][id]);
    }
    // Debug print for RED player's L0 output
    if (player_id == RED) {
        std::cout << "[DEBUG NNUE L0Act] Scalar l0_output_[RED]:" << std::endl;
        PrintFloatArraySample("  L0_Act_RED", l0_output_[RED], layer_sizes_[0]);
    }
  }

#endif
}

namespace {

// Horizontal sum for __m256 vector
// x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

}  // namespace

int32_t NNUE::Evaluate(PlayerColor turn) {
  std::cout << "[DEBUG NNUE Eval] Called Evaluate for turn: " << (int)turn << std::endl;
  // First, compute Layer 0 activations (ReLU applied to linear_output_0_)
  ComputeLayer0Activation();

#if defined __AVX2__ && USE_AVX2

  int l0_feat_size = layer_sizes_[0]; // e.g., 32
  int l0_avx2_chunks_per_player = ceil_div(l0_feat_size, 8);
  int l1_output_size = layer_sizes_[1]; // e.g., 32

  std::vector<float> l1_results_temp(l1_output_size); 
  for (int out_idx = 0; out_idx < l1_output_size; ++out_idx) {
    __m256 dot_product_sum_vec = avx2_zero_;
    
    // Loop through the 4 relative player views (0: current, 1: next, 2: partner, 3: prev)
    for (int relative_view_idx = 0; relative_view_idx < 4; ++relative_view_idx) {
      PlayerColor actual_player_color_for_l0 = static_cast<PlayerColor>((turn + relative_view_idx) % 4);
      
      // Loop through AVX chunks of features for this player's L0 output
      for (int l0_chunk_idx = 0; l0_chunk_idx < l0_avx2_chunks_per_player; ++l0_chunk_idx) {
        // Kernel chunk index corresponding to this relative view and feature chunk
        int kernel_colwise_flat_input_chunk_idx = relative_view_idx * l0_avx2_chunks_per_player + l0_chunk_idx;
        
        dot_product_sum_vec = _mm256_add_ps(
            dot_product_sum_vec,
            _mm256_mul_ps(
                avx2_l0_output_[actual_player_color_for_l0][l0_chunk_idx], // L0 output for the correct relative player
                avx2_kernel_colwise_[1][out_idx][kernel_colwise_flat_input_chunk_idx] // Kernel segment for this relative view
            )
        );
      }
    }
    l1_results_temp[out_idx] = std::max(0.0f, sum8(dot_product_sum_vec) + bias_[1][out_idx]);
  }
  std::cout << "[DEBUG NNUE Eval] AVX2 Layer 1 Output (after ReLU, before AVX store):" << std::endl;
  PrintFloatArraySample("  L1_Out_ScalarTemp", l1_results_temp.data(), l1_output_size);

  // Store Layer 1 results into avx2_l_output_[1]
  for (int out_idx_avx2 = 0; out_idx_avx2 < ceil_div(l1_output_size, 8); ++out_idx_avx2) {
    avx2_l_output_[1][out_idx_avx2] = _mm256_loadu_ps(l1_results_temp.data() + out_idx_avx2 * 8); 
  }

  // Compute further layers (Layer 2, Layer 3)
  // These layers operate on the output of the previous layer, which is already flattened.
  // The structure is standard dense layer calculation.
  for (int layer_id = 2; layer_id < num_layers_; layer_id++) {
    int in_size = layer_sizes_[layer_id - 1]; // Input size for current layer is output size of previous layer
    int avx2_in_size_ceil = ceil_div(in_size, 8);
    int out_size = layer_sizes_[layer_id];
    int avx2_out_size_ceil = ceil_div(out_size, 8);

    std::vector<float> l_results_current_layer(out_size); 
    for (int out_id = 0; out_id < out_size; out_id++) { // Iterate through each output neuron of current layer
      __m256 result_sum_vector = avx2_zero_; // Accumulator for the dot product
      for (int in_chunk_id = 0; in_chunk_id < avx2_in_size_ceil; in_chunk_id++) {
        result_sum_vector =
          _mm256_add_ps(
            result_sum_vector,
            _mm256_mul_ps(
              avx2_l_output_[layer_id - 1][in_chunk_id], // Output from previous layer
              avx2_kernel_colwise_[layer_id][out_id][in_chunk_id])); // Corresponding kernel weights
      }
      float f = sum8(result_sum_vector); // Sum the 8 floats
      f += bias_[layer_id][out_id]; // Add bias
      
      if (layer_id < num_layers_ - 1) { // Apply ReLU for hidden layers (not for the last output layer)
        f = std::max(0.0f, f);
      }
      l_results_current_layer[out_id] = f;
    }

    if (layer_id == 2) { 
        std::cout << "[DEBUG NNUE Eval] AVX2 Layer 2 Output (after ReLU, before AVX store):" << std::endl;
        PrintFloatArraySample("  L2_Out_ScalarTemp", l_results_current_layer.data(), out_size);
    }
    if (layer_id == 3) { // This is the final output layer (logit)
        std::cout << "[DEBUG NNUE Eval] AVX2 Layer 3 Output (Logit, before AVX store):" << std::endl;
        PrintFloatArraySample("  L3_Logit_ScalarTemp", l_results_current_layer.data(), out_size);
    }


    // Store current layer results into avx2_l_output_[layer_id]
    for (int out_id_avx2 = 0; out_id_avx2 < avx2_out_size_ceil; out_id_avx2++) {
      avx2_l_output_[layer_id][out_id_avx2] = _mm256_loadu_ps(l_results_current_layer.data() + out_id_avx2 * 8); 
    }
  }

  // Final output value from the network (this is the logit, pre-sigmoid)
  float logit = ((float*)avx2_l_output_[num_layers_ - 1])[0];
  std::cout << "[DEBUG NNUE Eval] AVX2 Final Logit from avx2_l_output_: " << std::fixed << std::setprecision(6) << logit << std::endl;

#else // Scalar (non-AVX2) path

  // Compute output of layer 1
  int l0_feature_size_per_player = layer_sizes_[0]; // e.g., 32
  int l1_output_size = layer_sizes_[1];             // e.g., 32

  // Temporary buffer to hold the correctly ordered L0 outputs for Layer 1 input
  // This buffer will store [L0_current, L0_next, L0_partner, L0_previous]
  std::vector<float> l1_input_buffer(4 * l0_feature_size_per_player); 

  // Populate l1_input_buffer based on 'turn'
  for (int relative_view_idx = 0; relative_view_idx < 4; ++relative_view_idx) {
    PlayerColor actual_player_color_for_l0 = static_cast<PlayerColor>((turn + relative_view_idx) % 4);
    std::memcpy(l1_input_buffer.data() + (relative_view_idx * l0_feature_size_per_player), 
                l0_output_[actual_player_color_for_l0], 
                l0_feature_size_per_player * sizeof(float));
  }
  std::cout << "[DEBUG NNUE Eval] Scalar l1_input_buffer (concatenated L0 outputs for RED, BLUE, YELLOW, GREEN if turn=RED):" << std::endl;
  PrintFloatArraySample("  L1_Input", l1_input_buffer.data(), 4 * l0_feature_size_per_player);


  // Now compute L1 output using the correctly ordered l1_input_buffer
  for (int l1_out_neuron_idx = 0; l1_out_neuron_idx < l1_output_size; ++l1_out_neuron_idx) {
    l_output_[1][l1_out_neuron_idx] = 0;
    // The kernel_[1] is indexed by the flat combined input feature index
    for (int flat_input_idx = 0; flat_input_idx < 4 * l0_feature_size_per_player; ++flat_input_idx) {
      l_output_[1][l1_out_neuron_idx] += l1_input_buffer[flat_input_idx] 
                                     * kernel_[1][flat_input_idx][l1_out_neuron_idx];
    }
    l_output_[1][l1_out_neuron_idx] = std::max(0.0f, l_output_[1][l1_out_neuron_idx] + bias_[1][l1_out_neuron_idx]);
  }
  std::cout << "[DEBUG NNUE Eval] Scalar Layer 1 Output (l_output_[1]):" << std::endl;
  PrintFloatArraySample("  L1_Out", l_output_[1], layer_sizes_[1]);


  // Compute further layers (Layer 2, Layer 3)
  // These layers operate on the output of the previous layer, which is already flattened.
  // The structure is standard dense layer calculation.
  for (int layer_id = 2; layer_id < num_layers_; layer_id++) {
    int in_size = layer_sizes_[layer_id - 1]; // Input size for current layer is output size of previous layer
    int out_size = layer_sizes_[layer_id]; // Output size of current layer

    for (int out_id = 0; out_id < out_size; out_id++) { // Iterate through each output neuron of current layer
      l_output_[layer_id][out_id] = 0; // Initialize sum for this output neuron
      for (int in_id = 0; in_id < in_size; in_id++) { // Loop through all input features from previous layer
        l_output_[layer_id][out_id] += l_output_[layer_id - 1][in_id] // Input feature from previous layer
          * kernel_[layer_id][in_id][out_id]; // Corresponding kernel weight
      }
      l_output_[layer_id][out_id] += bias_[layer_id][out_id]; // Add bias
      
      if (layer_id < num_layers_ - 1) { // Apply ReLU for hidden layers (not for the last output layer)
        l_output_[layer_id][out_id] = std::max(0.0f, l_output_[layer_id][out_id]);
      }
    }

    if (layer_id == 2) {
        std::cout << "[DEBUG NNUE Eval] Scalar Layer 2 Output (l_output_[2]):" << std::endl;
        PrintFloatArraySample("  L2_Out", l_output_[2], layer_sizes_[2]);
    }
    if (layer_id == 3) { // This is the final output layer (logit)
        std::cout << "[DEBUG NNUE Eval] Scalar Layer 3 Output (l_output_[3] - Logit):" << std::endl;
        PrintFloatArraySample("  L3_Logit", l_output_[3], layer_sizes_[3]);
    }
  }

  // Final output value from the network (this is the logit, pre-sigmoid)
  float logit = l_output_[num_layers_ - 1][0];
  std::cout << "[DEBUG NNUE Eval] Scalar Final Logit: " << std::fixed << std::setprecision(6) << logit << std::endl;

#endif

  // --- CRITICAL FIX: Apply sigmoid to the logit before conversion ---
  // The Keras model's last layer has a sigmoid activation.
  // The C++ `logit` variable is the *output of the final dense layer's linear computation + bias*,
  // which is the *pre-sigmoid value* (the logit).
  // The conversion formula `d * std::log(prob/(1.0f-prob))` expects `prob` to be a *probability* (0 to 1).
  // Therefore, we must apply the sigmoid function here to convert the logit to a probability.

  float actual_prob = 1.0f / (1.0f + std::exp(-logit)); 
  std::cout << "[DEBUG NNUE Eval] Actual Probability after Sigmoid: " << std::fixed << std::setprecision(6) << actual_prob << std::endl;

  // Clamp probability to avoid log(0) or log(1)
  constexpr float kEpsilon = 1e-8f; // Use float literal
  float clamped_prob = std::max(kEpsilon, std::min(1.0f - kEpsilon, actual_prob)); 
  std::cout << "[DEBUG NNUE Eval] Clamped Probability: " << std::fixed << std::setprecision(6) << clamped_prob << std::endl;

  // Map from probability to centipawns using the logit inverse (expected for this type of network)
  float d_factor = -10.0f / std::log(1.0f/.9f-1.0f); // Scaling factor (from original NNUE/Stockfish)
  float centipawns = 100.0f * d_factor * std::log(clamped_prob/(1.0f-clamped_prob)); // Log-odds scaled to centipawns

  std::cout << "[DEBUG NNUE Eval] Final Centipawns: " << std::fixed << std::setprecision(2) << centipawns << std::endl;
  return static_cast<int32_t>(centipawns);
}


}  // namespace chess