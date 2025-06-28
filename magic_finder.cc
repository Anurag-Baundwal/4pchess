/**
 * @file pext_table_generator.cc (Formerly magic_finder.cc)
 * @brief A utility program to generate tables for PEXT-based sliding piece attacks.
 *
 * This program generates tables for "half-pieces" on the 14x14 4-player board.
 * - Vertical Rooks (North/South)
 * - Horizontal Rooks (East/West)
 * - Diagonal Bishops (Northeast/Southwest)
 * - Anti-Diagonal Bishops (Northwest/Southeast)
 *
 * It uses the PDEP instruction (the inverse of PEXT) to generate every possible
 * blocker combination for a given mask and stores the pre-calculated attack for each.
 * This requires the generator to be compiled with BMI2 support.
 *
 * The output is a single binary file, 'magic_tables.bin', containing the
 * pre-computed masks and attack tables.
 *
 * The binary file format is a simple concatenation of the data for each piece type:
 * [Rook Vert PextEntries] [Rook Vert Attacks] [Rook Horiz PextEntries] [Rook Horiz Attacks] ...
 *
 * - Each PextEntry table is a fixed-size block: kNumSquares * sizeof(PextEntry)
 * - Each Attack table is prefixed by a uint64_t indicating its size (number of Bitboards),
 *   followed by the raw Bitboard data.
 *
 * To compile and run with Bazel (assuming the .bazelrc change is made):
 * 1. Build: bazel build -c opt --config=bmi2 //:magic_finder
 * 2. Run:   ./bazel-bin/magic_finder
 *
 * To compile and run with g++:
 * 1. Command: g++ -std=c++17 -O3 -mbmi2 -o pext_generator pext_table_generator.cc board.cc
 * 2. Run:     ./pext_generator
 */
#include <immintrin.h> // For _pdep_u64
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <array>
#include <string>
#include <chrono>
#include <fstream>

#include "board.h"
#include "FastUint256.h"

// PEXT requires BMI2, PDEP is also in BMI2
#if !defined(__BMI2__)
#error "This generator must be compiled with BMI2 support (e.g., -mbmi2 or /arch:AVX2)"
#endif

namespace chess {
namespace pext {

using namespace BitboardImpl;

// Struct to hold all data needed for a PEXT lookup for one square.
// This is much simpler than a MagicEntry.
// NOTE: This must be a POD (Plain Old Data) type for easy binary writing.
struct PextEntry {
    Bitboard mask;
    uint32_t offset; // Offset into the global attack table for this piece type
};

// --- Forward Declarations ---
Bitboard calculate_attacks(int sq, Bitboard blockers, RayDirection d1, RayDirection d2);
void generate_tables_for_square(int sq, RayDirection d1, RayDirection d2, PextEntry& pext_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset);
void write_piece_data_to_binary(std::ofstream& out, const PextEntry pext_entries[kNumSquares], const std::vector<Bitboard>& attacks);


// This function correctly calculates the attacks for a single ray,
// including the blocking piece itself, which is required for lookup tables.
Bitboard get_ray_attack_pext(int sq, RayDirection dir, const Bitboard& blockers) {
    Bitboard ray = kRayAttacks[sq][dir];
    Bitboard b = ray & blockers;
    if (!b.is_zero()) {
        int blocker_idx;
        
        // Increasing directions: first blocker is LSB
        bool is_increasing_ray = (dir == D_S || dir == D_E || dir == D_SE || dir == D_SW);
        if (is_increasing_ray) {
            blocker_idx = b.ctz();
        } 
        // Decreasing directions: first blocker is MSB
        else {
            blocker_idx = 255 - b.clz();
        }
        return ray & ~kRayAttacks[blocker_idx][dir];
    }
    return ray;
}

Bitboard calculate_attacks(int sq, Bitboard blockers, RayDirection d1, RayDirection d2) {
    return get_ray_attack_pext(sq, d1, blockers) | get_ray_attack_pext(sq, d2, blockers);
}


// --- Main PEXT Table Generation Logic ---

// Helper function to perform PDEP on our 256-bit bitboard
Bitboard pdep_256(uint64_t source, Bitboard mask) {
    Bitboard result = {};
    uint64_t current_source = source;

    for (int i = 0; i < 4; ++i) {
        // The number of bits to take from the source is the popcount of the mask's current limb.
        int bits_in_limb_mask = __builtin_popcountll(mask.limbs[i]);
        
        // Take that many bits from the current source
        uint64_t source_chunk = current_source & ((1ULL << bits_in_limb_mask) - 1);

        // Deposit them into the result limb
        result.limbs[i] = _pdep_u64(source_chunk, mask.limbs[i]);

        // Shift the source to prepare for the next limb
        current_source >>= bits_in_limb_mask;
    }
    return result;
}


void generate_tables_for_square(int sq, RayDirection d1, RayDirection d2, PextEntry& pext_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset) {
    if ((kLegalSquares & IndexToBitboard(sq)).is_zero()) {
        pext_entry = {Bitboard(0), 0};
        return;
    }

    // --- Mask Generation (same as magic finder) ---
    Bitboard mask = (kRayAttacks[sq][d1] | kRayAttacks[sq][d2]);
    Bitboard ray1 = kRayAttacks[sq][d1];
    Bitboard ray2 = kRayAttacks[sq][d2];
    auto is_decreasing_dir = [](RayDirection d) {
      return d == D_N || d == D_W || d == D_NW || d == D_NE;
    };
    if (!ray1.is_zero()) {
        if (is_decreasing_dir(d1)) mask &= ~IndexToBitboard(ray1.ctz());
        else mask &= ~IndexToBitboard(255 - ray1.clz());
    }
    if (!ray2.is_zero()) {
        if (is_decreasing_dir(d2)) mask &= ~IndexToBitboard(ray2.ctz());
        else mask &= ~IndexToBitboard(255 - ray2.clz());
    }
    // --- End Mask Generation ---

    pext_entry.mask = mask;
    int bits = mask.popcount();
    
    uint64_t num_configs = 1ULL << bits;
    std::vector<Bitboard> local_attack_table(num_configs);

    // Generate all blocker configurations and their attacks
    for (uint64_t i = 0; i < num_configs; ++i) {
        // Use PDEP to create the blocker pattern from the index 'i'
        Bitboard blockers = pdep_256(i, mask);
        local_attack_table[i] = calculate_attacks(sq, blockers, d1, d2);
    }
    
    // Add the generated local table to the global table
    pext_entry.offset = current_offset;
    global_attack_table.insert(global_attack_table.end(), local_attack_table.begin(), local_attack_table.end());
    current_offset += num_configs;

    std::cerr << "  -> sq=" << sq << ", mask bits=" << bits << ", table size=" << num_configs << std::endl;
}

// --- Binary Output Function ---

void write_piece_data_to_binary(std::ofstream& out, const PextEntry pext_entries[kNumSquares], const std::vector<Bitboard>& attacks) {
    // 1. Write the fixed-size PextEntry table.
    out.write(reinterpret_cast<const char*>(pext_entries), kNumSquares * sizeof(PextEntry));

    // 2. Write the size of the variable-sized attack table.
    uint64_t attack_table_size = attacks.size();
    out.write(reinterpret_cast<const char*>(&attack_table_size), sizeof(uint64_t));

    // 3. Write the raw data of the attack table.
    out.write(reinterpret_cast<const char*>(attacks.data()), attack_table_size * sizeof(Bitboard));

    if (!out) {
        std::cerr << "FATAL: Failed to write data to binary file.\n";
        exit(1);
    }
}


} // namespace pext
} // namespace chess

// --- Main Program ---
int main() {
    using namespace chess;
    using namespace chess::pext;

    auto start_time = std::chrono::high_resolution_clock::now();

    InitBitboards();

    PextEntry rook_vert_pext[kNumSquares];
    PextEntry rook_horiz_pext[kNumSquares];
    PextEntry bishop_diag_pext[kNumSquares];
    PextEntry bishop_anti_diag_pext[kNumSquares];

    std::vector<Bitboard> rook_vert_attacks;
    std::vector<Bitboard> rook_horiz_attacks;
    std::vector<Bitboard> bishop_diag_attacks;
    std::vector<Bitboard> bishop_anti_diag_attacks;

    uint32_t current_offset;

    // --- Generate Horizontal Rook Tables (East/West) ---
    current_offset = 0;
    std::cerr << "Generating tables for Horizontal Rooks (E/W)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        generate_tables_for_square(sq, D_E, D_W, rook_horiz_pext[sq], rook_horiz_attacks, current_offset);
    }
    std::cerr << "Done. Total attack table size: " << rook_horiz_attacks.size() << "\n\n";

    // --- Generate Vertical Rook Tables (North/South) ---
    current_offset = 0;
    std::cerr << "Generating tables for Vertical Rooks (N/S)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        generate_tables_for_square(sq, D_N, D_S, rook_vert_pext[sq], rook_vert_attacks, current_offset);
    }
    std::cerr << "Done. Total attack table size: " << rook_vert_attacks.size() << "\n\n";

    // --- Generate Diagonal Bishop Tables (NE/SW) ---
    current_offset = 0;
    std::cerr << "Generating tables for Diagonal Bishops (NE/SW)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        generate_tables_for_square(sq, D_NE, D_SW, bishop_diag_pext[sq], bishop_diag_attacks, current_offset);
    }
    std::cerr << "Done. Total attack table size: " << bishop_diag_attacks.size() << "\n\n";

    // --- Generate Anti-Diagonal Bishop Tables (NW/SE) ---
    current_offset = 0;
    std::cerr << "Generating tables for Anti-Diagonal Bishops (NW/SE)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        generate_tables_for_square(sq, D_NW, D_SE, bishop_anti_diag_pext[sq], bishop_anti_diag_attacks, current_offset);
    }
    std::cerr << "Done. Total attack table size: " << bishop_anti_diag_attacks.size() << "\n\n";

    // --- Write all generated data to a single binary file ---
    const std::string bin_filename = "magic_tables.bin";
    std::cerr << "\n--- Generating " << bin_filename << " ---\n";

    std::ofstream out(bin_filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "FATAL: Could not open " << bin_filename << " for writing.\n";
        return 1;
    }

    write_piece_data_to_binary(out, rook_vert_pext, rook_vert_attacks);
    write_piece_data_to_binary(out, rook_horiz_pext, rook_horiz_attacks);
    write_piece_data_to_binary(out, bishop_diag_pext, bishop_diag_attacks);
    write_piece_data_to_binary(out, bishop_anti_diag_pext, bishop_anti_diag_attacks);

    out.close();
    std::cerr << "Successfully wrote all tables to " << bin_filename << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cerr << "\nTotal generation time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds.\n";

    return 0;
}