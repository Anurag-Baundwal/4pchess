/**
 * @file magic_finder.cc
 * @brief A utility program to find magic numbers for 4-player chess sliding piece attacks.
 *
 * This program finds magic numbers for "half-pieces" on the 14x14 4-player board.
 * - Vertical Rooks (North/South)
 * - Horizontal Rooks (East/West)
 * - Diagonal Bishops (Northeast/Southwest)
 * - Anti-Diagonal Bishops (Northwest/Southeast)
 *
 * The output is a single binary file, 'magic_tables.bin', containing the
 * pre-computed magic numbers and attack tables. This binary file should be
 * loaded by the chess engine at startup.
 *
 * The binary file format is a simple concatenation of the data for each piece type:
 * [Rook Vert Magics] [Rook Vert Attacks] [Rook Horiz Magics] [Rook Horiz Attacks] ...
 *
 * - Each MagicEntry table is a fixed-size block: kNumSquares * sizeof(MagicEntry)
 * - Each Attack table is prefixed by a uint64_t indicating its size (number of Bitboards),
 *   followed by the raw Bitboard data.
 *
 * To compile and run with Bazel:
 * 1. Add this target to your BUILD file:
 *    cc_binary(
 *        name = "magic_finder",
 *        srcs = ["magic_finder.cc"],
 *        copts = ["-O3"],
 *        deps = [":board"],
 *    )
 * 2. Build: bazel build -c opt //:magic_finder
 * 3. Run:   ./bazel-bin/magic_finder
 * 
 * To compile and run with gcc:
 * 1. Command: g++ -std=c++17 -O3 -march=native -o magic_finder_gcc magic_finder.cc board.cc
 * 2. Run:     ./magic_finder_gcc
 */

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <array>
#include <string>
#include <chrono>
#include <fstream> // Required for binary file output

#include "board.h"
#include "FastUint256.h"

namespace chess {
namespace magic {

using namespace BitboardImpl;

// Struct to hold all data needed for a magic lookup for one square
// NOTE: This must be a POD (Plain Old Data) type for easy binary writing.
struct MagicEntry {
    Bitboard magic;
    Bitboard mask;
    int shift;
    uint32_t offset; // Offset into the global attack table for this piece type
};

// --- Forward Declarations ---
Bitboard get_blocker_config(int i, Bitboard mask);
Bitboard calculate_attacks(int sq, Bitboard blockers, RayDirection d1, RayDirection d2);
void find_magics_for_square(int sq, RayDirection d1, RayDirection d2, MagicEntry& magic_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset);
void write_piece_data_to_binary(std::ofstream& out, const MagicEntry magics[kNumSquares], const std::vector<Bitboard>& attacks);


// --- Helper Functions ---

std::mt19937_64 rng(0xDEADBEEFCAFEFULL);

Bitboard random_bitboard() {
    return Bitboard(rng(), rng(), rng(), rng());
}

Bitboard random_sparse_bitboard() {
    return random_bitboard() & random_bitboard() & random_bitboard();
}

Bitboard get_blocker_config(uint64_t i, Bitboard mask) {
    Bitboard result(0);
    Bitboard temp_mask = mask;
    while (!temp_mask.is_zero()) {
        int sq = temp_mask.ctz();
        temp_mask &= temp_mask - 1;
        if (i & 1) {
            result |= IndexToBitboard(sq);
        }
        i >>= 1;
    }
    return result;
}

// --- CORRECTED ATTACK CALCULATION ---

// This function correctly calculates the attacks for a single ray,
// including the blocking piece itself, which is required for magic bitboard tables.
Bitboard get_ray_attack_magic(int sq, RayDirection dir, const Bitboard& blockers) {
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

        // The attack set is all squares on the ray up to the blocker,
        // and we remove any squares on the ray that are "behind" the blocker.
        // This correctly includes the blocker square in the attack set.
        return ray & ~kRayAttacks[blocker_idx][dir];
    }
    // If no blockers, the attack set is the entire ray.
    return ray;
}

Bitboard calculate_attacks(int sq, Bitboard blockers, RayDirection d1, RayDirection d2) {
    return get_ray_attack_magic(sq, d1, blockers) | get_ray_attack_magic(sq, d2, blockers);
}


// --- Main Magic Finding Logic ---

void find_magics_for_square(int sq, RayDirection d1, RayDirection d2, MagicEntry& magic_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset) {
    if ((BitboardImpl::kLegalSquares & IndexToBitboard(sq)).is_zero()) {
        magic_entry = {Bitboard(0), Bitboard(0), 0, 0};
        return;
    }

    // --- Mask Generation ---
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

    magic_entry.mask = mask;
    int bits = mask.popcount();
    std::cerr << "  -> Trying sq=" << sq << ", mask bits=" << bits << std::endl;
    magic_entry.shift = 256 - bits;
    
    uint64_t num_configs = 1ULL << bits;
    std::vector<Bitboard> blockers(num_configs);
    std::vector<Bitboard> attacks(num_configs);

    for (uint64_t i = 0; i < num_configs; ++i) {
        blockers[i] = get_blocker_config(i, mask);
        attacks[i] = calculate_attacks(sq, blockers[i], d1, d2);
    }
    
    std::vector<Bitboard> temp_attack_table(num_configs);
    
    for (int attempts = 0; attempts < 100000000; ++attempts) {
        Bitboard magic = random_sparse_bitboard();
        
        // Don't use magics that could have issues with the mask. This is a common optimization.
        if ((magic & Bitboard(0xFF)).is_zero()) {
             continue;
        }
        
        std::fill(temp_attack_table.begin(), temp_attack_table.end(), Bitboard(0));
        bool success = true;

        for (uint64_t i = 0; i < num_configs; ++i) {
            // 1. Use the standard 256x256->256 bit multiplication.
            Bitboard product = blockers[i] * magic;

            // 2. Shift the 256-bit result to get the index.
            //    This takes the top 'bits' of the 256-bit product.
            int index = static_cast<int>(static_cast<uint64_t>(product >> magic_entry.shift));

            if (temp_attack_table[index].is_zero()) {
                temp_attack_table[index] = attacks[i];
            } else if (temp_attack_table[index] != attacks[i]) {
                success = false;
                break;
            }
        }

        if (success) {
            magic_entry.magic = magic;
            magic_entry.offset = current_offset;
            global_attack_table.insert(global_attack_table.end(), temp_attack_table.begin(), temp_attack_table.end());
            current_offset += num_configs;
            return;
        }
    }

    std::cerr << "\nERROR: Failed to find magic for square " << sq << " after max attempts.\n";
    exit(1);
}

// --- Binary Output Function ---

/**
 * @brief Writes the data for one piece type to an open binary file stream.
 * @param out The output file stream, opened in binary mode.
 * @param magics The array of MagicEntry structs to write.
 * @param attacks The vector of Bitboard attacks to write.
 */
void write_piece_data_to_binary(std::ofstream& out, const MagicEntry magics[kNumSquares], const std::vector<Bitboard>& attacks) {
    // 1. Write the fixed-size MagicEntry table.
    out.write(reinterpret_cast<const char*>(magics), kNumSquares * sizeof(MagicEntry));

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


} // namespace magic
} // namespace chess

// --- Main Program ---
int main() {
    using namespace chess;
    using namespace chess::magic;

    auto start_time = std::chrono::high_resolution_clock::now();

    InitBitboards();

    MagicEntry rook_vert_magics[kNumSquares];
    MagicEntry rook_horiz_magics[kNumSquares];
    MagicEntry bishop_diag_magics[kNumSquares];
    MagicEntry bishop_anti_diag_magics[kNumSquares];

    std::vector<Bitboard> rook_vert_attacks;
    std::vector<Bitboard> rook_horiz_attacks;
    std::vector<Bitboard> bishop_diag_attacks;
    std::vector<Bitboard> bishop_anti_diag_attacks;

    uint32_t current_offset;

    // --- Find Horizontal Rook Magics (East/West) ---
    current_offset = 0;
    std::cerr << "Finding magics for Horizontal Rooks (E/W)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        find_magics_for_square(sq, D_E, D_W, rook_horiz_magics[sq], rook_horiz_attacks, current_offset);
        if ((sq+1) % 16 == 0) { std::cerr << "." << std::flush; }
    }
    std::cerr << "\nDone. Total attack table size: " << rook_horiz_attacks.size() << "\n\n";

    // --- Find Vertical Rook Magics (North/South) ---
    current_offset = 0;
    std::cerr << "Finding magics for Vertical Rooks (N/S)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        find_magics_for_square(sq, D_N, D_S, rook_vert_magics[sq], rook_vert_attacks, current_offset);
        if ((sq+1) % 16 == 0) { std::cerr << "." << std::flush; }
    }
    std::cerr << "\nDone. Total attack table size: " << rook_vert_attacks.size() << "\n\n";

    // --- Find Diagonal Bishop Magics (NE/SW) ---
    current_offset = 0;
    std::cerr << "Finding magics for Diagonal Bishops (NE/SW)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        find_magics_for_square(sq, D_NE, D_SW, bishop_diag_magics[sq], bishop_diag_attacks, current_offset);
        if ((sq+1) % 16 == 0) { std::cerr << "." << std::flush; }
    }
    std::cerr << "\nDone. Total attack table size: " << bishop_diag_attacks.size() << "\n\n";

    // --- Find Anti-Diagonal Bishop Magics (NW/SE) ---
    current_offset = 0;
    std::cerr << "Finding magics for Anti-Diagonal Bishops (NW/SE)..." << std::endl;
    for (int sq = 0; sq < kNumSquares; ++sq) {
        find_magics_for_square(sq, D_NW, D_SE, bishop_anti_diag_magics[sq], bishop_anti_diag_attacks, current_offset);
        if ((sq+1) % 16 == 0) { std::cerr << "." << std::flush; }
    }
    std::cerr << "\nDone. Total attack table size: " << bishop_anti_diag_attacks.size() << "\n\n";

    // --- Write all generated data to a single binary file ---
    const std::string bin_filename = "magic_tables.bin";
    std::cerr << "\n--- Generating " << bin_filename << " ---\n";

    std::ofstream out(bin_filename, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "FATAL: Could not open " << bin_filename << " for writing.\n";
        return 1;
    }

    write_piece_data_to_binary(out, rook_vert_magics, rook_vert_attacks);
    write_piece_data_to_binary(out, rook_horiz_magics, rook_horiz_attacks);
    write_piece_data_to_binary(out, bishop_diag_magics, bishop_diag_attacks);
    write_piece_data_to_binary(out, bishop_anti_diag_magics, bishop_anti_diag_attacks);

    out.close();
    std::cerr << "Successfully wrote all tables to " << bin_filename << "\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cerr << "\nTotal generation time: " << std::fixed << std::setprecision(2) << elapsed.count() << " seconds.\n";

    return 0;
}