#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <fstream>

// For PEXT intrinsics
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#include <immintrin.h>
#endif

#include "board.h"

namespace chess {

// ============================================================================
// Bitboard Implementation Details
// ============================================================================
namespace BitboardImpl {

// Precomputed data
Bitboard kLegalSquares;
Bitboard kPawnStartMask[4];
Bitboard kPawnPromotionMask[4];
Bitboard kKnightAttacks[kNumSquares];
Bitboard kKingAttacks[kNumSquares];
Bitboard kPawnSinglePush[4][kNumSquares];
Bitboard kPawnDoublePush[4][kNumSquares];
Bitboard kPawnAttacks[4][kNumSquares];
Bitboard kRayAttacks[kNumSquares][8]; // 0-3: Bishop, 4-7: Rook
Bitboard kLineBetween[kNumSquares][kNumSquares];
Bitboard kCastlingEmptyMask[4][2]; // [color][side]
Bitboard kCastlingAttackMask[4][2]; // [color][side]
Bitboard kBackRankMasks[4];
Bitboard kSecondRankMasks[4];
Bitboard kCentralMask;
int kInitialRookSq[4][2];

// --- PEXT Bitboard Data (loaded from file) ---
// This struct must match the one in the new magic_finder.cc (pext_table_generator.cc)
struct PextEntry {
    Bitboard mask;
    uint32_t offset;
};

// Global tables for PEXT bitboards. These are populated by LoadPextTables().
PextEntry kRookHorizPext[kNumSquares];
PextEntry kRookVertPext[kNumSquares];
PextEntry kBishopDiagPext[kNumSquares];
PextEntry kBishopAntiDiagPext[kNumSquares];
std::vector<Bitboard> kRookHorizAttacksTable;
std::vector<Bitboard> kRookVertAttacksTable;
std::vector<Bitboard> kBishopDiagAttacksTable;
std::vector<Bitboard> kBishopAntiDiagAttacksTable;

// Helper function to read data for one piece type from the binary file.
void read_piece_data_from_binary(std::ifstream& in, PextEntry pext_entries[kNumSquares], std::vector<Bitboard>& attacks) {
    // 1. Read the fixed-size PextEntry table.
    in.read(reinterpret_cast<char*>(pext_entries), kNumSquares * sizeof(PextEntry));
    if (!in) {
        std::cerr << "FATAL: Failed to read PextEntry table from magic_tables.bin.\n";
        exit(1);
    }

    // 2. Read the size of the variable-sized attack table.
    uint64_t attack_table_size = 0;
    in.read(reinterpret_cast<char*>(&attack_table_size), sizeof(uint64_t));
     if (!in) {
        std::cerr << "FATAL: Failed to read attack table size from magic_tables.bin.\n";
        exit(1);
    }

    if (attack_table_size > 500000) { // A sanity check, your largest table is ~140k
        std::cerr << "FATAL: Attack table size is unreasonably large. "
                  << "magic_tables.bin is likely corrupt or not found." << std::endl;
        exit(1);
    }
    
    // 3. Resize the vector and read the raw data of the attack table.
    attacks.resize(attack_table_size);
    in.read(reinterpret_cast<char*>(attacks.data()), attack_table_size * sizeof(Bitboard));
    if (!in) {
        std::cerr << "FATAL: Failed to read attack table data from magic_tables.bin.\n";
        exit(1);
    }
}

void LoadPextTables() {
    static bool is_loaded = false;
    if (is_loaded) return;

    const std::string bin_filename = "magic_tables.bin";
    std::ifstream in(bin_filename, std::ios::binary);

    if (!in) {
        std::cerr << "\nFATAL ERROR: Could not open magic_tables.bin.\n"
                  << "The engine cannot run without this file.\n"
                  << "Please generate it by compiling and running the 'magic_finder' target with BMI2 enabled.\n"
                  << "Example: bazel run -c opt --config=bmi2 //:magic_finder\n" << std::endl;
        exit(1);
    }

    read_piece_data_from_binary(in, kRookVertPext, kRookVertAttacksTable);
    read_piece_data_from_binary(in, kRookHorizPext, kRookHorizAttacksTable);
    read_piece_data_from_binary(in, kBishopDiagPext, kBishopDiagAttacksTable);
    read_piece_data_from_binary(in, kBishopAntiDiagPext, kBishopAntiDiagAttacksTable);
    
    in.close();
    is_loaded = true;
}


// We keep the old magic bitboard loading logic for the fallback path
namespace magics {
    struct MagicEntry {
        Bitboard magic;
        Bitboard mask;
        int shift;
        uint32_t offset;
    };
    MagicEntry kRookHorizMagics[kNumSquares];
    MagicEntry kRookVertMagics[kNumSquares];
    MagicEntry kBishopDiagMagics[kNumSquares];
    MagicEntry kBishopAntiDiagMagics[kNumSquares];
    void read_piece_data_from_binary_magic(std::ifstream& in, MagicEntry magics[kNumSquares], std::vector<Bitboard>& attacks) {
        in.read(reinterpret_cast<char*>(magics), kNumSquares * sizeof(MagicEntry));
        if (!in) exit(1);
        uint64_t attack_table_size = 0;
        in.read(reinterpret_cast<char*>(&attack_table_size), sizeof(uint64_t));
        if (!in || attack_table_size > 500000) exit(1);
        attacks.resize(attack_table_size);
        in.read(reinterpret_cast<char*>(attacks.data()), attack_table_size * sizeof(Bitboard));
        if (!in) exit(1);
    }
    void LoadMagicTables() {
        static bool is_loaded_magic = false;
        if (is_loaded_magic) return;
        std::ifstream in("magic_tables.bin", std::ios::binary);
        if(!in) {
            std::cerr << "FATAL: Could not open magic_tables.bin for fallback magic bitboards.\n";
            exit(1);
        }
        read_piece_data_from_binary_magic(in, kRookVertMagics, kRookVertAttacksTable);
        read_piece_data_from_binary_magic(in, kRookHorizMagics, kRookHorizAttacksTable);
        read_piece_data_from_binary_magic(in, kBishopDiagMagics, kBishopDiagAttacksTable);
        read_piece_data_from_binary_magic(in, kBishopAntiDiagMagics, kBishopAntiDiagAttacksTable);
        in.close();
        is_loaded_magic = true;
    }
}

void InitBitboards() {
    static bool is_initialized = false;
    if (is_initialized) return;

    // Legal squares mask
    for (int r_14 = 0; r_14 < 14; ++r_14) {
        for (int c_14 = 0; c_14 < 14; ++c_14) {
             if (!((r_14 < 3 && (c_14 < 3 || c_14 > 10)) || (r_14 > 10 && (c_14 < 3 || c_14 > 10)))) {
                kLegalSquares |= IndexToBitboard(LocationToIndex(BoardLocation(r_14, c_14)));
            }
        }
    }

    // Pawn masks & moves
    int push_offsets[] = {-kBoardWidth, 1, kBoardWidth, -1};
    // Deltas for the two pawn captures for each color. Format: { {d_row, d_col}, {d_row, d_col} }
    const int pawn_capture_deltas[4][2][2] = {
        /* [RED]    */ { {-1, -1}, {-1,  1} },  // Attacks are (-dr, -dc) and (-dr, +dc)
        /* [BLUE]   */ { {-1,  1}, { 1,  1} },  // Attacks are (-dr, +dc) and (+dr, +dc)
        /* [YELLOW] */ { { 1, -1}, { 1,  1} },  // Attacks are (+dr, -dc) and (+dr, +dc)
        /* [GREEN]  */ { {-1, -1}, { 1, -1} }   // Attacks are (-dr, -dc) and (+dr, -dc)
    };

    for (int r_14 = 0; r_14 < 14; ++r_14) {
        for (int c_14 = 0; c_14 < 14; ++c_14) {
            BoardLocation from_loc(r_14, c_14);
            if (!(kLegalSquares & IndexToBitboard(LocationToIndex(from_loc)))) continue;

            int idx = LocationToIndex(from_loc);
            
            // Start & Promotion
            if (r_14 == 12) kPawnStartMask[RED] |= IndexToBitboard(idx);
            if (c_14 == 1)  kPawnStartMask[BLUE] |= IndexToBitboard(idx);
            if (r_14 == 1)  kPawnStartMask[YELLOW] |= IndexToBitboard(idx);
            if (c_14 == 12) kPawnStartMask[GREEN] |= IndexToBitboard(idx);
            
            if (r_14 == 3)  kPawnPromotionMask[RED] |= IndexToBitboard(idx);
            if (c_14 == 10) kPawnPromotionMask[BLUE] |= IndexToBitboard(idx);
            if (r_14 == 10) kPawnPromotionMask[YELLOW] |= IndexToBitboard(idx);
            if (c_14 == 3)  kPawnPromotionMask[GREEN] |= IndexToBitboard(idx);

            for (int color = 0; color < 4; ++color) {
                // Pushes
                BoardLocation to1 = from_loc.Relative(push_offsets[color] / kBoardWidth, push_offsets[color] % kBoardWidth);
                if ((kLegalSquares & IndexToBitboard(LocationToIndex(to1)))) {
                    kPawnSinglePush[color][idx] = IndexToBitboard(LocationToIndex(to1));
                    if (kPawnStartMask[color] & IndexToBitboard(idx)) {
                         BoardLocation to2 = to1.Relative(push_offsets[color] / kBoardWidth, push_offsets[color] % kBoardWidth);
                         if ((kLegalSquares & IndexToBitboard(LocationToIndex(to2)))) {
                            kPawnDoublePush[color][idx] = IndexToBitboard(LocationToIndex(to2));
                         }
                    }
                }
                // Captures
                for (int k = 0; k < 2; ++k) {
                    const int dr = pawn_capture_deltas[color][k][0];
                    const int dc = pawn_capture_deltas[color][k][1];
                    BoardLocation to_cap = from_loc.Relative(dr, dc);
                    if ((kLegalSquares & IndexToBitboard(LocationToIndex(to_cap)))) {
                        kPawnAttacks[color][idx] |= IndexToBitboard(LocationToIndex(to_cap));
                    }
                }
            }
        }
    }

    // Non-sliding pieces and rays
    for (int i = 0; i < kNumSquares; ++i) {
        BoardLocation from = IndexToLocation(i);
        if (!from.Present() || !(kLegalSquares & IndexToBitboard(i))) continue;

        // Knight
        int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
        int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
        for (int k = 0; k < 8; ++k) {
            BoardLocation to = from.Relative(dr[k], dc[k]);
            if ((kLegalSquares & IndexToBitboard(LocationToIndex(to)))) 
                kKnightAttacks[i] |= IndexToBitboard(LocationToIndex(to));
        }

        // King
        for (int r = -1; r <= 1; ++r) {
            for (int c = -1; c <= 1; ++c) {
                if (r == 0 && c == 0) continue;
                BoardLocation to = from.Relative(r, c);
                if ((kLegalSquares & IndexToBitboard(LocationToIndex(to)))) 
                    kKingAttacks[i] |= IndexToBitboard(LocationToIndex(to));
            }
        }

        // Rays
        int ray_dr[] = {-1, -1, 1, 1, -1, 0, 1, 0};
        int ray_dc[] = {1, -1, 1, -1, 0, 1, 0, -1};
        for (int d = 0; d < 8; ++d) {
            BoardLocation cur = from.Relative(ray_dr[d], ray_dc[d]);
            while ((kLegalSquares & IndexToBitboard(LocationToIndex(cur)))) {
                kRayAttacks[i][d] |= IndexToBitboard(LocationToIndex(cur));
                cur = cur.Relative(ray_dr[d], ray_dc[d]);
            }
        }
    }

    // Line Between Masks
    for (int i = 0; i < kNumSquares; i++) {
        for (int j = 0; j < kNumSquares; j++) {
            if (i == j) continue;
            for (int d = 0; d < 8; d++) {
                if ((kRayAttacks[i][d] & IndexToBitboard(j))) {
                    int opposite_dir;
                    switch (d) {
                        case D_NE: opposite_dir = D_SW; break;
                        case D_NW: opposite_dir = D_SE; break;
                        case D_SE: opposite_dir = D_NW; break;
                        case D_SW: opposite_dir = D_NE; break;
                        case D_N:  opposite_dir = D_S;  break;
                        case D_E:  opposite_dir = D_W;  break;
                        case D_S:  opposite_dir = D_N;  break;
                        case D_W:  opposite_dir = D_E;  break;
                    }
                    kLineBetween[i][j] = kRayAttacks[i][d] & kRayAttacks[j][opposite_dir];
                    break;
                }
            }
        }
    }
    
    // Castling Masks
    // King start squares: R(13,7), B(7,0), Y(0,6), G(6,13)
    // Rook start squares are more complex
    BoardLocation king_starts[] = {{13, 7}, {7, 0}, {0, 6}, {6, 13}};
    kInitialRookSq[RED][KINGSIDE] = LocationToIndex({13, 10});
    kInitialRookSq[RED][QUEENSIDE] = LocationToIndex({13, 3});
    kInitialRookSq[BLUE][KINGSIDE] = LocationToIndex({10, 0});
    kInitialRookSq[BLUE][QUEENSIDE] = LocationToIndex({3, 0});
    kInitialRookSq[YELLOW][KINGSIDE] = LocationToIndex({0, 3});
    kInitialRookSq[YELLOW][QUEENSIDE] = LocationToIndex({0, 10});
    kInitialRookSq[GREEN][KINGSIDE] = LocationToIndex({3, 13});
    kInitialRookSq[GREEN][QUEENSIDE] = LocationToIndex({10, 13});

    for(int c=0; c<4; ++c) {
        int king_sq = LocationToIndex(king_starts[c]);
        // Kingside
        kCastlingEmptyMask[c][KINGSIDE] = kLineBetween[king_sq][kInitialRookSq[c][KINGSIDE]];
        kCastlingAttackMask[c][KINGSIDE] = IndexToBitboard(king_sq) | IndexToBitboard(king_sq + push_offsets[(c+1)%4]) | IndexToBitboard(king_sq + 2*push_offsets[(c+1)%4]);
        // Queenside
        kCastlingEmptyMask[c][QUEENSIDE] = kLineBetween[king_sq][kInitialRookSq[c][QUEENSIDE]];
        kCastlingAttackMask[c][QUEENSIDE] = IndexToBitboard(king_sq) | IndexToBitboard(king_sq + push_offsets[(c+3)%4]) | IndexToBitboard(king_sq + 2*push_offsets[(c+3)%4]);
    }
    
    // Back Rank Masks
    for (int c = 0; c < 14; ++c) {
        BoardLocation loc_r(13, (int8_t)c);
        if (loc_r.Present())
            kBackRankMasks[RED] |= IndexToBitboard(LocationToIndex(loc_r));
        
        BoardLocation loc_y(0, (int8_t)c);
        if (loc_y.Present())
            kBackRankMasks[YELLOW] |= IndexToBitboard(LocationToIndex(loc_y));
    }
    for (int r = 0; r < 14; ++r) {
        BoardLocation loc_b((int8_t)r, 0);
        if (loc_b.Present())
            kBackRankMasks[BLUE] |= IndexToBitboard(LocationToIndex(loc_b));

        BoardLocation loc_g((int8_t)r, 13);
        if (loc_g.Present())
            kBackRankMasks[GREEN] |= IndexToBitboard(LocationToIndex(loc_g));
    }

    // Second-to-last Rank Masks (for mobility calculation)
    for (int c = 0; c < 14; ++c) {
        BoardLocation loc_r(12, (int8_t)c); // RED's 2nd rank
        if (loc_r.Present())
            kSecondRankMasks[RED] |= IndexToBitboard(LocationToIndex(loc_r));
        
        BoardLocation loc_y(1, (int8_t)c); // YELLOW's 2nd rank
        if (loc_y.Present())
            kSecondRankMasks[YELLOW] |= IndexToBitboard(LocationToIndex(loc_y));
    }
    for (int r = 0; r < 14; ++r) {
        BoardLocation loc_b((int8_t)r, 1); // BLUE's 2nd rank
        if (loc_b.Present())
            kSecondRankMasks[BLUE] |= IndexToBitboard(LocationToIndex(loc_b));

        BoardLocation loc_g((int8_t)r, 12); // GREEN's 2nd rank
        if (loc_g.Present())
            kSecondRankMasks[GREEN] |= IndexToBitboard(LocationToIndex(loc_g));
    }

    // Central 8x8 Mask
    for (int r_14 = 3; r_14 <= 10; ++r_14) {
        for (int c_14 = 3; c_14 <= 10; ++c_14) {
            kCentralMask |= IndexToBitboard(LocationToIndex(BoardLocation(r_14, c_14)));
        }
    }
    
    // Load bitboard tables from file
    #if defined(__BMI2__)
        std::cout << "Loading PEXT bitboard tables..." << std::endl;
        LoadPextTables();
        std::cout << "PEXT bitboard tables loaded successfully." << std::endl;
    #else
        std::cout << "Loading magic bitboard tables (fallback)..." << std::endl;
        magics::LoadMagicTables(); // Use the namespaced version
        std::cout << "Magic bitboard tables loaded successfully." << std::endl;
    #endif
    
    is_initialized = true;
}

// These correspond to the offsets for a 1-square move in that direction.
constexpr int PUSH_N = -kBoardWidth; // -16
constexpr int PUSH_E = 1;
constexpr int PUSH_S = kBoardWidth;  // +16
constexpr int PUSH_W = -1;
constexpr int PUSH_NE = PUSH_N + PUSH_E; // -15
constexpr int PUSH_NW = PUSH_N + PUSH_W; // -17
constexpr int PUSH_SE = PUSH_S + PUSH_E; // +17
constexpr int PUSH_SW = PUSH_S + PUSH_W; // +15

// A generic, templated shift function, just like Stockfish.
// This is much more efficient than a runtime-variable shift.
template<int ShiftOffset>
inline Bitboard shift(Bitboard b) {
    if constexpr (ShiftOffset > 0) {
        return b << ShiftOffset;
    } else {
        return b >> -ShiftOffset;
    }
}

} // namespace BitboardImpl
using namespace BitboardImpl;

constexpr int kMobilityMultiplier = 5;
Piece Piece::kNoPiece = Piece();
BoardLocation BoardLocation::kNoLocation = BoardLocation();
CastlingRights CastlingRights::kMissingRights = CastlingRights();

const Player kRedPlayer = Player(RED);
const Player kBluePlayer = Player(BLUE);
const Player kYellowPlayer = Player(YELLOW);
const Player kGreenPlayer = Player(GREEN);

Board::Board(
    Player turn,
    std::unordered_map<BoardLocation, Piece> location_to_piece,
    std::optional<std::unordered_map<Player, CastlingRights>> castling_rights,
    std::optional<EnpassantInitialization> enp)
  : turn_(std::move(turn))
{
  InitBitboards();

  for (auto& bb_arr : piece_bitboards_) for(auto& bb : bb_arr) bb.limbs.fill(0);
  for (auto& bb : color_bitboards_) bb.limbs.fill(0);
  for (auto& bb : team_bitboards_) bb.limbs.fill(0);

  for (int color = 0; color < 4; color++) {
    castling_rights_[color] = CastlingRights(false, false);
    if (castling_rights.has_value()) {
      auto& cr = *castling_rights;
      Player pl(static_cast<PlayerColor>(color));
      auto it = cr.find(pl);
      if (it != cr.end()) {
        castling_rights_[color] = it->second;
      }
    }
  }
  if (enp.has_value()) {
    enp_ = std::move(*enp);
  }

  for (const auto& it : location_to_piece) {
      SetPiece(it.first, it.second);
  }
  
  std::mt19937_64 rng(958829);
  for (int color = 0; color < 4; color++) {
    turn_hashes_[color] = rng();
  }
  for (int color = 0; color < 4; color++) {
    for (int piece_type = 0; piece_type < 6; piece_type++) {
      for (int i = 0; i < kNumSquares; i++) {
          piece_hashes_[color][piece_type][i] = rng();
      }
    }
  }
  InitializeHash();
}

void Board::InitializeHash() {
    hash_key_ = 0;
    for (int c = 0; c < 4; ++c) {
        for (int pt = 0; pt < 6; ++pt) {
            Bitboard bb = piece_bitboards_[c][pt];
            while(!bb.is_zero()) {
                int idx = bb.ctz();
                UpdatePieceHash(Piece(static_cast<PlayerColor>(c), static_cast<PieceType>(pt)), idx);
                bb &= (bb - 1);
            }
        }
    }
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));
}

Piece Board::GetPiece(int index) const {
    if (index < 0) return Piece::kNoPiece;
    return piece_on_square_[index];
}

Piece Board::GetPiece(const BoardLocation& location) const {
    return GetPiece(LocationToIndex(location));
}

void Board::SetPiece(const BoardLocation& location, const Piece& piece) {
    int index = LocationToIndex(location);
    if (index < 0 || !piece.Present()) return;

    Bitboard mask = IndexToBitboard(index);
    PlayerColor color = piece.GetColor();
    PieceType type = piece.GetPieceType();
    Team team = piece.GetTeam();

    piece_bitboards_[color][type] |= mask;
    color_bitboards_[color] |= mask;
    team_bitboards_[team] |= mask;

    int piece_eval = kPieceEvaluations[type];
    if (team == RED_YELLOW) piece_evaluation_ += piece_eval;
    else piece_evaluation_ -= piece_eval;
    player_piece_evaluations_[color] += piece_eval;

    piece_on_square_[index] = piece;

    UpdatePieceHash(piece, index);
}

void Board::RemovePiece(const BoardLocation& location) {
    Piece piece = GetPiece(location);
    int index = LocationToIndex(location);
    if (index < 0 || !piece.Present()) return;
    
    Bitboard mask = ~(IndexToBitboard(index));
    PlayerColor color = piece.GetColor();
    PieceType type = piece.GetPieceType();
    Team team = piece.GetTeam();

    piece_bitboards_[color][type] &= mask;
    color_bitboards_[color] &= mask;
    team_bitboards_[team] &= mask;
    
    int piece_eval = kPieceEvaluations[type];
    if (team == RED_YELLOW) piece_evaluation_ -= piece_eval;
    else piece_evaluation_ += piece_eval;
    player_piece_evaluations_[color] -= piece_eval;
    
    piece_on_square_[index] = Piece::kNoPiece;

    UpdatePieceHash(piece, index);
}

void Board::MovePiece(const BoardLocation& from_loc, const BoardLocation& to_loc) {
    Piece piece = GetPiece(from_loc);
    int from_idx = LocationToIndex(from_loc);
    int to_idx = LocationToIndex(to_loc);
    if (from_idx < 0 || to_idx < 0 || !piece.Present()) return;

    Bitboard move_mask = IndexToBitboard(from_idx) | IndexToBitboard(to_idx);
    PlayerColor color = piece.GetColor();
    PieceType type = piece.GetPieceType();
    Team team = piece.GetTeam();
    
    piece_bitboards_[color][type] ^= move_mask;
    color_bitboards_[color] ^= move_mask;
    team_bitboards_[team] ^= move_mask;

    piece_on_square_[to_idx] = piece;
    piece_on_square_[from_idx] = Piece::kNoPiece;

    UpdatePieceHash(piece, from_idx);
    UpdatePieceHash(piece, to_idx);
}

BoardLocation Board::GetKingLocation(PlayerColor color) const {
    Bitboard king_bb = piece_bitboards_[color][KING];
    if (king_bb.is_zero()) {
        return BoardLocation::kNoLocation;
    }
    return IndexToLocation(king_bb.ctz());
}

#if defined(__BMI2__)
// Helper function to perform PEXT on our 256-bit bitboard
uint64_t pext_256(Bitboard source, Bitboard mask) {
    uint64_t result = 0;
    int current_shift = 0;

    for (int i = 0; i < 4; ++i) {
        uint64_t extracted = _pext_u64(source.limbs[i], mask.limbs[i]);
        result |= (extracted << current_shift);
        current_shift += __builtin_popcountll(mask.limbs[i]);
    }
    return result;
}
#endif

// PEXT/Magic Bitboard implementation for Rook attacks
Bitboard Board::GetRookAttacks(int sq, Bitboard blockers) const {
#if defined(__BMI2__)
    // Horizontal attacks (E/W) with PEXT
    const PextEntry& horiz_entry = kRookHorizPext[sq];
    uint64_t horiz_index = pext_256(blockers, horiz_entry.mask);
    Bitboard horiz_attacks = kRookHorizAttacksTable[horiz_entry.offset + horiz_index];

    // Vertical attacks (N/S) with PEXT
    const PextEntry& vert_entry = kRookVertPext[sq];
    uint64_t vert_index = pext_256(blockers, vert_entry.mask);
    Bitboard vert_attacks = kRookVertAttacksTable[vert_entry.offset + vert_index];

    return horiz_attacks | vert_attacks;
#else
    // Horizontal attacks (E/W) with Magic Bitboards (FALLBACK)
    const magics::MagicEntry& horiz_entry = magics::kRookHorizMagics[sq];
    Bitboard horiz_product = (blockers & horiz_entry.mask) * horiz_entry.magic;
    int horiz_index = static_cast<int>(static_cast<uint64_t>(horiz_product >> horiz_entry.shift));
    Bitboard horiz_attacks = kRookHorizAttacksTable[horiz_entry.offset + horiz_index];

    // Vertical attacks (N/S) with Magic Bitboards (FALLBACK)
    const magics::MagicEntry& vert_entry = magics::kRookVertMagics[sq];
    Bitboard vert_product = (blockers & vert_entry.mask) * vert_entry.magic;
    int vert_index = static_cast<int>(static_cast<uint64_t>(vert_product >> vert_entry.shift));
    Bitboard vert_attacks = kRookVertAttacksTable[vert_entry.offset + vert_index];

    return horiz_attacks | vert_attacks;
#endif
}

// PEXT/Magic Bitboard implementation for Bishop attacks
Bitboard Board::GetBishopAttacks(int sq, Bitboard blockers) const {
#if defined(__BMI2__)
    // Diagonal attacks (NE/SW) with PEXT
    const PextEntry& diag_entry = kBishopDiagPext[sq];
    uint64_t diag_index = pext_256(blockers, diag_entry.mask);
    Bitboard diag_attacks = kBishopDiagAttacksTable[diag_entry.offset + diag_index];

    // Anti-diagonal attacks (NW/SE) with PEXT
    const PextEntry& anti_diag_entry = kBishopAntiDiagPext[sq];
    uint64_t anti_diag_index = pext_256(blockers, anti_diag_entry.mask);
    Bitboard anti_diag_attacks = kBishopAntiDiagAttacksTable[anti_diag_entry.offset + anti_diag_index];
    
    return diag_attacks | anti_diag_attacks;
#else
    // Diagonal attacks (NE/SW) with Magic Bitboards (FALLBACK)
    const magics::MagicEntry& diag_entry = magics::kBishopDiagMagics[sq];
    Bitboard diag_product = (blockers & diag_entry.mask) * diag_entry.magic;
    int diag_index = static_cast<int>(static_cast<uint64_t>(diag_product >> diag_entry.shift));
    Bitboard diag_attacks = kBishopDiagAttacksTable[diag_entry.offset + diag_index];

    // Anti-diagonal attacks (NW/SE) with Magic Bitboards (FALLBACK)
    const magics::MagicEntry& anti_diag_entry = magics::kBishopAntiDiagMagics[sq];
    Bitboard anti_diag_product = (blockers & anti_diag_entry.mask) * anti_diag_entry.magic;
    int anti_diag_index = static_cast<int>(static_cast<uint64_t>(anti_diag_product >> anti_diag_entry.shift));
    Bitboard anti_diag_attacks = kBishopAntiDiagAttacksTable[anti_diag_entry.offset + anti_diag_index];

    return diag_attacks | anti_diag_attacks;
#endif
}

Bitboard Board::GetQueenAttacks(int sq, Bitboard blockers) const {
    return GetRookAttacks(sq, blockers) | GetBishopAttacks(sq, blockers);
}

Bitboard Board::GetAttackersBB(int sq, Team team) const {
    Bitboard attackers = Bitboard(0);
    if (sq < 0) return attackers;
    
    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    PlayerColor c1 = (team == RED_YELLOW) ? RED : BLUE;
    PlayerColor c2 = (team == RED_YELLOW) ? YELLOW : GREEN;

    // Pawns need special handling as their attack depends on their color
    // Find attackers by checking which enemy pawns would be attacked by our (partner) pawn from the target square
    attackers |= (kPawnAttacks[GetPartner(Player(c1)).GetColor()][sq] & piece_bitboards_[c1][PAWN]);
    attackers |= (kPawnAttacks[GetPartner(Player(c2)).GetColor()][sq] & piece_bitboards_[c2][PAWN]);
    
    // Knights & King
    attackers |= (kKnightAttacks[sq] & (piece_bitboards_[c1][KNIGHT] | piece_bitboards_[c2][KNIGHT]));
    attackers |= (kKingAttacks[sq] & (piece_bitboards_[c1][KING] | piece_bitboards_[c2][KING]));

    // Sliding pieces
    Bitboard rooks_and_queens = piece_bitboards_[c1][ROOK] | piece_bitboards_[c2][ROOK] |
                                piece_bitboards_[c1][QUEEN] | piece_bitboards_[c2][QUEEN];
    attackers |= (GetRookAttacks(sq, all_pieces) & rooks_and_queens);
    
    Bitboard bishops_and_queens = piece_bitboards_[c1][BISHOP] | piece_bitboards_[c2][BISHOP] |
                                  piece_bitboards_[c1][QUEEN] | piece_bitboards_[c2][QUEEN];
    attackers |= (GetBishopAttacks(sq, all_pieces) & bishops_and_queens);

    return attackers;
}

bool Board::IsAttackedByTeam(Team team, int sq) const {
    return !GetAttackersBB(sq, team).is_zero();
}

namespace {
void AddMovesFromBB(MoveBuffer& moves, int from_idx, Bitboard to_bb, const Board& board,
                    CastlingRights initial_cr = CastlingRights::kMissingRights,
                    CastlingRights final_cr = CastlingRights::kMissingRights) {
    BoardLocation from = IndexToLocation(from_idx);
    while (!to_bb.is_zero()) {
        int to_idx = to_bb.ctz();
        to_bb &= to_bb - 1;
        moves.emplace_back(from, IndexToLocation(to_idx), board.GetPiece(to_idx), initial_cr, final_cr);
    }
}
void AddPawnMovesFromBB(MoveBuffer& moves, int from_idx, Bitboard to_bb, PlayerColor color,
                        const Piece& capture, const BoardLocation& ep_loc, const Piece& ep_capture) {
    BoardLocation from = IndexToLocation(from_idx);
    Bitboard promotion_squares = to_bb & kPawnPromotionMask[color];
    Bitboard normal_squares = to_bb & ~promotion_squares;
    
    while (!normal_squares.is_zero()) {
        int to_idx = normal_squares.ctz();
        normal_squares &= normal_squares - 1;
        moves.emplace_back(from, IndexToLocation(to_idx), capture, ep_loc, ep_capture, NO_PIECE);
    }
    while (!promotion_squares.is_zero()) {
        int to_idx = promotion_squares.ctz();
        promotion_squares &= promotion_squares - 1;
        BoardLocation to = IndexToLocation(to_idx);
        moves.emplace_back(from, to, capture, ep_loc, ep_capture, KNIGHT);
        moves.emplace_back(from, to, capture, ep_loc, ep_capture, BISHOP);
        moves.emplace_back(from, to, capture, ep_loc, ep_capture, ROOK);
        moves.emplace_back(from, to, capture, ep_loc, ep_capture, QUEEN);
    }
}
}


void Board::GetPawnMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Team team = player.GetTeam();

    const Bitboard my_pawns = piece_bitboards_[color][PAWN];
    if (my_pawns.is_zero()) {
        return;
    }
    
    const Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    const Bitboard empty_squares = ~all_pieces;
    const Bitboard enemy_pieces = team_bitboards_[OtherTeam(team)];
    const Bitboard promotion_rank = kPawnPromotionMask[color];

    // =======================================================================
    // 1. Generate Pushes (Single and Double) - CORRECTED LOGIC
    // =======================================================================
    
    Bitboard single_pushes, double_pushes;
    
    switch (color) {
        case RED: {
            // 1a. All single pushes for ALL pawns
            single_pushes = shift<PUSH_N>(my_pawns) & empty_squares;
            
            // 1b. Double pushes, which can only originate from the start rank
            Bitboard pawns_on_start_rank = my_pawns & kPawnStartMask[RED];
            Bitboard first_step_of_double = shift<PUSH_N>(pawns_on_start_rank) & empty_squares;
            double_pushes = shift<PUSH_N>(first_step_of_double) & empty_squares;
            break;
        }
        case BLUE: {
            single_pushes = shift<PUSH_E>(my_pawns) & empty_squares;
            Bitboard pawns_on_start_rank = my_pawns & kPawnStartMask[BLUE];
            Bitboard first_step_of_double = shift<PUSH_E>(pawns_on_start_rank) & empty_squares;
            double_pushes = shift<PUSH_E>(first_step_of_double) & empty_squares;
            break;
        }
        case YELLOW: {
            single_pushes = shift<PUSH_S>(my_pawns) & empty_squares;
            Bitboard pawns_on_start_rank = my_pawns & kPawnStartMask[YELLOW];
            Bitboard first_step_of_double = shift<PUSH_S>(pawns_on_start_rank) & empty_squares;
            double_pushes = shift<PUSH_S>(first_step_of_double) & empty_squares;
            break;
        }
        case GREEN: {
            single_pushes = shift<PUSH_W>(my_pawns) & empty_squares;
            Bitboard pawns_on_start_rank = my_pawns & kPawnStartMask[GREEN];
            Bitboard first_step_of_double = shift<PUSH_W>(pawns_on_start_rank) & empty_squares;
            double_pushes = shift<PUSH_W>(first_step_of_double) & empty_squares;
            break;
        }
    }

    // --- The rest of the logic for adding moves remains the same ---

    // Add single push moves (excluding promotions, which are handled later)
    Bitboard single_targets = single_pushes & ~promotion_rank;
    while (!single_targets.is_zero()) {
        int to_idx = single_targets.ctz();
        single_targets &= single_targets - 1;
        int from_idx;
        switch(color) {
            case RED:    from_idx = to_idx - PUSH_N; break;
            case BLUE:   from_idx = to_idx - PUSH_E; break;
            case YELLOW: from_idx = to_idx - PUSH_S; break;
            case GREEN:  from_idx = to_idx - PUSH_W; break;
        }
        // A double push target can also be a single push target for another pawn.
        // We must ensure the pawn we are moving actually existed.
        if((my_pawns & IndexToBitboard(from_idx)).is_zero()) continue;
        moves.emplace_back(IndexToLocation(from_idx), IndexToLocation(to_idx), Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE);
    }
    
    // Add double push moves
    while (!double_pushes.is_zero()) {
        int to_idx = double_pushes.ctz();
        double_pushes &= double_pushes - 1;
        int from_idx;
        switch(color) {
            case RED:    from_idx = to_idx - PUSH_N - PUSH_N; break;
            case BLUE:   from_idx = to_idx - PUSH_E - PUSH_E; break;
            case YELLOW: from_idx = to_idx - PUSH_S - PUSH_S; break;
            case GREEN:  from_idx = to_idx - PUSH_W - PUSH_W; break;
        }
        moves.emplace_back(IndexToLocation(from_idx), IndexToLocation(to_idx), Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE);
    }
    
    // =======================================================================
    // 2. Generate Captures (This logic was correct)
    // =======================================================================
    
    constexpr int capture_offsets[4][2] = {
        { PUSH_NW, PUSH_NE }, { PUSH_NE, PUSH_SE },
        { PUSH_SW, PUSH_SE }, { PUSH_NW, PUSH_SW }
    };
    
    Bitboard captures1, captures2;
    switch (color) {
        case RED:
            captures1 = shift<capture_offsets[RED][0]>(my_pawns) & enemy_pieces;
            captures2 = shift<capture_offsets[RED][1]>(my_pawns) & enemy_pieces;
            break;
        case BLUE:
            captures1 = shift<capture_offsets[BLUE][0]>(my_pawns) & enemy_pieces;
            captures2 = shift<capture_offsets[BLUE][1]>(my_pawns) & enemy_pieces;
            break;
        case YELLOW:
            captures1 = shift<capture_offsets[YELLOW][0]>(my_pawns) & enemy_pieces;
            captures2 = shift<capture_offsets[YELLOW][1]>(my_pawns) & enemy_pieces;
            break;
        case GREEN:
            captures1 = shift<capture_offsets[GREEN][0]>(my_pawns) & enemy_pieces;
            captures2 = shift<capture_offsets[GREEN][1]>(my_pawns) & enemy_pieces;
            break;
    }

    Bitboard all_captures = (captures1 | captures2) & ~promotion_rank;
    while (!all_captures.is_zero()) {
        int to_idx = all_captures.ctz();
        Bitboard to_bb = IndexToBitboard(to_idx);
        all_captures &= all_captures - 1;

        Bitboard from_bb;
        switch (color) {
            case RED:    from_bb = (shift<-capture_offsets[RED][0]>(to_bb) | shift<-capture_offsets[RED][1]>(to_bb)) & my_pawns; break;
            case BLUE:   from_bb = (shift<-capture_offsets[BLUE][0]>(to_bb) | shift<-capture_offsets[BLUE][1]>(to_bb)) & my_pawns; break;
            case YELLOW: from_bb = (shift<-capture_offsets[YELLOW][0]>(to_bb) | shift<-capture_offsets[YELLOW][1]>(to_bb)) & my_pawns; break;
            case GREEN:  from_bb = (shift<-capture_offsets[GREEN][0]>(to_bb) | shift<-capture_offsets[GREEN][1]>(to_bb)) & my_pawns; break;
        }

        while (!from_bb.is_zero()) {
            int from_idx = from_bb.ctz();
            from_bb &= from_bb - 1;
            moves.emplace_back(IndexToLocation(from_idx), IndexToLocation(to_idx), GetPiece(to_idx), BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE);
        }
    }
    
    // =======================================================================
    // 3. Generate Promotions (This logic was correct)
    // =======================================================================
    
    Bitboard promo_pushes = single_pushes & promotion_rank;
    Bitboard promo_captures = (captures1 | captures2) & promotion_rank;

    // Promotion pushes...
    while (!promo_pushes.is_zero()) {
        int to_idx = promo_pushes.ctz();
        promo_pushes &= promo_pushes - 1;
        int from_idx;
        switch(color) {
            case RED:    from_idx = to_idx - PUSH_N; break;
            case BLUE:   from_idx = to_idx - PUSH_E; break;
            case YELLOW: from_idx = to_idx - PUSH_S; break;
            case GREEN:  from_idx = to_idx - PUSH_W; break;
        }
        BoardLocation from = IndexToLocation(from_idx);
        BoardLocation to = IndexToLocation(to_idx);
        moves.emplace_back(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, QUEEN);
        moves.emplace_back(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, ROOK);
        moves.emplace_back(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, BISHOP);
        moves.emplace_back(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, KNIGHT);
    }
    
    // Promotion captures...
    while (!promo_captures.is_zero()) {
        int to_idx = promo_captures.ctz();
        Bitboard to_bb = IndexToBitboard(to_idx);
        promo_captures &= promo_captures - 1;
        
        Bitboard from_bb;
         switch (color) {
            case RED:    from_bb = (shift<-capture_offsets[RED][0]>(to_bb) | shift<-capture_offsets[RED][1]>(to_bb)) & my_pawns; break;
            case BLUE:   from_bb = (shift<-capture_offsets[BLUE][0]>(to_bb) | shift<-capture_offsets[BLUE][1]>(to_bb)) & my_pawns; break;
            case YELLOW: from_bb = (shift<-capture_offsets[YELLOW][0]>(to_bb) | shift<-capture_offsets[YELLOW][1]>(to_bb)) & my_pawns; break;
            case GREEN:  from_bb = (shift<-capture_offsets[GREEN][0]>(to_bb) | shift<-capture_offsets[GREEN][1]>(to_bb)) & my_pawns; break;
        }

        Piece captured_piece = GetPiece(to_idx);
        BoardLocation to = IndexToLocation(to_idx);

        while (!from_bb.is_zero()) {
            int from_idx = from_bb.ctz();
            from_bb &= from_bb - 1;
            BoardLocation from = IndexToLocation(from_idx);
            moves.emplace_back(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, QUEEN);
            moves.emplace_back(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, ROOK);
            moves.emplace_back(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, BISHOP);
            moves.emplace_back(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, KNIGHT);
        }
    }
    
    // =======================================================================
    // 4. Generate En Passant (Optimized)
    // =======================================================================

    auto find_relevant_move = [&](PlayerColor opponent_color) -> const Move* {
        int turns_ago = (color - opponent_color + 4) % 4;
        if (turns_ago > 0 && moves_.size() >= turns_ago) {
            return &moves_[moves_.size() - turns_ago];
        }
        const auto& enp_move = enp_.enp_moves[opponent_color];
        return enp_move.has_value() ? &*enp_move : nullptr;
    };

    const PlayerColor opponents[2] = { GetNextPlayer(player).GetColor(), GetPreviousPlayer(player).GetColor() };
    
    // Define push directions for all colors for easy lookup
    constexpr int push_offsets[] = {PUSH_N, PUSH_E, PUSH_S, PUSH_W};
    
    for (const PlayerColor opponent_color : opponents) {
        const Move* opponent_last_move = find_relevant_move(opponent_color);

        if (opponent_last_move == nullptr || !opponent_last_move->Present()) {
            continue;
        }

        // Check if it was a 2-square straight pawn push
        const auto& move_from = opponent_last_move->From();
        const auto& move_to = opponent_last_move->To();
        Piece moved_piece = GetPiece(move_to);

        if (moved_piece.GetPieceType() != PAWN ||
            moved_piece.GetColor() != opponent_color ||
            opponent_last_move->ManhattanDistance() != 2 ||
            (move_from.GetRow() != move_to.GetRow() && move_from.GetCol() != move_to.GetCol())) {
            continue;
        }
        
        // --- THIS IS THE CORE OPTIMIZATION ---
        // The opponent pawn landed on 'move_to'.
        // Our capturing pawn must be on the square BEHIND 'move_to' (relative to our push direction).
        // We find this square by shifting 'move_to' backwards.
        int moved_to_idx = LocationToIndex(move_to);
        Bitboard capturer_square_bb;

        switch (color) {
            case RED:    capturer_square_bb = shift<-PUSH_N>(IndexToBitboard(moved_to_idx)); break;
            case BLUE:   capturer_square_bb = shift<-PUSH_E>(IndexToBitboard(moved_to_idx)); break;
            case YELLOW: capturer_square_bb = shift<-PUSH_S>(IndexToBitboard(moved_to_idx)); break;
            case GREEN:  capturer_square_bb = shift<-PUSH_W>(IndexToBitboard(moved_to_idx)); break;
        }

        // Now, check if one of our pawns is actually on that square.
        Bitboard capturer_pawn = capturer_square_bb & my_pawns;

        if (!capturer_pawn.is_zero()) {
            int our_pawn_idx = capturer_pawn.ctz();
            
            // Calculate the capture destination (the square the opponent pawn skipped)
            int moved_from_idx = LocationToIndex(move_from);
            int ep_capture_dest_idx = (moved_from_idx + moved_to_idx) / 2;
            
            // Create the move. No loops needed.
            moves.emplace_back(
                IndexToLocation(our_pawn_idx),
                IndexToLocation(ep_capture_dest_idx),
                GetPiece(ep_capture_dest_idx), // Standard capture (if any)
                move_to,                        // En-passant location
                moved_piece                     // En-passant capture
            );
        }
    }
}

void Board::GetKnightMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Bitboard knights = piece_bitboards_[color][KNIGHT];
    Bitboard friendly_pieces = team_bitboards_[player.GetTeam()];
    
    while(!knights.is_zero()) {
        int from_idx = knights.ctz();
        knights &= knights - 1;
        Bitboard attacks = kKnightAttacks[from_idx] & ~friendly_pieces;
        AddMovesFromBB(moves, from_idx, attacks, *this);
    }
}

void Board::GetRookMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Bitboard rooks = piece_bitboards_[color][ROOK];
    Bitboard friendly_pieces = team_bitboards_[player.GetTeam()];
    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];

    const auto& initial_cr = castling_rights_[color];

    while(!rooks.is_zero()) {
        int from_idx = rooks.ctz();
        rooks &= rooks - 1;
        
        CastlingRights final_cr = initial_cr;
        if(initial_cr.Present()){
            if(from_idx == kInitialRookSq[color][KINGSIDE] && initial_cr.Kingside()){
                final_cr = CastlingRights(false, initial_cr.Queenside());
            } else if (from_idx == kInitialRookSq[color][QUEENSIDE] && initial_cr.Queenside()){
                final_cr = CastlingRights(initial_cr.Kingside(), false);
            }
        }
        
        Bitboard attacks = GetRookAttacks(from_idx, all_pieces) & ~friendly_pieces;
        AddMovesFromBB(moves, from_idx, attacks, *this, initial_cr, final_cr.Present() ? final_cr : CastlingRights::kMissingRights);
    }
}

void Board::GetBishopMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Bitboard bishops = piece_bitboards_[color][BISHOP];
    Bitboard friendly_pieces = team_bitboards_[player.GetTeam()];
    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];

    while(!bishops.is_zero()) {
        int from_idx = bishops.ctz();
        bishops &= bishops - 1;
        Bitboard attacks = GetBishopAttacks(from_idx, all_pieces) & ~friendly_pieces;
        AddMovesFromBB(moves, from_idx, attacks, *this);
    }
}

void Board::GetQueenMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Bitboard queens = piece_bitboards_[color][QUEEN];
    Bitboard friendly_pieces = team_bitboards_[player.GetTeam()];
    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];

    while(!queens.is_zero()) {
        int from_idx = queens.ctz();
        queens &= queens - 1;
        Bitboard attacks = GetQueenAttacks(from_idx, all_pieces) & ~friendly_pieces;
        AddMovesFromBB(moves, from_idx, attacks, *this);
    }
}

void Board::GetKingMoves2(MoveBuffer& moves, const Player& player) const {
    PlayerColor color = player.GetColor();
    Bitboard king = piece_bitboards_[color][KING];
    if (king.is_zero()) return;

    int from_idx = king.ctz();
    Bitboard friendly_pieces = team_bitboards_[player.GetTeam()];
    const auto& initial_cr = castling_rights_[color];
    CastlingRights final_cr(false, false);
    
    Bitboard attacks = kKingAttacks[from_idx] & ~friendly_pieces;
    AddMovesFromBB(moves, from_idx, attacks, *this, initial_cr, final_cr);

    // Castling
    if (initial_cr.Present() && !IsAttackedByTeam(OtherTeam(player.GetTeam()), from_idx)) {
        Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
        Team enemy_team = OtherTeam(player.GetTeam());
        BoardLocation king_from_loc = IndexToLocation(from_idx);

        // KINGSIDE
        if (initial_cr.Kingside() && (all_pieces & kCastlingEmptyMask[color][KINGSIDE]).is_zero()) {
            Bitboard attack_mask = kCastlingAttackMask[color][KINGSIDE];
            
            bool is_safe = true;
            while(!attack_mask.is_zero()){
                int sq = attack_mask.ctz();
                attack_mask &= (attack_mask-1);
                if(IsAttackedByTeam(enemy_team, sq)){
                    is_safe = false;
                    break;
                }
            }

            if(is_safe){
                BoardLocation king_to_loc, rook_from_loc, rook_to_loc;
                rook_from_loc = IndexToLocation(kInitialRookSq[color][KINGSIDE]);

                switch(color) {
                    case RED:    king_to_loc = king_from_loc.Relative(0, 2); rook_to_loc = king_from_loc.Relative(0, 1); break;
                    case BLUE:   king_to_loc = king_from_loc.Relative(2, 0); rook_to_loc = king_from_loc.Relative(1, 0); break;
                    case YELLOW: king_to_loc = king_from_loc.Relative(0, -2); rook_to_loc = king_from_loc.Relative(0, -1); break;
                    case GREEN:  king_to_loc = king_from_loc.Relative(-2, 0); rook_to_loc = king_from_loc.Relative(-1, 0); break;
                }
                
                SimpleMove rook_move(rook_from_loc, rook_to_loc);
                moves.emplace_back(king_from_loc, king_to_loc, rook_move, initial_cr, final_cr);
            }
        }

        // QUEENSIDE
        if (initial_cr.Queenside() && (all_pieces & kCastlingEmptyMask[color][QUEENSIDE]).is_zero()) {
            Bitboard attack_mask = kCastlingAttackMask[color][QUEENSIDE];
            
            bool is_safe = true;
            while(!attack_mask.is_zero()){
                int sq = attack_mask.ctz();
                attack_mask &= (attack_mask-1);
                if(IsAttackedByTeam(enemy_team, sq)){
                    is_safe = false;
                    break;
                }
            }
            
            if(is_safe){
                BoardLocation king_to_loc, rook_from_loc, rook_to_loc;
                rook_from_loc = IndexToLocation(kInitialRookSq[color][QUEENSIDE]);
                
                switch(color) {
                    case RED:    king_to_loc = king_from_loc.Relative(0, -2); rook_to_loc = king_from_loc.Relative(0, -1); break;
                    case BLUE:   king_to_loc = king_from_loc.Relative(-2, 0); rook_to_loc = king_from_loc.Relative(-1, 0); break;
                    case YELLOW: king_to_loc = king_from_loc.Relative(0, 2); rook_to_loc = king_from_loc.Relative(0, 1); break;
                    case GREEN:  king_to_loc = king_from_loc.Relative(2, 0); rook_to_loc = king_from_loc.Relative(1, 0); break;
                }

                SimpleMove rook_move(rook_from_loc, rook_to_loc);
                moves.emplace_back(king_from_loc, king_to_loc, rook_move, initial_cr, final_cr);
            }
        }
    }
}

size_t Board::GetPseudoLegalMoves2(Move* buffer, size_t limit) {
    MoveBuffer move_buffer;
    move_buffer.buffer = buffer;
    move_buffer.limit = limit;
    move_buffer.pos = 0; // Ensure buffer starts at 0

    Player player = GetTurn();
    GetPawnMoves2(move_buffer, player);
    GetKnightMoves2(move_buffer, player);
    GetBishopMoves2(move_buffer, player);
    GetRookMoves2(move_buffer, player);
    GetQueenMoves2(move_buffer, player);
    GetKingMoves2(move_buffer, player);

    return move_buffer.pos;
}

void Board::MakeMove(const Move& move) {
    const Player player = turn_;
    const BoardLocation from = move.From();
    const BoardLocation to = move.To();

    // Standard capture
    if (move.IsStandardCapture()) {
        RemovePiece(to);
    }
    
    // Move the piece
    if (move.GetPromotionPieceType() != NO_PIECE) {
        RemovePiece(from);
        SetPiece(to, Piece(player.GetColor(), move.GetPromotionPieceType()));
    } else {
        MovePiece(from, to);
    }

    // En-passant capture
    if (move.GetEnpassantLocation().Present()) {
        RemovePiece(move.GetEnpassantLocation());
    }

    // Castling rook move
    if (move.GetRookMove().Present()) {
        SimpleMove rook_move = move.GetRookMove();
        MovePiece(rook_move.From(), rook_move.To());
    }
    
    // Update castling rights
    if (move.GetCastlingRights().Present()) {
        castling_rights_[player.GetColor()] = move.GetCastlingRights();
    }
    
    int t = static_cast<int>(turn_.GetColor());
    UpdateTurnHash(t);
    turn_ = GetNextPlayer(turn_);
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    moves_.push_back(move);
}

void Board::UndoMove() {
    assert(!moves_.empty());
    const Move& move = moves_.back();
    
    Player turn_before = GetPreviousPlayer(turn_);

    // Update turn first to match state before the move
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));
    turn_ = turn_before;
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    const BoardLocation& to = move.To();
    const BoardLocation& from = move.From();
    
    // Restore castling rights
    if (move.GetInitialCastlingRights().Present()) {
        castling_rights_[turn_before.GetColor()] = move.GetInitialCastlingRights();
    }

    // Undo castling rook move
    if (move.GetRookMove().Present()) {
        SimpleMove rook_move = move.GetRookMove();
        MovePiece(rook_move.To(), rook_move.From());
    }

    // Move piece back
    if (move.GetPromotionPieceType() != NO_PIECE) {
        RemovePiece(to);
        SetPiece(from, Piece(turn_before.GetColor(), PAWN));
    } else {
        MovePiece(to, from);
    }

    // Restore captured piece
    if (move.IsStandardCapture()) {
        SetPiece(to, move.GetStandardCapture());
    }

    // Restore en-passant capture
    if (move.GetEnpassantLocation().Present()) {
        SetPiece(move.GetEnpassantLocation(), move.GetEnpassantCapture());
    }
    
    moves_.pop_back();
}

GameResult Board::GetGameResult() {
  if (GetKingLocation(turn_.GetColor()).Missing()) {
    return turn_.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
  }
  Player player = turn_;

  size_t num_moves = GetPseudoLegalMoves2(move_buffer_2_, kInternalMoveBufferSize);
  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = move_buffer_2_[i];
    MakeMove(move);
    GameResult king_capture_result = CheckWasLastMoveKingCapture();
    if (king_capture_result != IN_PROGRESS) {
      UndoMove();
      return king_capture_result;
    }
    bool legal = !IsKingInCheck(player); // Check if the move was legal
    UndoMove();
    if (legal) {
      return IN_PROGRESS; // <-- Exit immediately on the FIRST legal move.
    }
  }
  if (!IsKingInCheck(player)) {
    return STALEMATE;
  }
  return player.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
}

bool Board::IsKingInCheck(const Player& player) const {
  const auto king_location = GetKingLocation(player.GetColor());
  if (king_location.Missing()) {
    return true; // A missing king is a lost king
  }
  return IsAttackedByTeam(OtherTeam(player.GetTeam()), LocationToIndex(king_location));
}

bool Board::IsKingInCheck(Team team) const {
  if (team == RED_YELLOW) {
    return IsKingInCheck(Player(RED)) || IsKingInCheck(Player(YELLOW));
  }
  return IsKingInCheck(Player(BLUE)) || IsKingInCheck(Player(GREEN));
}

GameResult Board::CheckWasLastMoveKingCapture() const {
  if (!moves_.empty()) {
    const auto& last_move = moves_.back();
    const auto capture = last_move.GetCapturePiece();
    if (capture.Present() && capture.GetPieceType() == KING) {
      return capture.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
    }
  }
  return IN_PROGRESS;
}

Team Board::TeamToPlay() const { return GetTeam(GetTurn().GetColor()); }
int Board::PieceEvaluation() const { return piece_evaluation_; }
int Board::PieceEvaluation(PlayerColor color) const { return player_piece_evaluations_[color]; }

void Board::SetPlayer(const Player& player) {
  UpdateTurnHash(static_cast<int>(turn_.GetColor()));
  turn_ = player;
  UpdateTurnHash(static_cast<int>(turn_.GetColor()));
}

void Board::MakeNullMove() {
  SetPlayer(GetNextPlayer(turn_));
}

void Board::UndoNullMove() {
  SetPlayer(GetPreviousPlayer(turn_));
}

int Board::MobilityEvaluation(const Player& player) {
    Player current_turn = turn_;
    turn_ = player;
    size_t num_moves = GetPseudoLegalMoves2(move_buffer_2_, kInternalMoveBufferSize);
    turn_ = current_turn;
    return (int)num_moves * kMobilityMultiplier;
}

int Board::MobilityEvaluation() {
  int mobility = 0;
  mobility += MobilityEvaluation(Player(RED));
  mobility -= MobilityEvaluation(Player(BLUE));
  mobility += MobilityEvaluation(Player(YELLOW));
  mobility -= MobilityEvaluation(Player(GREEN));
  return mobility;
}

std::shared_ptr<Board> Board::CreateStandardSetup() {
  std::unordered_map<BoardLocation, Piece> location_to_piece;
  std::unordered_map<Player, CastlingRights> castling_rights;

  std::vector<PieceType> piece_types = { ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK };
  std::vector<PlayerColor> player_colors = {RED, BLUE, YELLOW, GREEN};

  for (const PlayerColor& color : player_colors) {
    Player player(color);
    castling_rights[player] = CastlingRights(true, true);

    BoardLocation piece_location;
    int delta_row = 0, delta_col = 0;
    int pawn_offset_row = 0, pawn_offset_col = 0;

    switch (color) {
    case RED:    piece_location = BoardLocation(13, 3); delta_col = 1;  pawn_offset_row = -1; break;
    case BLUE:   piece_location = BoardLocation(3, 0);  delta_row = 1;  pawn_offset_col = 1;  break;
    case YELLOW: piece_location = BoardLocation(0, 10); delta_col = -1; pawn_offset_row = 1;  break;
    case GREEN:  piece_location = BoardLocation(10, 13);delta_row = -1; pawn_offset_col = -1; break;
    default:     assert(false); break;
    }

    for (const PieceType piece_type : piece_types) {
      BoardLocation pawn_location = piece_location.Relative(pawn_offset_row, pawn_offset_col);
      location_to_piece[piece_location] = Piece(player.GetColor(), piece_type);
      location_to_piece[pawn_location] = Piece(player.GetColor(), PAWN);
      piece_location = piece_location.Relative(delta_row, delta_col);
    }
  }

  return std::make_shared<Board>(Player(RED), std::move(location_to_piece), std::move(castling_rights));
}

int Move::ManhattanDistance() const {
  if (!from_.Present() || !to_.Present()) return 0;
  return std::abs(from_.GetRow() - to_.GetRow())
       + std::abs(from_.GetCol() - to_.GetCol());
}

namespace {
std::string ToStr(PieceType piece_type) {
  switch (piece_type) {
  case PAWN: return "P"; case ROOK: return "R"; case KNIGHT: return "N";
  case BISHOP: return "B"; case KING: return "K"; case QUEEN: return "Q";
  default: return " ";
  }
}
} // namespace

std::ostream& operator<<(std::ostream& os, const Board& board) {
  for (int r = 0; r < 14; r++) {
    for (int c = 0; c < 14; c++) {
        BoardLocation loc(r, c);
        if((kLegalSquares & IndexToBitboard(LocationToIndex(loc))).is_zero()) {
            os << " ";
        } else {
            const auto piece = board.GetPiece(loc);
            if (piece.Missing()) { os << "."; } 
            else { os << ToStr(piece.GetPieceType()); }
        }
    }
    os << std::endl;
  }
  os << "Turn: " << board.GetTurn() << ", Hash: " << std::hex << board.HashKey() << std::dec << std::endl;
  return os;
}

std::string BoardLocation::PrettyStr() const {
  if (!Present()) return "null";
  std::string s;
  s += ('a' + GetCol());
  s += std::to_string(14 - GetRow());
  return s;
}
std::string Move::PrettyStr() const {
  if(!Present()) return "null";
  std::string s = from_.PrettyStr() + "-" + to_.PrettyStr();
  if (GetPromotionPieceType() != NO_PIECE) {
    s += ToStr(GetPromotionPieceType());
  }
  return s;
}

bool Board::DiscoversCheck(const Move& move) const {
    const int from_sq = BitboardImpl::LocationToIndex(move.From());
    const int to_sq = BitboardImpl::LocationToIndex(move.To());
    const Team my_team = turn_.GetTeam();
    const Team enemy_team = OtherTeam(my_team);

    // 1. Identify all enemy kings and our own sliders (Bishops, Rooks, Queens).
    const PlayerColor e1 = (enemy_team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor e2 = (enemy_team == RED_YELLOW) ? YELLOW : GREEN;
    Bitboard enemy_kings = piece_bitboards_[e1][KING] | piece_bitboards_[e2][KING];

    const PlayerColor f1 = (my_team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor f2 = (my_team == RED_YELLOW) ? YELLOW : GREEN;
    const Bitboard my_sliders = piece_bitboards_[f1][BISHOP] | piece_bitboards_[f2][BISHOP] |
                                piece_bitboards_[f1][ROOK]   | piece_bitboards_[f2][ROOK] |
                                piece_bitboards_[f1][QUEEN]  | piece_bitboards_[f2][QUEEN];

    const Bitboard occupied = team_bitboards_[0] | team_bitboards_[1];

    // 2. Loop through each enemy king to see if the moving piece is pinned to it.
    while (!enemy_kings.is_zero()) {
        int king_sq = enemy_kings.ctz();
        enemy_kings &= enemy_kings - 1;

        // 3. Find all of our sliders that have a line-of-sight to this king.
        //    This tells us which of our pieces *could* be pinning something to this king.
        Bitboard potential_pinners = GetQueenAttacks(king_sq, occupied) & my_sliders;

        // 4. For each potential pinner, check if 'from_sq' is the only piece between it and the king.
        while (!potential_pinners.is_zero()) {
            int slider_sq = potential_pinners.ctz();
            potential_pinners &= potential_pinners - 1;
            
            // Check if the moving piece is the *only* piece on the line between the king and our slider.
            if ((BitboardImpl::kLineBetween[king_sq][slider_sq] & occupied) == BitboardImpl::IndexToBitboard(from_sq)) {
                // The piece at 'from_sq' is pinned to the enemy king by our slider.
                
                // A discovered check occurs if the piece moves OFF the pin line.
                // If 'to_sq' is also on the line, it's a move along the pin, not a discovery.
                if ((BitboardImpl::kLineBetween[king_sq][slider_sq] & BitboardImpl::IndexToBitboard(to_sq)).is_zero()) {
                    return true; // The move discovers check!
                }
            }
        }
    }
    
    // No discovered checks were found.
    return false;
}

bool Board::DeliversCheck(const Move& move) {
    if(!move.Present()) return false;
    Piece moved = GetPiece(move.From());
    int to_sq = LocationToIndex(move.To());
    
    Bitboard all_pieces = (team_bitboards_[0] | team_bitboards_[1]);
    Bitboard all_after_move = (all_pieces ^ IndexToBitboard(LocationToIndex(move.From()))) | IndexToBitboard(to_sq);
    if(move.IsCapture()) {
        all_after_move &= ~IndexToBitboard(LocationToIndex(move.GetStandardCapture().Present() ? move.To() : move.GetEnpassantLocation()));
    }
    
    Team enemy_team = OtherTeam(moved.GetTeam());
    PlayerColor e1 = enemy_team == RED_YELLOW ? RED : BLUE;
    PlayerColor e2 = enemy_team == RED_YELLOW ? YELLOW : GREEN;
    Bitboard enemy_kings = piece_bitboards_[e1][KING] | piece_bitboards_[e2][KING];
    
    Bitboard attacks;
    PieceType promotion_type = move.GetPromotionPieceType();
    PieceType piece_to_check = (promotion_type != NO_PIECE) ? promotion_type : moved.GetPieceType();

    switch(piece_to_check){
        case PAWN:   attacks = kPawnAttacks[moved.GetColor()][to_sq]; break;
        case KNIGHT: attacks = kKnightAttacks[to_sq]; break;
        case BISHOP: attacks = GetBishopAttacks(to_sq, all_after_move); break;
        case ROOK:   attacks = GetRookAttacks(to_sq, all_after_move); break;
        case QUEEN:  attacks = GetQueenAttacks(to_sq, all_after_move); break;
        case KING:   attacks = kKingAttacks[to_sq]; break;
        default: return false;
    }
    
    if(!(attacks & enemy_kings).is_zero()) return true;
    return DiscoversCheck(move);
}

bool Move::DeliversCheck(Board& board) {
  if (delivers_check_ < 0) {
    delivers_check_ = board.DeliversCheck(*this);
  }
  return delivers_check_;
}

int Move::ApproxSEE(const Board& board, const int* piece_evaluations) const {
  const auto capture = GetCapturePiece();
  if(!capture.Present()) return 0;
  const auto piece = board.GetPiece(From());
  if (!piece.Present()) return 0; // Should not happen in a valid move
  int captured_val = piece_evaluations[capture.GetPieceType()];
  int attacker_val = piece_evaluations[piece.GetPieceType()];
  return captured_val - attacker_val;
}

// ============================================================================
// Static Exchange Evaluation (SEE) Implementation - FINAL CORRECTED VERSION
// ============================================================================

/**
 * @brief Finds the square of the least valuable attacker for a given team on a square.
 *
 * This is a specialized helper for SEE that uses a dynamic 'occupied' bitboard
 * to correctly calculate sliding attacks, thereby accounting for discovered attacks
 * that arise during the simulated capture sequence.
 *
 * @param board The board state (used to get piece locations and attack patterns).
 * @param sq The target square where the exchange is happening.
 * @param team The attacking team.
 * @param occupied The simulated bitboard of all occupied squares.
 * @param out_type A reference to store the PieceType of the found attacker.
 * @return The square index of the least valuable attacker, or -1 if none exists.
 */
int GetLeastValuableAttacker(const Board& board, int sq, Team team, const Bitboard& occupied, PieceType& out_type) {
    // This function generates attacks from a team towards 'sq', considering 'occupied'
    // as the set of blockers for sliding pieces.
    Bitboard attackers = Bitboard(0);
    const PlayerColor c1 = (team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor c2 = (team == RED_YELLOW) ? YELLOW : GREEN;
    const Bitboard team_pieces = board.color_bitboards_[c1] | board.color_bitboards_[c2];

    // Non-sliding pieces: their attacks are independent of other pieces on the board.
    attackers |= (BitboardImpl::kPawnAttacks[GetPartner(Player(c1)).GetColor()][sq] & board.piece_bitboards_[c1][PAWN]);
    attackers |= (BitboardImpl::kPawnAttacks[GetPartner(Player(c2)).GetColor()][sq] & board.piece_bitboards_[c2][PAWN]);
    attackers |= (BitboardImpl::kKnightAttacks[sq] & (board.piece_bitboards_[c1][KNIGHT] | board.piece_bitboards_[c2][KNIGHT]));
    attackers |= (BitboardImpl::kKingAttacks[sq] & (board.piece_bitboards_[c1][KING] | board.piece_bitboards_[c2][KING]));

    // Sliding pieces: these attacks depend on the dynamic 'occupied' bitboard.
    const Bitboard rooks_and_queens = (board.piece_bitboards_[c1][ROOK] | board.piece_bitboards_[c2][ROOK] |
                                     board.piece_bitboards_[c1][QUEEN] | board.piece_bitboards_[c2][QUEEN]);
    attackers |= (board.GetRookAttacks(sq, occupied) & rooks_and_queens);
    
    const Bitboard bishops_and_queens = (board.piece_bitboards_[c1][BISHOP] | board.piece_bitboards_[c2][BISHOP] |
                                       board.piece_bitboards_[c1][QUEEN] | board.piece_bitboards_[c2][QUEEN]);
    attackers |= (board.GetBishopAttacks(sq, occupied) & bishops_and_queens);
    
    // We only care about attackers that are actually still on the board in our simulation.
    Bitboard valid_attackers = attackers & occupied;

    if (valid_attackers.is_zero()) {
        return -1;
    }

    // Find the cheapest piece type among the valid attackers.
    for (int pt_idx = PAWN; pt_idx <= KING; ++pt_idx) {
        const PieceType pt = static_cast<PieceType>(pt_idx);
        const Bitboard type_attackers = (board.piece_bitboards_[c1][pt] | board.piece_bitboards_[c2][pt]) & valid_attackers;
        if (!type_attackers.is_zero()) {
            out_type = pt;
            return type_attackers.ctz(); // Return the first one found (guaranteed to be cheapest).
        }
    }
    return -1; // Should be unreachable if valid_attackers is not zero.
}


/**
 * @brief Recursively calculates the gain of a capture sequence on a square.
 */
int SeeRecursive(const Board& board, const int piece_evaluations[6], int target_sq, Bitboard occupied, Team side_to_attack, int victim_value) {
    PieceType lva_type;
    int lva_sq = GetLeastValuableAttacker(board, target_sq, side_to_attack, occupied, lva_type);

    // Base case: If the current side has no more attackers for the square, they can't
    // recapture, so their gain from this point is 0.
    if (lva_sq == -1) {
        return 0;
    }

    // Simulate the recapture: the least valuable attacker is now considered "off the board"
    Bitboard next_occupied = occupied ^ BitboardImpl::IndexToBitboard(lva_sq);

    // The gain is the value of the piece captured, minus what the opponent gains in return.
    int gain = victim_value - SeeRecursive(board, piece_evaluations, target_sq, next_occupied, OtherTeam(side_to_attack), piece_evaluations[lva_type]);

    // A player will not continue a losing exchange ("stand pat" principle).
    return std::max(0, gain);
}

/**
 * @brief Main function to calculate the Static Exchange Evaluation for a move.
 * This is the public-facing entry point, which sets up and starts the recursion.
 * The board parameter should be const, as SEE is a static analysis and does not change the board.
 */
int StaticExchangeEvaluationCapture(const int piece_evaluations[6], const Board& board, const Move& move) {
    if (!move.IsCapture()) {
        return 0;
    }
    
    const Piece captured_piece = move.GetCapturePiece();
    const Piece attacker_piece = board.GetPiece(move.From());

    if (!captured_piece.Present() || !attacker_piece.Present()) {
        return 0; // Should not happen for a valid capture
    }

    const int from_sq = BitboardImpl::LocationToIndex(move.From());
    const int to_sq = BitboardImpl::LocationToIndex(move.To());

    // The initial gain is simply the value of the piece we are capturing.
    const int initial_gain = piece_evaluations[captured_piece.GetPieceType()];
    
    // The attacker now becomes the victim for the opponent's first recapture.
    const int new_victim_value = piece_evaluations[attacker_piece.GetPieceType()];
    
    // === ROBUST BOARD STATE SIMULATION ===
    // This logic now perfectly mirrors what a MakeMove/UndoMove pair would do,
    // handling both standard and en-passant captures correctly.

    Bitboard occupied_before_move = board.team_bitboards_[RED_YELLOW] | board.team_bitboards_[BLUE_GREEN];
    
    // Step 1: Simulate the attacker moving from its original square to the destination.
    // This leaves the 'from' square empty and the 'to' square occupied by the attacker.
    Bitboard occupied_after_move = (occupied_before_move & ~BitboardImpl::IndexToBitboard(from_sq)) | BitboardImpl::IndexToBitboard(to_sq);

    // Step 2: If it was an en-passant capture, we must also explicitly remove the captured pawn.
    if (move.GetEnpassantLocation().Present()) {
        int ep_captured_sq = BitboardImpl::LocationToIndex(move.GetEnpassantLocation());
        occupied_after_move &= ~BitboardImpl::IndexToBitboard(ep_captured_sq);
    }
    
    const Team opponent_team = OtherTeam(board.GetTurn().GetTeam());

    // Calculate what the opponent can gain from this new, correctly simulated board state.
    int opponent_gain = SeeRecursive(
        board,
        piece_evaluations,
        to_sq,
        occupied_after_move,
        opponent_team,
        new_victim_value
    );

    return initial_gain - opponent_gain;
}

int Move::SEE(Board& board, const int* piece_evaluations) {
  if (see_ == kSeeNotSet) {
    see_ = StaticExchangeEvaluationCapture(piece_evaluations, board, *this);
  }
  return see_;
}

namespace {
std::string ToStr(PlayerColor color) {
  switch (color) {
  case RED:
    return "RED";
  case BLUE:
    return "BLUE";
  case YELLOW:
    return "YELLOW";
  case GREEN:
    return "GREEN";
  default:
    return "UNINITIALIZED_PLAYER";
  }
}
} // namespace

std::ostream& operator<<(
    std::ostream& os, const Piece& piece) {
  if (!piece.Present()) {
      os << "NoPiece";
      return os;
  }
  os << ToStr(piece.GetColor()) << "(" << ToStr(piece.GetPieceType()) << ")";
  return os;
}

std::ostream& operator<<(
    std::ostream& os, const PlacedPiece& placed_piece) {
  os << placed_piece.GetPiece() << "@" << placed_piece.GetLocation();
  return os;
}

std::ostream& operator<<(
    std::ostream& os, const Player& player) {
  os << "Player(" << ToStr(player.GetColor()) << ")";
  return os;
}

std::ostream& operator<<(
    std::ostream& os, const BoardLocation& location) {
  if (!location.Present()) {
      os << "Loc(null)";
      return os;
  }
  os << "Loc(" << (int)location.GetRow() << ", " << (int)location.GetCol() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Move& move) {
  if (!move.Present()) {
      os << "Move(null)";
      return os;
  }
  os << "Move(" << move.From() << " -> " << move.To() << ")";
  return os;
}

}  // namespace chess