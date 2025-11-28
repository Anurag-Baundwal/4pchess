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
#include <chrono>
#include <cstring> // For memcpy

// For PEXT/PDEP intrinsics
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

// OPTIMIZATION: Full line mask for O(1) alignment
Bitboard kLineMask[kNumSquares][kNumSquares];

Bitboard kCastlingEmptyMask[4][2]; // [color][side]
Bitboard kCastlingAttackMask[4][2]; // [color][side]
Bitboard kBackRankMasks[4];
Bitboard kSecondRankMasks[4];
Bitboard kCentralMask;
int kInitialRookSq[4][2];

// --- PEXT Bitboard Data ---
struct PextEntry {
    Bitboard mask;
    uint32_t offset;
};

PextEntry kRookHorizPext[kNumSquares];
PextEntry kRookVertPext[kNumSquares];
PextEntry kBishopDiagPext[kNumSquares];
PextEntry kBishopAntiDiagPext[kNumSquares];

std::vector<Bitboard> kRookHorizAttacksTable;
std::vector<Bitboard> kRookVertAttacksTable;
std::vector<Bitboard> kBishopDiagAttacksTable;
std::vector<Bitboard> kBishopAntiDiagAttacksTable;

// OPTIMIZATION: Raw pointers to attack tables
const Bitboard* g_RookHorizAttacksRaw = nullptr;
const Bitboard* g_RookVertAttacksRaw = nullptr;
const Bitboard* g_BishopDiagAttacksRaw = nullptr;
const Bitboard* g_BishopAntiDiagAttacksRaw = nullptr;

// --- Magic Bitboard Data ---
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
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//                      RUNTIME TABLE GENERATION
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
namespace TableGenerator {

Bitboard get_ray_attack(int sq, RayDirection dir, const Bitboard& blockers) {
    Bitboard ray = kRayAttacks[sq][dir];
    Bitboard b = ray & blockers;
    if (!b.is_zero()) {
        int blocker_idx;
        bool is_increasing_ray = (dir == D_S || dir == D_E || dir == D_SE || dir == D_SW);
        if (is_increasing_ray) blocker_idx = b.ctz();
        else blocker_idx = 255 - b.clz();
        return ray & ~kRayAttacks[blocker_idx][dir];
    }
    return ray;
}

Bitboard calculate_attacks_for_rays(int sq, Bitboard blockers, RayDirection d1, RayDirection d2) {
    return get_ray_attack(sq, d1, blockers) | get_ray_attack(sq, d2, blockers);
}

Bitboard get_attack_mask(int sq, RayDirection d1, RayDirection d2) {
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
    return mask;
}

#if defined(__BMI2__)
Bitboard pdep_256(uint64_t source, Bitboard mask) {
    Bitboard result = {};
    uint64_t current_source = source;
    for (int i = 0; i < 4; ++i) {
        int bits_in_limb_mask = __builtin_popcountll(mask.limbs[i]);
        uint64_t source_chunk = current_source & ((1ULL << bits_in_limb_mask) - 1);
        result.limbs[i] = _pdep_u64(source_chunk, mask.limbs[i]);
        current_source >>= bits_in_limb_mask;
    }
    return result;
}

void GeneratePextTablesForSquare(int sq, RayDirection d1, RayDirection d2, PextEntry& pext_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset) {
    if ((kLegalSquares & IndexToBitboard(sq)).is_zero()) {
        pext_entry = {Bitboard(0), 0};
        return;
    }
    Bitboard mask = get_attack_mask(sq, d1, d2);
    pext_entry.mask = mask;
    int bits = mask.popcount();
    uint64_t num_configs = 1ULL << bits;
    
    pext_entry.offset = current_offset;
    uint32_t initial_size = global_attack_table.size();
    global_attack_table.resize(initial_size + num_configs);

    for (uint64_t i = 0; i < num_configs; ++i) {
        Bitboard blockers = pdep_256(i, mask);
        global_attack_table[initial_size + i] = calculate_attacks_for_rays(sq, blockers, d1, d2);
    }
    current_offset += num_configs;
}

void GeneratePextTables() {
    uint32_t current_offset;
    auto gen_pext_set = [&](const char* name, PextEntry entries[], std::vector<Bitboard>& attacks, RayDirection d1, RayDirection d2){
        current_offset = 0;
        attacks.clear();
        attacks.reserve(140000); 
        for (int sq = 0; sq < kNumSquares; ++sq) {
            GeneratePextTablesForSquare(sq, d1, d2, entries[sq], attacks, current_offset);
        }
        attacks.shrink_to_fit();
    };
    gen_pext_set("Vertical Rooks", kRookVertPext, kRookVertAttacksTable, D_N, D_S);
    gen_pext_set("Horizontal Rooks", kRookHorizPext, kRookHorizAttacksTable, D_E, D_W);
    gen_pext_set("Diagonal Bishops", kBishopDiagPext, kBishopDiagAttacksTable, D_NE, D_SW);
    gen_pext_set("Anti-Diagonal Bishops", kBishopAntiDiagPext, kBishopAntiDiagAttacksTable, D_NW, D_SE);
}
#else
// Fallback logic for non-BMI2 (Magic Bitboards)
Bitboard pdep_fallback(uint64_t index, Bitboard mask) {
    Bitboard result(0);
    Bitboard temp_mask = mask;
    for (uint64_t i = index; i != 0; i >>= 1) {
        int lsb_idx = temp_mask.ctz();
        temp_mask &= temp_mask - 1; 
        if (i & 1) result |= IndexToBitboard(lsb_idx);
    }
    return result;
}
std::mt19937_64 rng(0xBADF00D5EED); 
Bitboard generate_magic_candidate() {
    return Bitboard(rng(), rng(), rng(), rng()) & Bitboard(rng(), rng(), rng(), rng()) & Bitboard(rng(), rng(), rng(), rng());
}
void FindMagicForSquare(int sq, RayDirection d1, RayDirection d2, magics::MagicEntry& magic_entry, std::vector<Bitboard>& global_attack_table, uint32_t& current_offset) {
    if ((kLegalSquares & IndexToBitboard(sq)).is_zero()) {
        magic_entry = {Bitboard(0), Bitboard(0), 0, 0};
        return;
    }
    Bitboard mask = get_attack_mask(sq, d1, d2);
    int num_mask_bits = mask.popcount();
    uint64_t num_configs = 1ULL << num_mask_bits;
    magic_entry.mask = mask;
    magic_entry.shift = 256 - num_mask_bits;
    std::vector<Bitboard> local_attacks(num_configs);
    std::vector<Bitboard> blockers(num_configs);
    for (uint64_t i = 0; i < num_configs; ++i) {
        blockers[i] = pdep_fallback(i, mask);
        local_attacks[i] = calculate_attacks_for_rays(sq, blockers[i], d1, d2);
    }
    std::vector<Bitboard> used_attacks(num_configs);
    for (int attempts = 0; attempts < 10000000; ++attempts) {
        Bitboard magic = generate_magic_candidate();
        if (((mask * magic) >> (256-8)).popcount() < 6) continue;
        magic_entry.magic = magic;
        std::fill(used_attacks.begin(), used_attacks.end(), Bitboard(0));
        bool collision = false;
        for (uint64_t i = 0; i < num_configs; ++i) {
            Bitboard product = blockers[i] * magic;
            int index = static_cast<int>(static_cast<uint64_t>(product >> magic_entry.shift));
            if (used_attacks[index].is_zero()) used_attacks[index] = local_attacks[i];
            else if (used_attacks[index] != local_attacks[i]) { collision = true; break; }
        }
        if (!collision) {
            magic_entry.offset = current_offset;
            global_attack_table.insert(global_attack_table.end(), used_attacks.begin(), used_attacks.end());
            current_offset += num_configs;
            return;
        }
    }
    exit(1);
}
void GenerateMagicTables() {
    uint32_t current_offset;
    auto gen_magic_set = [&](const char* name, magics::MagicEntry entries[], std::vector<Bitboard>& attacks, RayDirection d1, RayDirection d2){
        current_offset = 0;
        attacks.clear();
        attacks.reserve(140000);
        for (int sq = 0; sq < kNumSquares; ++sq) FindMagicForSquare(sq, d1, d2, entries[sq], attacks, current_offset);
        attacks.shrink_to_fit();
    };
    gen_magic_set("Vertical Rooks", magics::kRookVertMagics, kRookVertAttacksTable, D_N, D_S);
    gen_magic_set("Horizontal Rooks", magics::kRookHorizMagics, kRookHorizAttacksTable, D_E, D_W);
    gen_magic_set("Diagonal Bishops", magics::kBishopDiagMagics, kBishopDiagAttacksTable, D_NE, D_SW);
    gen_magic_set("Anti-Diagonal Bishops", magics::kBishopAntiDiagMagics, kBishopAntiDiagAttacksTable, D_NW, D_SE);
}
#endif
} // namespace TableGenerator

void InitBitboards() {
    static bool is_initialized = false;
    if (is_initialized) return;

    for (int r_14 = 0; r_14 < 14; ++r_14) {
        for (int c_14 = 0; c_14 < 14; ++c_14) {
             if (!((r_14 < 3 && (c_14 < 3 || c_14 > 10)) || (r_14 > 10 && (c_14 < 3 || c_14 > 10)))) {
                kLegalSquares |= IndexToBitboard(LocationToIndex(BoardLocation(r_14, c_14)));
            }
        }
    }

    int push_offsets[] = {-kBoardWidth, 1, kBoardWidth, -1};
    const int pawn_capture_deltas[4][2][2] = {
        { {-1, -1}, {-1,  1} }, { {-1,  1}, { 1,  1} },
        { { 1, -1}, { 1,  1} }, { {-1, -1}, { 1, -1} }
    };

    for (int r_14 = 0; r_14 < 14; ++r_14) {
        for (int c_14 = 0; c_14 < 14; ++c_14) {
            BoardLocation from_loc(r_14, c_14);
            if (!(kLegalSquares & IndexToBitboard(LocationToIndex(from_loc)))) continue;
            int idx = LocationToIndex(from_loc);
            
            if (r_14 == 12) kPawnStartMask[RED] |= IndexToBitboard(idx);
            if (c_14 == 1)  kPawnStartMask[BLUE] |= IndexToBitboard(idx);
            if (r_14 == 1)  kPawnStartMask[YELLOW] |= IndexToBitboard(idx);
            if (c_14 == 12) kPawnStartMask[GREEN] |= IndexToBitboard(idx);
            
            if (r_14 == 3)  kPawnPromotionMask[RED] |= IndexToBitboard(idx);
            if (c_14 == 10) kPawnPromotionMask[BLUE] |= IndexToBitboard(idx);
            if (r_14 == 10) kPawnPromotionMask[YELLOW] |= IndexToBitboard(idx);
            if (c_14 == 3)  kPawnPromotionMask[GREEN] |= IndexToBitboard(idx);

            for (int color = 0; color < 4; ++color) {
                BoardLocation to1 = from_loc.Relative(push_offsets[color] / kBoardWidth, push_offsets[color] % kBoardWidth);
                if ((kLegalSquares & IndexToBitboard(LocationToIndex(to1)))) {
                    kPawnSinglePush[color][idx] = IndexToBitboard(LocationToIndex(to1));
                    if (kPawnStartMask[color] & IndexToBitboard(idx)) {
                         BoardLocation to2 = to1.Relative(push_offsets[color] / kBoardWidth, push_offsets[color] % kBoardWidth);
                         if ((kLegalSquares & IndexToBitboard(LocationToIndex(to2)))) kPawnDoublePush[color][idx] = IndexToBitboard(LocationToIndex(to2));
                    }
                }
                for (int k = 0; k < 2; ++k) {
                    const int dr = pawn_capture_deltas[color][k][0];
                    const int dc = pawn_capture_deltas[color][k][1];
                    BoardLocation to_cap = from_loc.Relative(dr, dc);
                    if ((kLegalSquares & IndexToBitboard(LocationToIndex(to_cap)))) kPawnAttacks[color][idx] |= IndexToBitboard(LocationToIndex(to_cap));
                }
            }
        }
    }

    for (int i = 0; i < kNumSquares; ++i) {
        BoardLocation from = IndexToLocation(i);
        if (!from.Present() || !(kLegalSquares & IndexToBitboard(i))) continue;
        int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
        int dc[] = {-1, 1, -2, 2, -2, 2, -1, 1};
        for (int k = 0; k < 8; ++k) {
            BoardLocation to = from.Relative(dr[k], dc[k]);
            if ((kLegalSquares & IndexToBitboard(LocationToIndex(to)))) kKnightAttacks[i] |= IndexToBitboard(LocationToIndex(to));
        }
        for (int r = -1; r <= 1; ++r) {
            for (int c = -1; c <= 1; ++c) {
                if (r == 0 && c == 0) continue;
                BoardLocation to = from.Relative(r, c);
                if ((kLegalSquares & IndexToBitboard(LocationToIndex(to)))) kKingAttacks[i] |= IndexToBitboard(LocationToIndex(to));
            }
        }
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
                    // OPTIMIZATION: kLineMask
                    kLineMask[i][j] = kRayAttacks[i][d] | kRayAttacks[i][opposite_dir] | IndexToBitboard(i);
                    break;
                }
            }
        }
    }
    
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
        kCastlingEmptyMask[c][KINGSIDE] = kLineBetween[king_sq][kInitialRookSq[c][KINGSIDE]];
        kCastlingAttackMask[c][KINGSIDE] = IndexToBitboard(king_sq) | IndexToBitboard(king_sq + push_offsets[(c+1)%4]) | IndexToBitboard(king_sq + 2*push_offsets[(c+1)%4]);
        kCastlingEmptyMask[c][QUEENSIDE] = kLineBetween[king_sq][kInitialRookSq[c][QUEENSIDE]];
        kCastlingAttackMask[c][QUEENSIDE] = IndexToBitboard(king_sq) | IndexToBitboard(king_sq + push_offsets[(c+3)%4]) | IndexToBitboard(king_sq + 2*push_offsets[(c+3)%4]);
    }
    
    for (int c = 0; c < 14; ++c) {
        BoardLocation loc_r(13, (int8_t)c);
        if (loc_r.Present()) kBackRankMasks[RED] |= IndexToBitboard(LocationToIndex(loc_r));
        BoardLocation loc_y(0, (int8_t)c);
        if (loc_y.Present()) kBackRankMasks[YELLOW] |= IndexToBitboard(LocationToIndex(loc_y));
    }
    for (int r = 0; r < 14; ++r) {
        BoardLocation loc_b((int8_t)r, 0);
        if (loc_b.Present()) kBackRankMasks[BLUE] |= IndexToBitboard(LocationToIndex(loc_b));
        BoardLocation loc_g((int8_t)r, 13);
        if (loc_g.Present()) kBackRankMasks[GREEN] |= IndexToBitboard(LocationToIndex(loc_g));
    }

    for (int c = 0; c < 14; ++c) {
        BoardLocation loc_r(12, (int8_t)c); 
        if (loc_r.Present()) kSecondRankMasks[RED] |= IndexToBitboard(LocationToIndex(loc_r));
        BoardLocation loc_y(1, (int8_t)c); 
        if (loc_y.Present()) kSecondRankMasks[YELLOW] |= IndexToBitboard(LocationToIndex(loc_y));
    }
    for (int r = 0; r < 14; ++r) {
        BoardLocation loc_b((int8_t)r, 1); 
        if (loc_b.Present()) kSecondRankMasks[BLUE] |= IndexToBitboard(LocationToIndex(loc_b));
        BoardLocation loc_g((int8_t)r, 12); 
        if (loc_g.Present()) kSecondRankMasks[GREEN] |= IndexToBitboard(LocationToIndex(loc_g));
    }

    for (int r_14 = 3; r_14 <= 10; ++r_14) {
        for (int c_14 = 3; c_14 <= 10; ++c_14) {
            kCentralMask |= IndexToBitboard(LocationToIndex(BoardLocation(r_14, c_14)));
        }
    }
    
    #if defined(__BMI2__)
        TableGenerator::GeneratePextTables();
    #else
        TableGenerator::GenerateMagicTables();
    #endif

    // OPTIMIZATION: Capture pointers to generated tables
    g_RookHorizAttacksRaw = kRookHorizAttacksTable.data();
    g_RookVertAttacksRaw = kRookVertAttacksTable.data();
    g_BishopDiagAttacksRaw = kBishopDiagAttacksTable.data();
    g_BishopAntiDiagAttacksRaw = kBishopAntiDiagAttacksTable.data();

    is_initialized = true;
}

constexpr int PUSH_N = -kBoardWidth; 
constexpr int PUSH_E = 1;
constexpr int PUSH_S = kBoardWidth;  
constexpr int PUSH_W = -1;
constexpr int PUSH_NE = PUSH_N + PUSH_E; 
constexpr int PUSH_NW = PUSH_N + PUSH_W; 
constexpr int PUSH_SE = PUSH_S + PUSH_E; 
constexpr int PUSH_SW = PUSH_S + PUSH_W; 

template<int ShiftOffset>
inline Bitboard shift(Bitboard b) {
    if constexpr (ShiftOffset > 0) return b << ShiftOffset;
    else return b >> -ShiftOffset;
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

  checkers_ = Bitboard(0);
  for(int c=0; c<4; ++c) {
      blockers_for_king_[c] = Bitboard(0);
      pinners_[c] = Bitboard(0);
  }
  move_history_ptr_ = 0;
  safety_stack_ptr_ = 0;

  for (int color = 0; color < 4; color++) {
    castling_rights_[color] = CastlingRights(false, false);
    if (castling_rights.has_value()) {
      auto& cr = *castling_rights;
      Player pl(static_cast<PlayerColor>(color));
      auto it = cr.find(pl);
      if (it != cr.end()) castling_rights_[color] = it->second;
    }
  }
  if (enp.has_value()) enp_ = std::move(*enp);

  for (const auto& it : location_to_piece) SetPiece(it.first, it.second);
  
  std::mt19937_64 rng(958829);
  for (int color = 0; color < 4; color++) turn_hashes_[color] = rng();
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
    const Bitboard& king_bb = piece_bitboards_[color][KING];
    if (king_bb.is_zero()) return BoardLocation::kNoLocation;
    return IndexToLocation(king_bb.ctz());
}

#if defined(__BMI2__)
uint64_t pext_256(const Bitboard& source, const Bitboard& mask) {
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

// OPTIMIZATION: Raw pointer access
Bitboard Board::GetRookAttacks(int sq, const Bitboard& blockers) const {
#if defined(__BMI2__)
    const PextEntry& horiz_entry = kRookHorizPext[sq];
    uint64_t horiz_index = pext_256(blockers, horiz_entry.mask);
    Bitboard horiz_attacks = g_RookHorizAttacksRaw[horiz_entry.offset + horiz_index];

    const PextEntry& vert_entry = kRookVertPext[sq];
    uint64_t vert_index = pext_256(blockers, vert_entry.mask);
    Bitboard vert_attacks = g_RookVertAttacksRaw[vert_entry.offset + vert_index];
    return horiz_attacks | vert_attacks;
#else
    const magics::MagicEntry& horiz_entry = magics::kRookHorizMagics[sq];
    Bitboard horiz_product = (blockers & horiz_entry.mask) * horiz_entry.magic;
    int horiz_index = static_cast<int>(static_cast<uint64_t>(horiz_product >> horiz_entry.shift));
    Bitboard horiz_attacks = g_RookHorizAttacksRaw[horiz_entry.offset + horiz_index];

    const magics::MagicEntry& vert_entry = magics::kRookVertMagics[sq];
    Bitboard vert_product = (blockers & vert_entry.mask) * vert_entry.magic;
    int vert_index = static_cast<int>(static_cast<uint64_t>(vert_product >> vert_entry.shift));
    Bitboard vert_attacks = g_RookVertAttacksRaw[vert_entry.offset + vert_index];
    return horiz_attacks | vert_attacks;
#endif
}

Bitboard Board::GetBishopAttacks(int sq, const Bitboard& blockers) const {
#if defined(__BMI2__)
    const PextEntry& diag_entry = kBishopDiagPext[sq];
    uint64_t diag_index = pext_256(blockers, diag_entry.mask);
    Bitboard diag_attacks = g_BishopDiagAttacksRaw[diag_entry.offset + diag_index];

    const PextEntry& anti_diag_entry = kBishopAntiDiagPext[sq];
    uint64_t anti_diag_index = pext_256(blockers, anti_diag_entry.mask);
    Bitboard anti_diag_attacks = g_BishopAntiDiagAttacksRaw[anti_diag_entry.offset + anti_diag_index];
    return diag_attacks | anti_diag_attacks;
#else
    const magics::MagicEntry& diag_entry = magics::kBishopDiagMagics[sq];
    Bitboard diag_product = (blockers & diag_entry.mask) * diag_entry.magic;
    int diag_index = static_cast<int>(static_cast<uint64_t>(diag_product >> diag_entry.shift));
    Bitboard diag_attacks = g_BishopDiagAttacksRaw[diag_entry.offset + diag_index];

    const magics::MagicEntry& anti_diag_entry = magics::kBishopAntiDiagMagics[sq];
    Bitboard anti_diag_product = (blockers & anti_diag_entry.mask) * anti_diag_entry.magic;
    int anti_diag_index = static_cast<int>(static_cast<uint64_t>(anti_diag_product >> anti_diag_entry.shift));
    Bitboard anti_diag_attacks = g_BishopAntiDiagAttacksRaw[anti_diag_entry.offset + anti_diag_index];
    return diag_attacks | anti_diag_attacks;
#endif
}

Bitboard Board::GetQueenAttacks(int sq, const Bitboard& blockers) const {
    return GetRookAttacks(sq, blockers) | GetBishopAttacks(sq, blockers);
}

Bitboard Board::GetAttackersBB(int sq, Team team) const {
    Bitboard attackers = Bitboard(0);
    if (sq < 0) return attackers;
    
    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    PlayerColor c1 = (team == RED_YELLOW) ? RED : BLUE;
    PlayerColor c2 = (team == RED_YELLOW) ? YELLOW : GREEN;

    attackers |= (kPawnAttacks[GetPartner(Player(c1)).GetColor()][sq] & piece_bitboards_[c1][PAWN]);
    attackers |= (kPawnAttacks[GetPartner(Player(c2)).GetColor()][sq] & piece_bitboards_[c2][PAWN]);
    
    attackers |= (kKnightAttacks[sq] & (piece_bitboards_[c1][KNIGHT] | piece_bitboards_[c2][KNIGHT]));
    attackers |= (kKingAttacks[sq] & (piece_bitboards_[c1][KING] | piece_bitboards_[c2][KING]));

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

bool Board::AttackersToExist(int sq, const Bitboard& occupied, Team team) const {
    PlayerColor c1 = (team == RED_YELLOW) ? RED : BLUE;
    PlayerColor c2 = (team == RED_YELLOW) ? YELLOW : GREEN;

    if (!(BitboardImpl::kPawnAttacks[GetPartner(Player(c1)).GetColor()][sq] & piece_bitboards_[c1][PAWN]).is_zero()) return true;
    if (!(BitboardImpl::kPawnAttacks[GetPartner(Player(c2)).GetColor()][sq] & piece_bitboards_[c2][PAWN]).is_zero()) return true;
    
    if (!(BitboardImpl::kKnightAttacks[sq] & (piece_bitboards_[c1][KNIGHT] | piece_bitboards_[c2][KNIGHT])).is_zero()) return true;
    if (!(BitboardImpl::kKingAttacks[sq] & (piece_bitboards_[c1][KING] | piece_bitboards_[c2][KING])).is_zero()) return true;

    Bitboard rooks_queens = piece_bitboards_[c1][ROOK] | piece_bitboards_[c2][ROOK] |
                            piece_bitboards_[c1][QUEEN] | piece_bitboards_[c2][QUEEN];
    if (!(GetRookAttacks(sq, occupied) & rooks_queens).is_zero()) return true;
    
    Bitboard bishops_queens = piece_bitboards_[c1][BISHOP] | piece_bitboards_[c2][BISHOP] |
                              piece_bitboards_[c1][QUEEN] | piece_bitboards_[c2][QUEEN];
    if (!(GetBishopAttacks(sq, occupied) & bishops_queens).is_zero()) return true;

    return false;
}

void Board::RefreshKingSafety() {
    PlayerColor us = turn_.GetColor();
    Team us_team = turn_.GetTeam();
    
    BoardLocation king_loc = GetKingLocation(us);
    if (!king_loc.Present()) {
        checkers_ = Bitboard::max(); 
        blockers_for_king_[us] = Bitboard(0); 
        return;
    }

    int king_sq = LocationToIndex(king_loc);
    checkers_ = GetAttackersBB(king_sq, OtherTeam(us_team));

    UpdateSliderBlockers(us); 
}

void Board::UpdateSliderBlockers(PlayerColor c) {
    blockers_for_king_[c] = Bitboard(0);
    pinners_[c] = Bitboard(0);

    BoardLocation king_loc = GetKingLocation(c);
    if (!king_loc.Present()) return;
    int king_sq = LocationToIndex(king_loc);

    Team team = GetTeam(c);
    Team enemy_team = OtherTeam(team);
    
    PlayerColor e1 = (enemy_team == RED_YELLOW) ? RED : BLUE;
    PlayerColor e2 = (enemy_team == RED_YELLOW) ? YELLOW : GREEN;
    
    Bitboard enemy_rooks = (piece_bitboards_[e1][ROOK] | piece_bitboards_[e2][ROOK] |
                            piece_bitboards_[e1][QUEEN] | piece_bitboards_[e2][QUEEN]);
    
    Bitboard enemy_bishops = (piece_bitboards_[e1][BISHOP] | piece_bitboards_[e2][BISHOP] |
                              piece_bitboards_[e1][QUEEN] | piece_bitboards_[e2][QUEEN]);

    Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    
    static const Bitboard kEmpty(0);

    Bitboard ortho_candidates = GetRookAttacks(king_sq, kEmpty) & enemy_rooks;
    while (!ortho_candidates.is_zero()) {
        int sniper_sq = ortho_candidates.ctz();
        ortho_candidates &= (ortho_candidates - 1);
        Bitboard between = BitboardImpl::kLineBetween[king_sq][sniper_sq] & all_pieces;
        if (!between.is_zero() && (between & (between - 1)).is_zero()) {
            if (!(between & team_bitboards_[team]).is_zero()) {
                blockers_for_king_[c] |= between;
                pinners_[c] |= BitboardImpl::IndexToBitboard(sniper_sq);
            }
        }
    }

    Bitboard diag_candidates = GetBishopAttacks(king_sq, kEmpty) & enemy_bishops;
    while (!diag_candidates.is_zero()) {
        int sniper_sq = diag_candidates.ctz();
        diag_candidates &= (diag_candidates - 1);
        Bitboard between = BitboardImpl::kLineBetween[king_sq][sniper_sq] & all_pieces;
        if (!between.is_zero() && (between & (between - 1)).is_zero()) {
            if (!(between & team_bitboards_[team]).is_zero()) {
                blockers_for_king_[c] |= between;
                pinners_[c] |= BitboardImpl::IndexToBitboard(sniper_sq);
            }
        }
    }
}

bool Board::IsLegal(const Move& move) const {
    if (!move.Present()) return false;
    PlayerColor us = turn_.GetColor();
    int from_sq = LocationToIndex(move.From());
    int to_sq = LocationToIndex(move.To());
    int king_sq = LocationToIndex(GetKingLocation(us));
    
    if (move.GetEnpassantLocation().Present()) {
        BoardLocation cap_loc = move.GetEnpassantLocation();
        int cap_sq = LocationToIndex(cap_loc);
        Bitboard occupied = (team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN]);
        occupied &= ~BitboardImpl::IndexToBitboard(from_sq);
        occupied &= ~BitboardImpl::IndexToBitboard(cap_sq);
        occupied |= BitboardImpl::IndexToBitboard(to_sq);

        Team enemy_team = OtherTeam(turn_.GetTeam());
        PlayerColor e1 = (enemy_team == RED_YELLOW) ? RED : BLUE;
        PlayerColor e2 = (enemy_team == RED_YELLOW) ? YELLOW : GREEN;
        Bitboard rooks = piece_bitboards_[e1][ROOK] | piece_bitboards_[e2][ROOK] |
                         piece_bitboards_[e1][QUEEN] | piece_bitboards_[e2][QUEEN];
        if (!(GetRookAttacks(king_sq, occupied) & rooks).is_zero()) return false;
        
        Bitboard bishops = piece_bitboards_[e1][BISHOP] | piece_bitboards_[e2][BISHOP] |
                           piece_bitboards_[e1][QUEEN] | piece_bitboards_[e2][QUEEN];
        if (!(GetBishopAttacks(king_sq, occupied) & bishops).is_zero()) return false;
        return true; 
    }

    if (GetPiece(from_sq).GetPieceType() == KING) {
        if (move.GetRookMove().Present()) {
            if (!checkers_.is_zero()) return false;
            return true;
        }
        Bitboard occupied = (team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN]) ^ BitboardImpl::IndexToBitboard(from_sq);
        if (AttackersToExist(to_sq, occupied, OtherTeam(turn_.GetTeam()))) return false;
        return true;
    }

    if (!checkers_.is_zero() && (checkers_ & (checkers_ - 1)).operator bool()) return false;

    // OPTIMIZATION: O(1) Pin Check using kLineMask
    if (IsPinned(from_sq)) {
        if ((BitboardImpl::kLineMask[king_sq][from_sq] & BitboardImpl::IndexToBitboard(to_sq)).is_zero()) {
            return false;
        }
    }

    if (!checkers_.is_zero()) {
        int checker_sq = checkers_.ctz();
        if (to_sq == checker_sq) return true; 
        Bitboard blocking_squares = BitboardImpl::kLineBetween[king_sq][checker_sq];
        if ((blocking_squares & BitboardImpl::IndexToBitboard(to_sq)).is_zero()) return false;
    }
    return true;
}

namespace {
// Helper for pointer arithmetic
ExtMove* AddMovesFromBB(ExtMove* buffer, int from_idx, Bitboard to_bb, const Board& board,
                    CastlingRights initial_cr = CastlingRights::kMissingRights,
                    CastlingRights final_cr = CastlingRights::kMissingRights) {
    BoardLocation from = IndexToLocation(from_idx);
    while (!to_bb.is_zero()) {
        int to_idx = to_bb.ctz();
        to_bb &= to_bb - 1;
        *buffer++ = ExtMove(Move(from, IndexToLocation(to_idx), board.GetPiece(to_idx), initial_cr, final_cr));
    }
    return buffer;
}
}

// ============================================================================
// TEMPLATED MOVE GENERATION
// ============================================================================

template<PlayerColor Us>
ExtMove* Board::GetPawnMovesT(ExtMove* buffer) const {
    // Compile-time constants
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    constexpr int PUSH = (Us == RED) ? PUSH_N : (Us == BLUE) ? PUSH_E : (Us == YELLOW) ? PUSH_S : PUSH_W;
    
    // Derived constants for captures
    constexpr int CAP_1 = (Us == RED) ? PUSH_NW : (Us == BLUE) ? PUSH_NE : (Us == YELLOW) ? PUSH_SE : PUSH_SW;
    constexpr int CAP_2 = (Us == RED) ? PUSH_NE : (Us == BLUE) ? PUSH_SE : (Us == YELLOW) ? PUSH_SW : PUSH_NW;

    const Bitboard& my_pawns = piece_bitboards_[Us][PAWN];
    if (my_pawns.is_zero()) return buffer;
    
    const Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    const Bitboard empty_squares = ~all_pieces;
    const Bitboard enemy_pieces = team_bitboards_[OtherTeam(team)];
    const Bitboard promotion_rank = kPawnPromotionMask[Us];

    // Single pushes
    Bitboard single_pushes = shift<PUSH>(my_pawns) & empty_squares;
    
    // Double pushes
    Bitboard pawns_on_start = my_pawns & kPawnStartMask[Us];
    Bitboard first_step = shift<PUSH>(pawns_on_start) & empty_squares;
    Bitboard double_pushes = shift<PUSH>(first_step) & empty_squares;

    // Normal moves (non-promotion)
    Bitboard single_targets = single_pushes & ~promotion_rank;
    while (!single_targets.is_zero()) {
        int to_idx = single_targets.ctz();
        single_targets &= single_targets - 1;
        int from_idx = to_idx - PUSH;
        // Verify source has pawn (should be guaranteed by bitboard logic but good for debugging if needed)
        // if((my_pawns & IndexToBitboard(from_idx)).is_zero()) continue;
        *buffer++ = ExtMove(Move(IndexToLocation(from_idx), IndexToLocation(to_idx), Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE));
    }
    
    while (!double_pushes.is_zero()) {
        int to_idx = double_pushes.ctz();
        double_pushes &= double_pushes - 1;
        int from_idx = to_idx - PUSH - PUSH;
        *buffer++ = ExtMove(Move(IndexToLocation(from_idx), IndexToLocation(to_idx), Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE));
    }
    
    // Captures
    Bitboard captures1 = shift<CAP_1>(my_pawns) & enemy_pieces;
    Bitboard captures2 = shift<CAP_2>(my_pawns) & enemy_pieces;

    Bitboard all_captures = (captures1 | captures2) & ~promotion_rank;
    while (!all_captures.is_zero()) {
        int to_idx = all_captures.ctz();
        Bitboard to_bb = IndexToBitboard(to_idx);
        all_captures &= all_captures - 1;

        // Determine which pawns could have captured to to_idx
        // Reverse shift to find potential origins
        Bitboard from_bb = (shift<-CAP_1>(to_bb) | shift<-CAP_2>(to_bb)) & my_pawns;

        while (!from_bb.is_zero()) {
            int from_idx = from_bb.ctz();
            from_bb &= from_bb - 1;
            *buffer++ = ExtMove(Move(IndexToLocation(from_idx), IndexToLocation(to_idx), GetPiece(to_idx), BoardLocation::kNoLocation, Piece::kNoPiece, NO_PIECE));
        }
    }
    
    // Promotions
    Bitboard promo_pushes = single_pushes & promotion_rank;
    while (!promo_pushes.is_zero()) {
        int to_idx = promo_pushes.ctz();
        promo_pushes &= promo_pushes - 1;
        int from_idx = to_idx - PUSH;
        BoardLocation from = IndexToLocation(from_idx);
        BoardLocation to = IndexToLocation(to_idx);
        *buffer++ = ExtMove(Move(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, QUEEN));
        *buffer++ = ExtMove(Move(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, ROOK));
        *buffer++ = ExtMove(Move(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, BISHOP));
        *buffer++ = ExtMove(Move(from, to, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece, KNIGHT));
    }
    
    Bitboard promo_captures = (captures1 | captures2) & promotion_rank;
    while (!promo_captures.is_zero()) {
        int to_idx = promo_captures.ctz();
        Bitboard to_bb = IndexToBitboard(to_idx);
        promo_captures &= promo_captures - 1;
        
        Bitboard from_bb = (shift<-CAP_1>(to_bb) | shift<-CAP_2>(to_bb)) & my_pawns;
        Piece captured_piece = GetPiece(to_idx);
        BoardLocation to = IndexToLocation(to_idx);

        while (!from_bb.is_zero()) {
            int from_idx = from_bb.ctz();
            from_bb &= from_bb - 1;
            BoardLocation from = IndexToLocation(from_idx);
            *buffer++ = ExtMove(Move(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, QUEEN));
            *buffer++ = ExtMove(Move(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, ROOK));
            *buffer++ = ExtMove(Move(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, BISHOP));
            *buffer++ = ExtMove(Move(from, to, captured_piece, BoardLocation::kNoLocation, Piece::kNoPiece, KNIGHT));
        }
    }
    
    // EN PASSANT
    constexpr PlayerColor Next = static_cast<PlayerColor>((Us + 1) % 4);
    constexpr PlayerColor Prev = static_cast<PlayerColor>((Us + 3) % 4);
    const PlayerColor opponents[2] = { Next, Prev };

    for (const PlayerColor opponent_color : opponents) {
        const Move* opponent_last_move = nullptr;
        int turns_ago = (Us - opponent_color + 4) % 4;
        
        if (turns_ago > 0 && move_history_ptr_ >= turns_ago) {
            opponent_last_move = &move_history_[move_history_ptr_ - turns_ago];
        } else {
            const auto& enp_move = enp_.enp_moves[opponent_color];
            if (enp_move.has_value()) opponent_last_move = &*enp_move;
        }

        if (opponent_last_move == nullptr || !opponent_last_move->Present()) continue;

        const auto& move_from = opponent_last_move->From();
        const auto& move_to = opponent_last_move->To();
        // Since this is a check outside bitboards, we access Piece via board array or helper
        // Ideally we would have cached this info, but accessing one piece is okay.
        Piece moved_piece = GetPiece(move_to);

        if (moved_piece.GetPieceType() != PAWN ||
            moved_piece.GetColor() != opponent_color ||
            opponent_last_move->ManhattanDistance() != 2 ||
            (move_from.GetRow() != move_to.GetRow() && move_from.GetCol() != move_to.GetCol())) {
            continue;
        }
        
        int moved_to_idx = LocationToIndex(move_to);
        Bitboard capturer_square_bb = shift<-PUSH>(IndexToBitboard(moved_to_idx));
        Bitboard capturer_pawn = capturer_square_bb & my_pawns;

        if (!capturer_pawn.is_zero()) {
            int our_pawn_idx = capturer_pawn.ctz();
            int moved_from_idx = LocationToIndex(move_from);
            int ep_capture_dest_idx = (moved_from_idx + moved_to_idx) / 2;
            
            *buffer++ = ExtMove(Move(
                IndexToLocation(our_pawn_idx),
                IndexToLocation(ep_capture_dest_idx),
                GetPiece(ep_capture_dest_idx), 
                move_to,                        
                moved_piece                     
            ));
        }
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GetKnightMovesT(ExtMove* buffer) const {
    Bitboard knights = piece_bitboards_[Us][KNIGHT];
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    const Bitboard friendly_pieces = team_bitboards_[team];
    
    while(!knights.is_zero()) {
        int from_idx = knights.ctz();
        knights &= knights - 1;
        Bitboard attacks = kKnightAttacks[from_idx] & ~friendly_pieces;
        buffer = AddMovesFromBB(buffer, from_idx, attacks, *this);
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GetBishopMovesT(ExtMove* buffer) const {
    Bitboard bishops = piece_bitboards_[Us][BISHOP];
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    const Bitboard friendly_pieces = team_bitboards_[team];
    const Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    while(!bishops.is_zero()) {
        int from_idx = bishops.ctz();
        bishops &= bishops - 1;
        Bitboard attacks = GetBishopAttacks(from_idx, all_pieces) & ~friendly_pieces;
        buffer = AddMovesFromBB(buffer, from_idx, attacks, *this);
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GetRookMovesT(ExtMove* buffer) const {
    Bitboard rooks = piece_bitboards_[Us][ROOK];
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    const Bitboard friendly_pieces = team_bitboards_[team];
    const Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    const auto& initial_cr = castling_rights_[Us];
    
    while(!rooks.is_zero()) {
        int from_idx = rooks.ctz();
        rooks &= rooks - 1;
        CastlingRights final_cr = initial_cr;
        if(initial_cr.Present()){
            if(from_idx == kInitialRookSq[Us][KINGSIDE] && initial_cr.Kingside()){
                final_cr = CastlingRights(false, initial_cr.Queenside());
            } else if (from_idx == kInitialRookSq[Us][QUEENSIDE] && initial_cr.Queenside()){
                final_cr = CastlingRights(initial_cr.Kingside(), false);
            }
        }
        Bitboard attacks = GetRookAttacks(from_idx, all_pieces) & ~friendly_pieces;
        buffer = AddMovesFromBB(buffer, from_idx, attacks, *this, initial_cr, final_cr.Present() ? final_cr : CastlingRights::kMissingRights);
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GetQueenMovesT(ExtMove* buffer) const {
    Bitboard queens = piece_bitboards_[Us][QUEEN];
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    const Bitboard friendly_pieces = team_bitboards_[team];
    const Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
    while(!queens.is_zero()) {
        int from_idx = queens.ctz();
        queens &= queens - 1;
        Bitboard attacks = GetQueenAttacks(from_idx, all_pieces) & ~friendly_pieces;
        buffer = AddMovesFromBB(buffer, from_idx, attacks, *this);
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GetKingMovesT(ExtMove* buffer) const {
    Bitboard king = piece_bitboards_[Us][KING];
    if (king.is_zero()) return buffer;

    int from_idx = king.ctz();
    constexpr Team team = (Us == RED || Us == YELLOW) ? RED_YELLOW : BLUE_GREEN;
    const Bitboard friendly_pieces = team_bitboards_[team];
    const auto& initial_cr = castling_rights_[Us];
    CastlingRights final_cr(false, false);
    
    Bitboard attacks = kKingAttacks[from_idx] & ~friendly_pieces;
    buffer = AddMovesFromBB(buffer, from_idx, attacks, *this, initial_cr, final_cr);

    if (initial_cr.Present() && !IsAttackedByTeam(OtherTeam(team), from_idx)) {
        Bitboard all_pieces = team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN];
        Team enemy_team = OtherTeam(team);
        BoardLocation king_from_loc = IndexToLocation(from_idx);

        if (initial_cr.Kingside() && (all_pieces & kCastlingEmptyMask[Us][KINGSIDE]).is_zero()) {
            Bitboard attack_mask = kCastlingAttackMask[Us][KINGSIDE];
            bool is_safe = true;
            while(!attack_mask.is_zero()){
                int sq = attack_mask.ctz();
                attack_mask &= (attack_mask-1);
                if(IsAttackedByTeam(enemy_team, sq)){ is_safe = false; break; }
            }
            if(is_safe){
                BoardLocation king_to_loc, rook_from_loc, rook_to_loc;
                rook_from_loc = IndexToLocation(kInitialRookSq[Us][KINGSIDE]);
                
                // Templated compile-time selection for relative coords is harder without more infrastructure
                // stick to switch or if-constexpr for clarity, compiler will optimize constants
                if constexpr (Us == RED) { 
                    king_to_loc = king_from_loc.Relative(0, 2); rook_to_loc = king_from_loc.Relative(0, 1); 
                } else if constexpr (Us == BLUE) {
                    king_to_loc = king_from_loc.Relative(2, 0); rook_to_loc = king_from_loc.Relative(1, 0);
                } else if constexpr (Us == YELLOW) {
                    king_to_loc = king_from_loc.Relative(0, -2); rook_to_loc = king_from_loc.Relative(0, -1);
                } else { // GREEN
                    king_to_loc = king_from_loc.Relative(-2, 0); rook_to_loc = king_from_loc.Relative(-1, 0);
                }
                *buffer++ = ExtMove(Move(king_from_loc, king_to_loc, SimpleMove(rook_from_loc, rook_to_loc), initial_cr, final_cr));
            }
        }
        if (initial_cr.Queenside() && (all_pieces & kCastlingEmptyMask[Us][QUEENSIDE]).is_zero()) {
            Bitboard attack_mask = kCastlingAttackMask[Us][QUEENSIDE];
            bool is_safe = true;
            while(!attack_mask.is_zero()){
                int sq = attack_mask.ctz();
                attack_mask &= (attack_mask-1);
                if(IsAttackedByTeam(enemy_team, sq)){ is_safe = false; break; }
            }
            if(is_safe){
                BoardLocation king_to_loc, rook_from_loc, rook_to_loc;
                rook_from_loc = IndexToLocation(kInitialRookSq[Us][QUEENSIDE]);
                
                if constexpr (Us == RED) {
                    king_to_loc = king_from_loc.Relative(0, -2); rook_to_loc = king_from_loc.Relative(0, -1);
                } else if constexpr (Us == BLUE) {
                    king_to_loc = king_from_loc.Relative(-2, 0); rook_to_loc = king_from_loc.Relative(-1, 0);
                } else if constexpr (Us == YELLOW) {
                    king_to_loc = king_from_loc.Relative(0, 2); rook_to_loc = king_from_loc.Relative(0, 1);
                } else { // GREEN
                    king_to_loc = king_from_loc.Relative(2, 0); rook_to_loc = king_from_loc.Relative(1, 0);
                }
                *buffer++ = ExtMove(Move(king_from_loc, king_to_loc, SimpleMove(rook_from_loc, rook_to_loc), initial_cr, final_cr));
            }
        }
    }
    return buffer;
}

template<PlayerColor Us>
ExtMove* Board::GenerateMovesT(ExtMove* buffer) const {
    buffer = GetPawnMovesT<Us>(buffer);
    buffer = GetKnightMovesT<Us>(buffer);
    buffer = GetBishopMovesT<Us>(buffer);
    buffer = GetRookMovesT<Us>(buffer);
    buffer = GetQueenMovesT<Us>(buffer);
    buffer = GetKingMovesT<Us>(buffer);
    return buffer;
}

ExtMove* Board::GetPseudoLegalMoves2(ExtMove* buffer) const {
    switch (turn_.GetColor()) {
        case RED:    return GenerateMovesT<RED>(buffer);
        case BLUE:   return GenerateMovesT<BLUE>(buffer);
        case YELLOW: return GenerateMovesT<YELLOW>(buffer);
        case GREEN:  return GenerateMovesT<GREEN>(buffer);
        default:     return buffer;
    }
}

void Board::MakeMove(const Move& move) {
    // FIXED: Save safety state to stack
    SafetyInfo& backup = safety_stack_[safety_stack_ptr_++];
    backup.checkers = checkers_;
    std::memcpy(backup.blockers_for_king, blockers_for_king_, sizeof(blockers_for_king_));
    std::memcpy(backup.pinners, pinners_, sizeof(pinners_));

    const Player player = turn_;
    const BoardLocation from = move.From();
    const BoardLocation to = move.To();

    if (move.IsStandardCapture()) RemovePiece(to);
    
    if (move.GetPromotionPieceType() != NO_PIECE) {
        RemovePiece(from);
        SetPiece(to, Piece(player.GetColor(), move.GetPromotionPieceType()));
    } else {
        MovePiece(from, to);
    }

    if (move.GetEnpassantLocation().Present()) RemovePiece(move.GetEnpassantLocation());
    if (move.GetRookMove().Present()) {
        SimpleMove rook_move = move.GetRookMove();
        MovePiece(rook_move.From(), rook_move.To());
    }
    
    if (move.GetCastlingRights().Present()) castling_rights_[player.GetColor()] = move.GetCastlingRights();
    
    int t = static_cast<int>(turn_.GetColor());
    UpdateTurnHash(t);
    turn_ = GetNextPlayer(turn_);
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    if (move_history_ptr_ < kMaxGameDepth) {
        move_history_[move_history_ptr_++] = move;
    } else {
        std::cerr << "History overflow" << std::endl;
        abort();
    }
}

void Board::UndoMove() {
    assert(move_history_ptr_ > 0);
    const Move& move = move_history_[--move_history_ptr_];
    
    Player turn_before = GetPreviousPlayer(turn_);
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));
    turn_ = turn_before;
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    const BoardLocation& to = move.To();
    const BoardLocation& from = move.From();
    
    if (move.GetInitialCastlingRights().Present()) castling_rights_[turn_before.GetColor()] = move.GetInitialCastlingRights();

    if (move.GetRookMove().Present()) {
        SimpleMove rook_move = move.GetRookMove();
        MovePiece(rook_move.To(), rook_move.From());
    }

    if (move.GetPromotionPieceType() != NO_PIECE) {
        RemovePiece(to);
        SetPiece(from, Piece(turn_before.GetColor(), PAWN));
    } else {
        MovePiece(to, from);
    }

    if (move.IsStandardCapture()) SetPiece(to, move.GetStandardCapture());
    if (move.GetEnpassantLocation().Present()) SetPiece(move.GetEnpassantLocation(), move.GetEnpassantCapture());
    
    // FIXED: Restore safety state
    --safety_stack_ptr_;
    const SafetyInfo& backup = safety_stack_[safety_stack_ptr_];
    checkers_ = backup.checkers;
    std::memcpy(blockers_for_king_, backup.blockers_for_king, sizeof(blockers_for_king_));
    std::memcpy(pinners_, backup.pinners, sizeof(pinners_));
}

GameResult Board::GetGameResult() {
  if (GetKingLocation(turn_.GetColor()).Missing()) {
      return turn_.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
  }
  Player player = turn_;

  ExtMove move_buffer_internal[300];
  ExtMove* end_ptr = GetPseudoLegalMoves2(move_buffer_internal);
  
  for (ExtMove* m_ptr = move_buffer_internal; m_ptr < end_ptr; ++m_ptr) {
    const auto& move = *m_ptr;
    MakeMove(move);
    GameResult king_capture_result = CheckWasLastMoveKingCapture();
    if (king_capture_result != IN_PROGRESS) {
      UndoMove();
      return king_capture_result;
    }
    bool legal = !IsKingInCheck(player); 
    UndoMove();
    if (legal) return IN_PROGRESS; 
  }
  if (!IsKingInCheck(player)) return STALEMATE;
  return player.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
}

bool Board::IsKingInCheck(const Player& player) const {
  // OPTIMIZATION: Use cached check status if asking about current turn
  if (player.GetColor() == turn_.GetColor()) {
      return !checkers_.is_zero();
  }
  const auto king_location = GetKingLocation(player.GetColor());
  if (king_location.Missing()) return true; 
  return IsAttackedByTeam(OtherTeam(player.GetTeam()), LocationToIndex(king_location));
}

bool Board::IsKingInCheck(Team team) const {
  if (team == RED_YELLOW) return IsKingInCheck(Player(RED)) || IsKingInCheck(Player(YELLOW));
  return IsKingInCheck(Player(BLUE)) || IsKingInCheck(Player(GREEN));
}

GameResult Board::CheckWasLastMoveKingCapture() const {
    if (LastMoveWasCapture()) {
        const auto& last_move = GetLastMove();
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

void Board::MakeNullMove() { SetPlayer(GetNextPlayer(turn_)); }
void Board::UndoNullMove() { SetPlayer(GetPreviousPlayer(turn_)); }

int Board::MobilityEvaluation(const Player& player) {
    Player current_turn = turn_;
    turn_ = player;
    ExtMove buffer[300];
    ExtMove* end = GetPseudoLegalMoves2(buffer);
    turn_ = current_turn;
    return (int)(end - buffer) * kMobilityMultiplier;
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
  return std::abs(from_.GetRow() - to_.GetRow()) + std::abs(from_.GetCol() - to_.GetCol());
}

namespace {
std::string ToStr(PieceType piece_type) {
  switch (piece_type) {
  case PAWN: return "P"; case ROOK: return "R"; case KNIGHT: return "N";
  case BISHOP: return "B"; case KING: return "K"; case QUEEN: return "Q";
  default: return " ";
  }
}
} 

std::ostream& operator<<(std::ostream& os, const Board& board) {
  for (int r = 0; r < 14; r++) {
    for (int c = 0; c < 14; c++) {
        BoardLocation loc(r, c);
        if((kLegalSquares & IndexToBitboard(LocationToIndex(loc))).is_zero()) os << " ";
        else {
            const auto piece = board.GetPiece(loc);
            if (piece.Missing()) os << "."; 
            else os << ToStr(piece.GetPieceType()); 
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
  if (GetPromotionPieceType() != NO_PIECE) s += ToStr(GetPromotionPieceType());
  return s;
}

bool Board::DiscoversCheck(const Move& move) const {
    const int from_sq = BitboardImpl::LocationToIndex(move.From());
    const int to_sq = BitboardImpl::LocationToIndex(move.To());
    const Team my_team = turn_.GetTeam();
    const Team enemy_team = OtherTeam(my_team);
    const PlayerColor e1 = (enemy_team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor e2 = (enemy_team == RED_YELLOW) ? YELLOW : GREEN;
    Bitboard enemy_kings = piece_bitboards_[e1][KING] | piece_bitboards_[e2][KING];
    const PlayerColor f1 = (my_team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor f2 = (my_team == RED_YELLOW) ? YELLOW : GREEN;
    const Bitboard my_sliders = piece_bitboards_[f1][BISHOP] | piece_bitboards_[f2][BISHOP] |
                                piece_bitboards_[f1][ROOK]   | piece_bitboards_[f2][ROOK] |
                                piece_bitboards_[f1][QUEEN]  | piece_bitboards_[f2][QUEEN];
    const Bitboard occupied = team_bitboards_[0] | team_bitboards_[1];

    while (!enemy_kings.is_zero()) {
        int king_sq = enemy_kings.ctz();
        enemy_kings &= enemy_kings - 1;
        Bitboard potential_pinners = GetQueenAttacks(king_sq, occupied) & my_sliders;
        while (!potential_pinners.is_zero()) {
            int slider_sq = potential_pinners.ctz();
            potential_pinners &= potential_pinners - 1;
            if ((BitboardImpl::kLineBetween[king_sq][slider_sq] & occupied) == BitboardImpl::IndexToBitboard(from_sq)) {
                if ((BitboardImpl::kLineBetween[king_sq][slider_sq] & BitboardImpl::IndexToBitboard(to_sq)).is_zero()) return true; 
            }
        }
    }
    return false;
}

bool Board::DeliversCheck(const Move& move) {
    if(!move.Present()) return false;
    Piece moved = GetPiece(move.From());
    int to_sq = LocationToIndex(move.To());
    Bitboard all_pieces = (team_bitboards_[0] | team_bitboards_[1]);
    Bitboard all_after_move = (all_pieces ^ IndexToBitboard(LocationToIndex(move.From()))) | IndexToBitboard(to_sq);
    if(move.IsCapture()) all_after_move &= ~IndexToBitboard(LocationToIndex(move.GetStandardCapture().Present() ? move.To() : move.GetEnpassantLocation()));
    
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
  if (delivers_check_ < 0) delivers_check_ = board.DeliversCheck(*this);
  return delivers_check_;
}

int Move::ApproxSEE(const Board& board, const int* piece_evaluations) const {
  const auto capture = GetCapturePiece();
  if(!capture.Present()) return 0;
  const auto piece = board.GetPiece(From());
  if (!piece.Present()) return 0; 
  int captured_val = piece_evaluations[capture.GetPieceType()];
  int attacker_val = piece_evaluations[piece.GetPieceType()];
  return captured_val - attacker_val;
}

int GetLeastValuableAttacker(const Board& board, int sq, Team team, const Bitboard& occupied, PieceType& out_type) {
    Bitboard attackers = Bitboard(0);
    const PlayerColor c1 = (team == RED_YELLOW) ? RED : BLUE;
    const PlayerColor c2 = (team == RED_YELLOW) ? YELLOW : GREEN;

    attackers |= (BitboardImpl::kPawnAttacks[GetPartner(Player(c1)).GetColor()][sq] & board.piece_bitboards_[c1][PAWN]);
    attackers |= (BitboardImpl::kPawnAttacks[GetPartner(Player(c2)).GetColor()][sq] & board.piece_bitboards_[c2][PAWN]);
    attackers |= (BitboardImpl::kKnightAttacks[sq] & (board.piece_bitboards_[c1][KNIGHT] | board.piece_bitboards_[c2][KNIGHT]));
    attackers |= (BitboardImpl::kKingAttacks[sq] & (board.piece_bitboards_[c1][KING] | board.piece_bitboards_[c2][KING]));

    const Bitboard rooks_and_queens = (board.piece_bitboards_[c1][ROOK] | board.piece_bitboards_[c2][ROOK] |
                                     board.piece_bitboards_[c1][QUEEN] | board.piece_bitboards_[c2][QUEEN]);
    attackers |= (board.GetRookAttacks(sq, occupied) & rooks_and_queens);
    
    const Bitboard bishops_and_queens = (board.piece_bitboards_[c1][BISHOP] | board.piece_bitboards_[c2][BISHOP] |
                                       board.piece_bitboards_[c1][QUEEN] | board.piece_bitboards_[c2][QUEEN]);
    attackers |= (board.GetBishopAttacks(sq, occupied) & bishops_and_queens);
    
    Bitboard valid_attackers = attackers & occupied;
    if (valid_attackers.is_zero()) return -1;

    for (int pt_idx = PAWN; pt_idx <= KING; ++pt_idx) {
        const PieceType pt = static_cast<PieceType>(pt_idx);
        const Bitboard type_attackers = (board.piece_bitboards_[c1][pt] | board.piece_bitboards_[c2][pt]) & valid_attackers;
        if (!type_attackers.is_zero()) {
            out_type = pt;
            return type_attackers.ctz(); 
        }
    }
    return -1; 
}

int SeeRecursive(const Board& board, const int piece_evaluations[6], int target_sq, Bitboard occupied, Team side_to_attack, int victim_value) {
    PieceType lva_type;
    int lva_sq = GetLeastValuableAttacker(board, target_sq, side_to_attack, occupied, lva_type);
    if (lva_sq == -1) return 0;
    Bitboard next_occupied = occupied ^ BitboardImpl::IndexToBitboard(lva_sq);
    int gain = victim_value - SeeRecursive(board, piece_evaluations, target_sq, next_occupied, OtherTeam(side_to_attack), piece_evaluations[lva_type]);
    return std::max(0, gain);
}

int StaticExchangeEvaluationCapture(const int piece_evaluations[6], const Board& board, const Move& move) {
    if (!move.IsCapture()) return 0;
    const Piece captured_piece = move.GetCapturePiece();
    const Piece attacker_piece = board.GetPiece(move.From());
    if (!captured_piece.Present() || !attacker_piece.Present()) return 0; 
    const int from_sq = BitboardImpl::LocationToIndex(move.From());
    const int to_sq = BitboardImpl::LocationToIndex(move.To());
    const int initial_gain = piece_evaluations[captured_piece.GetPieceType()];
    const int new_victim_value = piece_evaluations[attacker_piece.GetPieceType()];
    
    Bitboard occupied_before_move = board.team_bitboards_[RED_YELLOW] | board.team_bitboards_[BLUE_GREEN];
    Bitboard occupied_after_move = (occupied_before_move & ~BitboardImpl::IndexToBitboard(from_sq)) | BitboardImpl::IndexToBitboard(to_sq);

    if (move.GetEnpassantLocation().Present()) {
        int ep_captured_sq = BitboardImpl::LocationToIndex(move.GetEnpassantLocation());
        occupied_after_move &= ~BitboardImpl::IndexToBitboard(ep_captured_sq);
    }
    
    const Team opponent_team = OtherTeam(board.GetTurn().GetTeam());
    int opponent_gain = SeeRecursive(board, piece_evaluations, to_sq, occupied_after_move, opponent_team, new_victim_value);
    return initial_gain - opponent_gain;
}

int Move::SEE(Board& board, const int* piece_evaluations) {
  if (see_ == kSeeNotSet) see_ = StaticExchangeEvaluationCapture(piece_evaluations, board, *this);
  return see_;
}

namespace {
std::string ToStr(PlayerColor color) {
  switch (color) {
  case RED: return "RED"; case BLUE: return "BLUE"; case YELLOW: return "YELLOW"; case GREEN: return "GREEN";
  default: return "UNINITIALIZED_PLAYER";
  }
}
} 

std::ostream& operator<<(std::ostream& os, const Piece& piece) {
  if (!piece.Present()) { os << "NoPiece"; return os; }
  os << ToStr(piece.GetColor()) << "(" << ToStr(piece.GetPieceType()) << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const PlacedPiece& placed_piece) {
  os << placed_piece.GetPiece() << "@" << placed_piece.GetLocation();
  return os;
}
std::ostream& operator<<(std::ostream& os, const Player& player) {
  os << "Player(" << ToStr(player.GetColor()) << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const BoardLocation& location) {
  if (!location.Present()) { os << "Loc(null)"; return os; }
  os << "Loc(" << (int)location.GetRow() << ", " << (int)location.GetCol() << ")";
  return os;
}
std::ostream& operator<<(std::ostream& os, const Move& move) {
  if (!move.Present()) { os << "Move(null)"; return os; }
  os << "Move(" << move.From() << " -> " << move.To() << ")";
  return os;
}

}  // namespace chess