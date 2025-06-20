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

#include "board.h"

namespace chess {

// ============================================================================
// Bitboard Implementation Details
// ============================================================================
namespace BitboardImpl {

constexpr int kBoardWidth = 16;
constexpr int kBoardHeight = 15;
constexpr int kNumSquares = kBoardWidth * kBoardHeight; // 240, fits in 256-bit

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
int kInitialRookSq[4][2];

enum RayDirection { D_NE, D_NW, D_SE, D_SW, D_N, D_E, D_S, D_W };

// Conversion helpers
inline int LocationToIndex(const BoardLocation& loc) {
    if (!loc.Present()) return -1;
    return (loc.GetRow() + 1) * kBoardWidth + (loc.GetCol() + 1);
}

inline BoardLocation IndexToLocation(int index) {
    if (index < 0 || index >= kNumSquares) return BoardLocation::kNoLocation;
    // Check if index corresponds to a valid board square (not padding)
    int r = (index / kBoardWidth) - 1;
    int c = (index % kBoardWidth) - 1;
    if (r < 0 || r >= 14 || c < 0 || c >= 14) return BoardLocation::kNoLocation;
    return BoardLocation(r, c);
}

inline Bitboard IndexToBitboard(int index) {
    if (index < 0 || index >= kNumSquares) return Bitboard(0);
    return Bitboard(1) << index;
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
    
    is_initialized = true;
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
    Bitboard mask = IndexToBitboard(index);
    for (int c = 0; c < 4; ++c) {
        if (!(color_bitboards_[c] & mask).is_zero()) {
            for (int pt = 0; pt < 6; ++pt) {
                if (!(piece_bitboards_[c][pt] & mask).is_zero()) {
                    return Piece(static_cast<PlayerColor>(c), static_cast<PieceType>(pt));
                }
            }
        }
    }
    return Piece::kNoPiece;
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

Bitboard Board::GetRookAttacks(int sq, Bitboard blockers) const {
    Bitboard attacks;
    // North
    Bitboard ray_n = kRayAttacks[sq][D_N];
    Bitboard b_n = ray_n & blockers;
    if (!b_n.is_zero()) attacks |= (ray_n ^ kRayAttacks[b_n.ctz()][D_N]);
    else attacks |= ray_n;
    // South
    Bitboard ray_s = kRayAttacks[sq][D_S];
    Bitboard b_s = ray_s & blockers;
    if (!b_s.is_zero()) attacks |= (ray_s ^ kRayAttacks[255 - b_s.clz()][D_S]);
    else attacks |= ray_s;
    // East
    Bitboard ray_e = kRayAttacks[sq][D_E];
    Bitboard b_e = ray_e & blockers;
    if (!b_e.is_zero()) attacks |= (ray_e ^ kRayAttacks[b_e.ctz()][D_E]);
    else attacks |= ray_e;
    // West
    Bitboard ray_w = kRayAttacks[sq][D_W];
    Bitboard b_w = ray_w & blockers;
    if (!b_w.is_zero()) attacks |= (ray_w ^ kRayAttacks[255 - b_w.clz()][D_W]);
    else attacks |= ray_w;
    return attacks;
}

Bitboard Board::GetBishopAttacks(int sq, Bitboard blockers) const {
    Bitboard attacks;
    // NE
    Bitboard ray_ne = kRayAttacks[sq][D_NE];
    Bitboard b_ne = ray_ne & blockers;
    if (!b_ne.is_zero()) attacks |= (ray_ne ^ kRayAttacks[b_ne.ctz()][D_NE]);
    else attacks |= ray_ne;
    // SW
    Bitboard ray_sw = kRayAttacks[sq][D_SW];
    Bitboard b_sw = ray_sw & blockers;
    if (!b_sw.is_zero()) attacks |= (ray_sw ^ kRayAttacks[255 - b_sw.clz()][D_SW]);
    else attacks |= ray_sw;
    // NW
    Bitboard ray_nw = kRayAttacks[sq][D_NW];
    Bitboard b_nw = ray_nw & blockers;
    if (!b_nw.is_zero()) attacks |= (ray_nw ^ kRayAttacks[b_nw.ctz()][D_NW]);
    else attacks |= ray_nw;
    // SE
    Bitboard ray_se = kRayAttacks[sq][D_SE];
    Bitboard b_se = ray_se & blockers;
    if (!b_se.is_zero()) attacks |= (ray_se ^ kRayAttacks[255 - b_se.clz()][D_SE]);
    else attacks |= ray_se;
    return attacks;
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
    Bitboard pawns = piece_bitboards_[color][PAWN];
    Bitboard empty_squares = ~(team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN]);
    Bitboard enemy_pieces = team_bitboards_[OtherTeam(team)];

    Bitboard pawns_copy = pawns;
    while (!pawns_copy.is_zero()) {
        int from_idx = pawns_copy.ctz();
        pawns_copy &= pawns_copy - 1;
        
        // --- 1. Standard Pawn Pushes ---
        Bitboard push1 = kPawnSinglePush[color][from_idx] & empty_squares;
        if (!push1.is_zero()) {
            AddPawnMovesFromBB(moves, from_idx, push1, color, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece);
            // Check for double push only if single push is possible
            if (kPawnStartMask[color] & IndexToBitboard(from_idx)) {
                Bitboard push2 = kPawnDoublePush[color][from_idx] & empty_squares;
                if (!push2.is_zero()) {
                    AddPawnMovesFromBB(moves, from_idx, push2, color, Piece::kNoPiece, BoardLocation::kNoLocation, Piece::kNoPiece);
                }
            }
        }
        
        // --- 2. Standard Pawn Captures ---
        Bitboard attacks = kPawnAttacks[color][from_idx] & enemy_pieces;
        while (!attacks.is_zero()) {
            int to_idx = attacks.ctz();
            attacks &= attacks - 1;
            AddPawnMovesFromBB(moves, from_idx, IndexToBitboard(to_idx), color, GetPiece(to_idx), BoardLocation::kNoLocation, Piece::kNoPiece);
        }
    }
    
    // --- 3. Corrected En-Passant Logic ---
    // Helper to find the relevant last move for a given opponent.
    // This restores the logic from the original mailbox version.
    auto find_relevant_move = [&](PlayerColor opponent_color) -> const Move* {
        // Calculate how many turns ago the given opponent moved.
        // e.g., If it's Red's (0) turn, Green (3) moved 1 turn ago, Blue (1) moved 3 turns ago.
        int turns_ago = (color - opponent_color + 4) % 4;
        
        // Check the move history first.
        if (moves_.size() >= turns_ago) {
            return &moves_[moves_.size() - turns_ago];
        }
        
        // If history is too short (e.g., loaded from FEN), check the enp_ struct.
        const auto& enp_move = enp_.enp_moves[opponent_color];
        if (enp_move.has_value()) {
            return &*enp_move;
        }
        
        return nullptr;
    };

    // An en-passant capture is possible against either opponent. Check both.
    const PlayerColor opponents[2] = { GetNextPlayer(player).GetColor(), GetPreviousPlayer(player).GetColor() };

    for (const PlayerColor opponent_color : opponents) {
        const Move* opponent_last_move = find_relevant_move(opponent_color);

        if (opponent_last_move == nullptr || !opponent_last_move->Present()) {
            continue;
        }

        // Verify the opponent's move was a 2-square pawn push.
        Piece moved_piece = GetPiece(opponent_last_move->To());
        if (moved_piece.GetPieceType() != PAWN || moved_piece.GetColor() != opponent_color) {
            continue;
        }
        if (opponent_last_move->ManhattanDistance() != 2) {
            continue;
        }

        // This was a valid EP-creating move. The square behind the pawn is the target.
        int from_idx = LocationToIndex(opponent_last_move->From());
        int to_idx = LocationToIndex(opponent_last_move->To());
        int ep_target_idx = (from_idx + to_idx) / 2;

        // Find which of OUR pawns can make the capture using the reverse-attack trick.
        // A RED pawn attacks square X if a YELLOW (partner) pawn at X would attack the RED pawn's square.
        PlayerColor partner_color = GetPartner(player).GetColor();
        Bitboard potential_attackers_bb = kPawnAttacks[partner_color][ep_target_idx];
        Bitboard our_attacking_pawns = pawns & potential_attackers_bb;
        
        if (our_attacking_pawns.is_zero()) {
            continue;
        }

        // --- Restore the simultaneous capture feature ---
        // 'ep_capture' is the pawn that moved two squares.
        // 'standard_capture' is a piece that might be on the destination square.
        BoardLocation ep_target_loc = IndexToLocation(ep_target_idx);
        Piece standard_capture = GetPiece(ep_target_loc);
        Piece ep_capture = moved_piece;
        BoardLocation ep_capture_loc = opponent_last_move->To();
        
        // The move is illegal if a FRIENDLY piece is on the destination square.
        if (standard_capture.Present() && standard_capture.GetTeam() == team) {
            continue;
        }

        // Generate the en-passant move(s).
        while (!our_attacking_pawns.is_zero()) {
            int our_pawn_idx = our_attacking_pawns.ctz();
            our_attacking_pawns &= (our_attacking_pawns - 1);
            BoardLocation our_pawn_loc = IndexToLocation(our_pawn_idx);
            
            // En-passant cannot result in a promotion, so NO_PIECE is passed.
            // The correct Move constructor is used to specify both standard and e.p. captures.
            moves.emplace_back(our_pawn_loc, 
                               ep_target_loc, 
                               standard_capture, 
                               ep_capture_loc, 
                               ep_capture,
                               NO_PIECE);
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
        AddMovesFromBB(moves, from_idx, attacks, *this, initial_cr, final_cr);
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
                // Define king and rook moves with simple, clear relative logic
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
    const Piece moved_piece = GetPiece(from);

    const auto initial_castling_rights = castling_rights_[player.GetColor()];

    // Standard capture
    if (move.IsStandardCapture()) {
        RemovePiece(to);
    }
    
    // Move the piece (using MovePiece for efficiency)
    MovePiece(from, to);
    if (move.GetPromotionPieceType() != NO_PIECE) {
        RemovePiece(to); // Remove the moved pawn
        SetPiece(to, Piece(player.GetColor(), move.GetPromotionPieceType())); // Add the new piece
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
    
    // Update castling rights (move generation pre-calculates the final state)
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
    moves_.pop_back();
    
    Player turn_before = GetPreviousPlayer(turn_);

    // Update turn first to match state before the move
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));
    turn_ = turn_before;
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    const BoardLocation& to = move.To();
    const BoardLocation& from = move.From();
    
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

    // Restore castling rights
    if (move.GetInitialCastlingRights().Present()) {
        castling_rights_[turn_before.GetColor()] = move.GetInitialCastlingRights();
    }
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
    bool legal = !IsKingInCheck(player);
    UndoMove();
    if (legal) {
      return IN_PROGRESS;
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
  // This call does everything needed:
  // 1. Updates turn_ to the next player.
  // 2. Updates the Zobrist hash correctly.
  SetPlayer(GetNextPlayer(turn_));
}

void Board::UndoNullMove() {
  // This call correctly reverts the state:
  // 1. Updates turn_ back to the previous player.
  // 2. Reverts the Zobrist hash correctly.
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

const CastlingRights& Board::GetCastlingRights(const Player& player) { return castling_rights_[player.GetColor()]; }
Team OtherTeam(Team team) { return team == RED_YELLOW ? BLUE_GREEN : RED_YELLOW; }
inline Team GetTeam(PlayerColor color) { return (color == RED || color == YELLOW) ? RED_YELLOW : BLUE_GREEN; }
Player GetNextPlayer(const Player& player) { return Player(static_cast<PlayerColor>((player.GetColor() + 1) % 4));}
Player GetPreviousPlayer(const Player& player) { return Player(static_cast<PlayerColor>((player.GetColor() + 3) % 4));}
Player GetPartner(const Player& player) { return Player(static_cast<PlayerColor>((player.GetColor() + 2) % 4));}

std::string BoardLocation::PrettyStr() const {
  std::string s;
  s += ('a' + GetCol());
  s += std::to_string(14 - GetRow());
  return s;
}
std::string Move::PrettyStr() const {
  if(!Present()) return "null";
  std::string s = from_.PrettyStr() + to_.PrettyStr();
  if (GetPromotionPieceType() != NO_PIECE) {
    s += ToStr(GetPromotionPieceType());
  }
  return s;
}

bool Board::DiscoversCheck(const Move& move) const {
    int from_sq = LocationToIndex(move.From());
    int to_sq = LocationToIndex(move.To());
    Team my_team = turn_.GetTeam();
    Team enemy_team = OtherTeam(my_team);

    Bitboard friendly_sliders = (piece_bitboards_[turn_.GetColor()][BISHOP] | piece_bitboards_[turn_.GetColor()][ROOK] | piece_bitboards_[turn_.GetColor()][QUEEN] |
                                 piece_bitboards_[GetPartner(turn_).GetColor()][BISHOP] | piece_bitboards_[GetPartner(turn_).GetColor()][ROOK] | piece_bitboards_[GetPartner(turn_).GetColor()][QUEEN]);

    PlayerColor e1 = enemy_team == RED_YELLOW ? RED : BLUE;
    PlayerColor e2 = enemy_team == RED_YELLOW ? YELLOW : GREEN;
    Bitboard enemy_kings = piece_bitboards_[e1][KING] | piece_bitboards_[e2][KING];

    while(!enemy_kings.is_zero()){
        int king_sq = enemy_kings.ctz();
        enemy_kings &= enemy_kings - 1;
        
        Bitboard line = kLineBetween[king_sq][from_sq];
        if(!line.is_zero() && !(line & friendly_sliders).is_zero()){
            // The moving piece was on a line between a friendly slider and an enemy king.
            // If it moves off that line, it's a discovered check.
            if((line & IndexToBitboard(to_sq)).is_zero()) {
                return true;
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
    if(move.IsCapture()) {
        all_after_move &= ~IndexToBitboard(to_sq); // Don't count captured piece for blocking
    }
    
    Team enemy_team = OtherTeam(moved.GetTeam());
    PlayerColor e1 = enemy_team == RED_YELLOW ? RED : BLUE;
    PlayerColor e2 = enemy_team == RED_YELLOW ? YELLOW : GREEN;
    Bitboard enemy_kings = piece_bitboards_[e1][KING] | piece_bitboards_[e2][KING];
    
    Bitboard attacks;
    switch(moved.GetPieceType()){
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

int Move::ApproxSEE(const Board& board, const int* piece_evaluations) {
  const auto capture = GetCapturePiece();
  if(!capture.Present()) return 0;
  const auto piece = board.GetPiece(From());
  int captured_val = piece_evaluations[capture.GetPieceType()];
  int attacker_val = piece_evaluations[piece.GetPieceType()];
  return captured_val - attacker_val;
}

// Recursive helper for Static Exchange Evaluation.
// It finds the least valuable attacker for 'sq' from the 'attacker_team'
// and simulates the exchange recursively.
// 'occupied' is a bitboard of all pieces currently on the board for this simulation.
int SeeRecursive(
    const Board& board,
    const int piece_evaluations[6],
    int sq,
    Team attacker_team,
    Bitboard occupied) {

    int from_sq = -1;
    PieceType attacker_type = NO_PIECE;

    // 1. Find the least valuable attacker for the given square 'sq'.
    // We iterate from Pawn to King to find the cheapest piece type that can attack.
    Bitboard attackers_bb = board.GetAttackersBB(sq, attacker_team);
    attackers_bb &= occupied; // Only consider pieces that are still on the board in our simulation

    if (attackers_bb.is_zero()) {
        return 0; // No more attackers, exchange ends.
    }

    // Find the piece type and location of the least valuable attacker
    for (int pt_idx = 0; pt_idx < 6; ++pt_idx) {
        PieceType pt = static_cast<PieceType>(pt_idx);
        PlayerColor c1 = (attacker_team == RED_YELLOW) ? RED : BLUE;
        PlayerColor c2 = (attacker_team == RED_YELLOW) ? YELLOW : GREEN;
        
        Bitboard type_attackers = (board.piece_bitboards_[c1][pt] | board.piece_bitboards_[c2][pt]) & attackers_bb;
        
        if (!type_attackers.is_zero()) {
            attacker_type = pt;
            from_sq = type_attackers.ctz(); // Get the square of the first attacker of this type
            break;
        }
    }
    
    if (from_sq == -1) {
       return 0; // Should be covered by the initial check, but good for safety.
    }

    // 2. Simulate the capture for the recursion.
    // The "victim" for the next stage of the recursion is the piece that just attacked.
    int victim_value = piece_evaluations[attacker_type];
    
    // Remove the attacker from the board for the next recursive call
    Bitboard next_occupied = occupied ^ BitboardImpl::IndexToBitboard(from_sq);

    // 3. Recurse. The gain for us is the victim's value minus what the opponent gains.
    int gain = victim_value - SeeRecursive(board, piece_evaluations, sq, OtherTeam(attacker_team), next_occupied);

    // We are not forced to continue a losing exchange.
    return std::max(0, gain);
}

int StaticExchangeEvaluationCapture(
    const int piece_evaluations[6],
    Board& board,
    const Move& move) {
    
    if (!move.IsCapture()) {
        return 0;
    }
    
    Piece captured_piece = move.GetCapturePiece();
    int victim_value = piece_evaluations[captured_piece.GetPieceType()];
    
    int from_sq = BitboardImpl::LocationToIndex(move.From());
    int to_sq = BitboardImpl::LocationToIndex(move.To());

    // Setup the board state as if the move has been made
    Bitboard all_pieces = board.team_bitboards_[0] | board.team_bitboards_[1];
    Bitboard occupied_after_move = (all_pieces ^ BitboardImpl::IndexToBitboard(from_sq)) | BitboardImpl::IndexToBitboard(to_sq);

    // The captured piece is removed from the 'occupied' bitboard for the simulation.
    // The attacker's value is now the "victim" on the square for the opponent's turn.
    Piece attacker_piece = board.GetPiece(move.From());
    int first_attacker_value = piece_evaluations[attacker_piece.GetPieceType()];

    // The opponent will now recapture. Their gain is calculated by the recursive helper.
    int opponent_gain = SeeRecursive(
        board,
        piece_evaluations,
        to_sq,
        OtherTeam(board.GetTurn().GetTeam()), // It's the opponent's turn to recapture
        occupied_after_move
    );

    // Our final gain is the value of the piece we took, minus the opponent's net gain.
    return victim_value - opponent_gain;
}


int Move::SEE(Board& board, const int* piece_evaluations) {
  if (see_ == kSeeNotSet) {
    see_ = StaticExchangeEvaluationCapture(piece_evaluations, board, *this);
  }
  return see_;
}

}  // namespace chess