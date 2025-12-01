#ifndef _BOARD_H_
#define _BOARD_H_

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <cstring> // for memcpy
#include <cassert>
#include <cmath>   // for abs in ManhattanDistance

#include "FastUint256.h"

using Bitboard = my_math::FastUint256;

namespace chess {

class BoardLocation;

namespace BitboardImpl {
    constexpr int kBoardWidth = 16;
    constexpr int kBoardHeight = 15;
    constexpr int kNumSquares = kBoardWidth * kBoardHeight;

    // OPTIMIZATION: Lookup tables
    extern BoardLocation kIndexToLocation[256];
    extern int kLocationToIndex[256];
    
    extern Bitboard kSquareBitboards[256];
    extern Bitboard kZero; 

    // Definitions moved to bottom
    int LocationToIndex(const BoardLocation& loc);
    BoardLocation IndexToLocation(int index);
    const Bitboard& IndexToBitboard(int index); 
    
    void InitBitboards(); 

    extern Bitboard kLegalSquares;
    extern Bitboard kKnightAttacks[kNumSquares]; 
    extern Bitboard kKingAttacks[kNumSquares];   
    extern Bitboard kRayAttacks[kNumSquares][8]; 
    extern Bitboard kLineBetween[kNumSquares][kNumSquares];
    
    extern Bitboard kLineMask[kNumSquares][kNumSquares];

    extern Bitboard kBackRankMasks[4];
    extern Bitboard kSecondRankMasks[4];
    extern Bitboard kCentralMask;
    
    // Moved extern here so inline MakeMove can see it
    extern int kInitialRookSq[4][2];
    extern Bitboard kCastlingEmptyMask[4][2];
    extern Bitboard kCastlingAttackMask[4][2];
    extern Bitboard kPawnPromotionMask[4];

    enum RayDirection { D_NE, D_NW, D_SE, D_SW, D_N, D_E, D_S, D_W }; 
}

constexpr int kNumPieceTypes = 6;

enum PieceType : int8_t {
  PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5,
  NO_PIECE = 6,
};

constexpr int kPieceEvaluations[6] = { 50, 300, 400, 500, 1000, 10000 };

enum PlayerColor : int8_t {
  UNINITIALIZED_PLAYER = -1,
  RED = 0, BLUE = 1, YELLOW = 2, GREEN = 3,
};

enum Team : int8_t {
  RED_YELLOW = 0, BLUE_GREEN = 1, NO_TEAM = 2, CURRENT_TEAM = 3,
};

class Board;
class Move;

struct SafetyInfo {
    Bitboard checkers;
    Bitboard pinned;
    Bitboard pinners;
};

// Helper for SEE
int StaticExchangeEvaluationCapture(const int piece_evaluations[6], const Board& board, const Move& move);
int SeeRecursive(const Board& board, const int piece_evaluations[6], int target_sq, Bitboard occupied, Team side_to_attack, int victim_value);
int GetLeastValuableAttacker(const Board& board, int sq, Team team, const Bitboard& occupied, PieceType& out_type);

class Player {
 public:
  Player() : color_(UNINITIALIZED_PLAYER) { }
  explicit Player(PlayerColor color) : color_(color) { }

  PlayerColor GetColor() const { return color_; }
  Team GetTeam() const {
    return (color_ == RED || color_ == YELLOW) ? RED_YELLOW : BLUE_GREEN;
  }
  bool operator==(const Player& other) const { return color_ == other.color_; }
  bool operator!=(const Player& other) const { return !(*this == other); }
  friend std::ostream& operator<<(std::ostream& os, const Player& player);

 private:
  PlayerColor color_;
};

// --- MOVED HELPER FUNCTIONS UP HERE SO BOARD CAN SEE THEM ---
inline Team GetTeam(PlayerColor color) { 
    return (color == RED || color == YELLOW) ? RED_YELLOW : BLUE_GREEN; 
}
inline Team OtherTeam(Team team) { 
    return team == RED_YELLOW ? BLUE_GREEN : RED_YELLOW; 
}
inline Player GetNextPlayer(const Player& player) { 
    return Player(static_cast<PlayerColor>((player.GetColor() + 1) % 4));
}
inline Player GetPreviousPlayer(const Player& player) { 
    return Player(static_cast<PlayerColor>((player.GetColor() + 3) % 4));
}
inline Player GetPartner(const Player& player) { 
    return Player(static_cast<PlayerColor>((player.GetColor() + 2) % 4));
}
// ------------------------------------------------------------

} // namespace chess

template <>
struct std::hash<chess::Player> {
  std::size_t operator()(const chess::Player& x) const {
    return std::hash<int>()(x.GetColor());
  }
};

namespace chess {

class Piece {
 public:
  Piece() : Piece(false, RED, NO_PIECE) { }
  Piece(bool present, PlayerColor color, PieceType piece_type) {
    bits_ = (((int8_t)present) << 7) | (((int8_t)color) << 5) | (((int8_t)piece_type) << 2);
  }
  Piece(PlayerColor color, PieceType piece_type) : Piece(true, color, piece_type) { }
  Piece(Player player, PieceType piece_type) : Piece(true, player.GetColor(), piece_type) { }

  bool Present() const { return bits_ & (1 << 7); }
  bool Missing() const { return !Present(); }
  PlayerColor GetColor() const { return static_cast<PlayerColor>((bits_ & 0b01100000) >> 5); }
  PieceType GetPieceType() const { return static_cast<PieceType>((bits_ & 0b00011100) >> 2); }

  bool operator==(const Piece& other) const { return bits_ == other.bits_; }
  bool operator!=(const Piece& other) const { return bits_ != other.bits_; }
  Player GetPlayer() const { return Player(GetColor()); }
  Team GetTeam() const { return GetPlayer().GetTeam(); }
  friend std::ostream& operator<<(std::ostream& os, const Piece& piece);
  static Piece kNoPiece;

 private:
  int8_t bits_;
};

class BoardLocation {
 public:
  BoardLocation() : loc_(196) {}
  BoardLocation(int8_t row, int8_t col) {
    loc_ = (row < 0 || row >= 14 || col < 0 || col >= 14) ? 196 : 14 * row + col;
  }
  explicit BoardLocation(uint8_t raw) : loc_(raw) {}

  bool Present() const { return loc_ < 196; }
  bool Missing() const { return !Present(); }
  int8_t GetRow() const { return loc_ / 14; }
  int8_t GetCol() const { return loc_ % 14; }
  uint8_t GetRawValue() const { return loc_; }

  BoardLocation Relative(int8_t delta_row, int8_t delta_col) const {
    if (!Present()) return BoardLocation();
    return BoardLocation(GetRow() + delta_row, GetCol() + delta_col);
  }
  bool operator==(const BoardLocation& other) const { return loc_ == other.loc_; }
  bool operator!=(const BoardLocation& other) const { return loc_ != other.loc_; }
  friend std::ostream& operator<<(std::ostream& os, const BoardLocation& location);
  std::string PrettyStr() const;
  static BoardLocation kNoLocation;

 private:
  uint8_t loc_;
};

} // namespace chess

template <>
struct std::hash<chess::BoardLocation> {
  std::size_t operator()(const chess::BoardLocation& x) const {
    std::size_t hash = 14479 + 14593 * x.GetRow();
    hash += 24439 * x.GetCol();
    return hash;
  }
};

namespace chess {

enum CastlingType { KINGSIDE = 0, QUEENSIDE = 1 };

class CastlingRights {
 public:
  CastlingRights() = default;
  CastlingRights(bool kingside, bool queenside) : bits_(0b10000000 | (kingside << 6) | (queenside << 5)) { }
  bool Present() const { return bits_ & (1 << 7); }
  bool Kingside() const { return bits_ & (1 << 6); }
  bool Queenside() const { return bits_ & (1 << 5); }
  bool operator==(const CastlingRights& other) const { return bits_ == other.bits_; }
  bool operator!=(const CastlingRights& other) const { return !(*this == other); }
  static CastlingRights kMissingRights;
 private:
  int8_t bits_ = 0;
};

// COMPRESSED MOVE CLASS
class Move {
 public:
  enum MoveType { TYPE_NORMAL = 0, TYPE_PROMO = 1, TYPE_EP = 2, TYPE_CASTLING = 3 };

  Move() : data_(0) {}
  
  // Standard constructor (Normal / Capture)
  Move(BoardLocation from, BoardLocation to);

  // Promotion constructor
  Move(BoardLocation from, BoardLocation to, PieceType promo);

  // Special constructor (Type specified explicitely)
  Move(BoardLocation from, BoardLocation to, MoveType type);

  // Compatibility constructor for Castling logic in generator
  static Move MakeCastling(BoardLocation from, BoardLocation to) {
      return Move(from, to, TYPE_CASTLING);
  }
  
  // En Passant now requires the capture location (victim) to handle perpendicular moves correctly
  static Move MakeEnPassant(BoardLocation from, BoardLocation to, BoardLocation capture_loc);

  bool Present() const { return data_ != 0; }

  // OPTIMIZATION: Direct Index Access
  inline int FromIndex() const { return data_ & 0xFF; }
  inline int ToIndex() const { return (data_ >> 8) & 0xFF; }

  BoardLocation From() const { return BitboardImpl::IndexToLocation(FromIndex()); }
  BoardLocation To() const { return BitboardImpl::IndexToLocation(ToIndex()); }
  
  MoveType Type() const { return static_cast<MoveType>((data_ >> 16) & 3); }
  
  bool IsCastling() const { return Type() == TYPE_CASTLING; }
  bool IsEnPassant() const { return Type() == TYPE_EP; }
  bool IsPromotion() const { return Type() == TYPE_PROMO; }

  PieceType GetPromotionPieceType() const { 
      if (!IsPromotion()) return NO_PIECE;
      return static_cast<PieceType>((data_ >> 18) & 0x7);
  }
  
  // Returns the location of the victim pawn for EP moves
  BoardLocation GetEnpassantLocation() const;

  bool operator==(const Move& other) const { return data_ == other.data_; }
  bool operator!=(const Move& other) const { return data_ != other.data_; }
  
  int ManhattanDistance() const;
  friend std::ostream& operator<<(std::ostream& os, const Move& move);
  std::string PrettyStr() const;
  bool DeliversCheck(Board& board);
  int SEE(Board& board, const int* piece_evaluations);
  int ApproxSEE(const Board& board, const int* piece_evaluations) const;

 private:
  uint32_t data_;
  int8_t delivers_check_ = -1;
  static constexpr int kSeeNotSet = -9999999;
  int see_ = kSeeNotSet;
};

struct ExtMove : public Move {
    int value;
    ExtMove() = default;
    ExtMove(const Move& m) : Move(m), value(0) {}
    ExtMove(Move&& m) : Move(std::move(m)), value(0) {}
    void operator=(const Move& m) { *static_cast<Move*>(this) = m; value = 0; }
};

enum GameResult { IN_PROGRESS = 0, WIN_RY = 1, WIN_BG = 2, STALEMATE = 3 };

class PlacedPiece {
 public:
  PlacedPiece() = default;
  PlacedPiece(const BoardLocation& location, const Piece& piece) : location_(location), piece_(piece) { }
  const BoardLocation& GetLocation() const { return location_; }
  const Piece& GetPiece() const { return piece_; }
  friend std::ostream& operator<<(std::ostream& os, const PlacedPiece& placed_piece);
 private:
  BoardLocation location_;
  Piece piece_;
};

struct EnpassantInitialization {
    int target_indices[4] = {-1, -1, -1, -1};
};

struct UndoInfo {
    Piece captured_piece;
    CastlingRights castling_rights[4]; 
    uint8_t prev_ep_target; 
};

class Board {
 public:
  Board(Player turn, std::unordered_map<BoardLocation, Piece> location_to_piece,
        std::optional<std::unordered_map<Player, CastlingRights>> castling_rights = std::nullopt,
        std::optional<EnpassantInitialization> enp = std::nullopt);
  Board(const Board&) = default;

  // Static initialization helper for Zobrist hashes
  static void InitializeStaticHashes();

  SafetyInfo CalculateSafety(PlayerColor us) const;
  template<PlayerColor Us> SafetyInfo CalculateSafetyT() const;

  // Optimized IsLegal for move generation filtering
  bool IsLegal(const Move& move, const SafetyInfo& safety) const {
    if (!move.Present()) return false;
    PlayerColor us = turn_.GetColor();
    int from_sq = move.FromIndex();
    int to_sq = move.ToIndex();
    // Use kLocationToIndex table directly
    int king_sq = BitboardImpl::kLocationToIndex[GetKingLocation(us).GetRawValue()];
    
    if (move.IsEnPassant()) {
        int cap_sq = BitboardImpl::kLocationToIndex[move.GetEnpassantLocation().GetRawValue()];
        
        // Copy by value, but modify cheaply
        Bitboard occupied = (team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN]);
        
        occupied.clear_bit(from_sq);
        occupied.clear_bit(cap_sq);
        occupied.set_bit(to_sq);

        // Helper call
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

    // King move checks
    if (GetPiece(from_sq).GetPieceType() == KING) {
        if (move.IsCastling()) {
            if (!safety.checkers.is_zero()) return false;
            return true;
        }
        Bitboard occupied = (team_bitboards_[RED_YELLOW] | team_bitboards_[BLUE_GREEN]);
        occupied.clear_bit(from_sq); // King moves out

        // Helper call
        if (AttackersToExist(to_sq, occupied, OtherTeam(turn_.GetTeam()))) return false;
        return true;
    }

    // Standard legality checks using precalculated safety info
    if (!safety.checkers.is_zero() && (safety.checkers & (safety.checkers - Bitboard(1))).operator bool()) return false;

    // Pin check using fast test()
    if (safety.pinned.test(from_sq)) {
        if (!BitboardImpl::kLineMask[king_sq][from_sq].test(to_sq)) {
            return false;
        }
    }

    if (!safety.checkers.is_zero()) {
        int checker_sq = safety.checkers.ctz();
        if (to_sq == checker_sq) return true; 
        if (!BitboardImpl::kLineBetween[king_sq][checker_sq].test(to_sq)) return false;
    }
    return true;
  }
  bool IsLegal(const Move& move) const;

  ExtMove* GetPseudoLegalMoves2(ExtMove* buffer) const;
  template<PlayerColor Us> ExtMove* GenerateMovesT(ExtMove* buffer, const SafetyInfo& safety) const;

  bool IsKingInCheck(const Player& player) const;
  bool IsKingInCheck(Team team) const;

  GameResult CheckWasLastMoveKingCapture() const;
  GameResult GetGameResult();

  Team TeamToPlay() const;
  int PieceEvaluation() const;
  int PieceEvaluation(PlayerColor color) const;
  int MobilityEvaluation();
  int MobilityEvaluation(const Player& player);
  const Player& GetTurn() const { return turn_; }
  bool IsAttackedByTeam(Team team, int sq) const;

  Bitboard GetAttackersBB(int sq, Team team) const;

  BoardLocation GetKingLocation(PlayerColor color) const;
  bool DeliversCheck(const Move& move);
  Piece GetPiece(const BoardLocation& location) const;
  Piece GetPiece(int index) const;
  bool DiscoversCheck(const Move& move) const;
  int64_t HashKey() const { return hash_key_; }
  static std::shared_ptr<Board> CreateStandardSetup();
  const CastlingRights& GetCastlingRights(const Player& player) const;

  // INLINED FUNCTIONS (Moved from .cc)
  inline void MakeMove(const Move& move);
  inline void UndoMove(const Move& move);

  bool LastMoveWasCapture() const {
      if (move_history_ptr_ == 0) return false;
      return undo_stack_[move_history_ptr_ - 1].captured_piece.Present();
  }
  
  int NumMoves() const { return move_history_ptr_; }
  
  void SetPlayer(const Player& player);
  void MakeNullMove();
  void UndoNullMove();
  
  const uint8_t* GetEnPassantTargets() const { return en_passant_target_; }

  friend class AlphaBetaPlayer;
  friend int StaticExchangeEvaluationCapture(const int[6], const Board&, const Move&);
  friend int SeeRecursive(const Board&, const int[6], int, Bitboard, Team, int);
  friend int GetLeastValuableAttacker(const Board&, int, Team, const Bitboard&, PieceType&);

  bool IsPinned(int sq, const SafetyInfo& safety) const {
      return safety.pinned.test(sq);
  }
 
 private:
  template<PlayerColor Us> ExtMove* GetPawnMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetKnightMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetBishopMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetRookMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetQueenMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetKingMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  
  // Helper functions now inline too for MakeMove/UndoMove
  inline void SetPiece(const BoardLocation& location, const Piece& piece);
  inline void RemovePiece(const BoardLocation& location);
  inline void MovePiece(const BoardLocation& from, const BoardLocation& to);
  
  // Overloads for index usage
  inline void SetPiece(int index, const Piece& piece);
  inline void RemovePiece(int index);
  inline void MovePiece(int from_idx, int to_idx);

  Bitboard GetRookAttacks(int sq, const Bitboard& blockers) const;
  Bitboard GetBishopAttacks(int sq, const Bitboard& blockers) const;
  Bitboard GetQueenAttacks(int sq, const Bitboard& blockers) const;

  bool AttackersToExist(int sq, const Bitboard& occupied, Team team) const;

  void InitializeHash();
  void UpdatePieceHash(const Piece& piece, int index) {
    hash_key_ ^= piece_hashes_[piece.GetColor()][piece.GetPieceType()][index];
  }
  void UpdateTurnHash(int turn) {
    hash_key_ ^= turn_hashes_[turn];
  }

  Player turn_;

  Bitboard piece_bitboards_[4][6];
  Bitboard color_bitboards_[4];   
  Bitboard team_bitboards_[2];     

  static constexpr int kMaxGameDepth = 4096;
  
  Piece piece_on_square_[256];
  
  CastlingRights castling_rights_[4];
  
  uint8_t en_passant_target_[4];
  
  UndoInfo undo_stack_[kMaxGameDepth]; 
  int move_history_ptr_ = 0;
  
  int piece_evaluation_ = 0;
  int player_piece_evaluations_[4] = {0, 0, 0, 0};

  int64_t hash_key_ = 0;
  
  // Static hash arrays to reduce object size
  static int64_t piece_hashes_[4][6][256];
  static int64_t turn_hashes_[4];
  static int64_t en_passant_hashes_[256];
  static int64_t castling_hashes_[4][2]; 
};

namespace BitboardImpl {
inline int LocationToIndex(const BoardLocation& loc) {
    return kLocationToIndex[loc.GetRawValue()];
}
inline BoardLocation IndexToLocation(int index) {
    return kIndexToLocation[index & 0xFF];
}
inline const Bitboard& IndexToBitboard(int index) {
    if (index < 0 || index >= kNumSquares) return kZero;
    return kSquareBitboards[index];
}
} 

// ----------------------------------------------------------------------------
// INLINE IMPLEMENTATIONS
// ----------------------------------------------------------------------------

inline void Board::SetPiece(int index, const Piece& piece) {
    if (index < 0 || !piece.Present()) return;
    const Bitboard& mask = BitboardImpl::IndexToBitboard(index);
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
// Overload for legacy support (or refactor calls)
inline void Board::SetPiece(const BoardLocation& location, const Piece& piece) {
    SetPiece(BitboardImpl::LocationToIndex(location), piece);
}

inline void Board::RemovePiece(int index) {
    Piece piece = GetPiece(index);
    if (index < 0 || !piece.Present()) return;
    
    // Optimization: bitwise NOT of reference
    const Bitboard& mask = BitboardImpl::IndexToBitboard(index); 
    
    PlayerColor color = piece.GetColor();
    PieceType type = piece.GetPieceType();
    Team team = piece.GetTeam();

    piece_bitboards_[color][type] &= ~mask;
    color_bitboards_[color] &= ~mask;
    team_bitboards_[team] &= ~mask;
    
    int piece_eval = kPieceEvaluations[type];
    if (team == RED_YELLOW) piece_evaluation_ -= piece_eval;
    else piece_evaluation_ += piece_eval;
    player_piece_evaluations_[color] -= piece_eval;
    
    piece_on_square_[index] = Piece::kNoPiece;
    UpdatePieceHash(piece, index);
}
inline void Board::RemovePiece(const BoardLocation& location) {
    RemovePiece(BitboardImpl::LocationToIndex(location));
}

inline void Board::MovePiece(int from_idx, int to_idx) {
    Piece piece = GetPiece(from_idx);
    if (from_idx < 0 || to_idx < 0 || !piece.Present()) return;

    Bitboard move_mask = BitboardImpl::IndexToBitboard(from_idx) | BitboardImpl::IndexToBitboard(to_idx);
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
inline void Board::MovePiece(const BoardLocation& from, const BoardLocation& to) {
    MovePiece(BitboardImpl::LocationToIndex(from), BitboardImpl::LocationToIndex(to));
}

inline void Board::MakeMove(const Move& move) {
    // BOUNDS CHECK BEFORE WRITING
    if (move_history_ptr_ >= kMaxGameDepth) {
        std::cerr << "History overflow at depth " << move_history_ptr_ << std::endl;
        abort();
    }

    const Player player = turn_;
    // OPTIMIZATION: Use direct indices
    int from_sq = move.FromIndex();
    int to_sq = move.ToIndex();
    PlayerColor us = player.GetColor();

    // Save Undo Information
    auto& undo = undo_stack_[move_history_ptr_];
    undo.captured_piece = Piece::kNoPiece;
    std::memcpy(undo.castling_rights, castling_rights_, sizeof(castling_rights_));
    
    // Snapshot castling rights to calculate hash diff later
    CastlingRights pre_move_castling[4];
    std::memcpy(pre_move_castling, castling_rights_, sizeof(castling_rights_));
    
    // --- 1. Handle En Passant State (Clear/Store old) ---
    uint8_t old_ep = en_passant_target_[us];
    undo.prev_ep_target = old_ep;
    if (old_ep != 255) hash_key_ ^= en_passant_hashes_[old_ep];
    en_passant_target_[us] = 255; 
    // ----------------------------------------------------

    if (move.IsCastling()) {
        // Handle Castling
        int r_from_idx = -1;
        int r_to_idx = -1;
        
        // Lookup kInitialRookSq which is now extern
        BoardLocation from = BitboardImpl::IndexToLocation(from_sq);
        BoardLocation to = BitboardImpl::IndexToLocation(to_sq);

        if (player.GetColor() == RED) {
            if (to.GetCol() > from.GetCol()) { // KS
                 r_from_idx = BitboardImpl::kInitialRookSq[RED][KINGSIDE];
                 r_to_idx = from_sq + 1;
            } else { // QS
                 r_from_idx = BitboardImpl::kInitialRookSq[RED][QUEENSIDE];
                 r_to_idx = from_sq - 1;
            }
        } else if (player.GetColor() == BLUE) {
             if (to.GetRow() > from.GetRow()) { 
                 r_from_idx = BitboardImpl::kInitialRookSq[BLUE][KINGSIDE];
                 r_to_idx = from_sq + BitboardImpl::kBoardWidth;
             } else { 
                 r_from_idx = BitboardImpl::kInitialRookSq[BLUE][QUEENSIDE];
                 r_to_idx = from_sq - BitboardImpl::kBoardWidth;
             }
        } else if (player.GetColor() == YELLOW) {
             if (to.GetCol() < from.GetCol()) { 
                 r_from_idx = BitboardImpl::kInitialRookSq[YELLOW][KINGSIDE];
                 r_to_idx = from_sq - 1;
             } else { 
                 r_from_idx = BitboardImpl::kInitialRookSq[YELLOW][QUEENSIDE];
                 r_to_idx = from_sq + 1;
             }
        } else { // GREEN
             if (to.GetRow() < from.GetRow()) { 
                 r_from_idx = BitboardImpl::kInitialRookSq[GREEN][KINGSIDE];
                 r_to_idx = from_sq - BitboardImpl::kBoardWidth;
             } else { 
                 r_from_idx = BitboardImpl::kInitialRookSq[GREEN][QUEENSIDE];
                 r_to_idx = from_sq + BitboardImpl::kBoardWidth;
             }
        }
        
        MovePiece(from_sq, to_sq); // Move King
        MovePiece(r_from_idx, r_to_idx); // Move Rook
        
        castling_rights_[player.GetColor()] = CastlingRights(false, false);

    } else if (move.IsEnPassant()) {
        // Extract the explicit capture location from the move object
        // NOTE: GetEnpassantLocation uses bit shifting on data_, it's fast.
        int cap_idx = BitboardImpl::LocationToIndex(move.GetEnpassantLocation());
        
        undo.captured_piece = GetPiece(cap_idx); 
        RemovePiece(cap_idx);
        MovePiece(from_sq, to_sq);
        
    } else {
        // Normal or Promo
        if (!GetPiece(to_sq).Missing()) {
            undo.captured_piece = GetPiece(to_sq);
            RemovePiece(to_sq);
        }
        
        if (move.IsPromotion()) {
            RemovePiece(from_sq);
            SetPiece(to_sq, Piece(player.GetColor(), move.GetPromotionPieceType()));
        } else {
            MovePiece(from_sq, to_sq);
        }
        
        // Update Castling Rights
        if (GetPiece(to_sq).GetPieceType() == KING) {
             castling_rights_[player.GetColor()] = CastlingRights(false, false);
        }
        if (castling_rights_[player.GetColor()].Present()) {
             if (from_sq == BitboardImpl::kInitialRookSq[player.GetColor()][KINGSIDE]) 
                 castling_rights_[player.GetColor()] = CastlingRights(false, castling_rights_[player.GetColor()].Queenside());
             else if (from_sq == BitboardImpl::kInitialRookSq[player.GetColor()][QUEENSIDE])
                 castling_rights_[player.GetColor()] = CastlingRights(castling_rights_[player.GetColor()].Kingside(), false);
        }
        if (undo.captured_piece.GetPieceType() == ROOK) {
             PlayerColor enemy = undo.captured_piece.GetColor();
             if (castling_rights_[enemy].Present()) {
                 if (to_sq == BitboardImpl::kInitialRookSq[enemy][KINGSIDE])
                     castling_rights_[enemy] = CastlingRights(false, castling_rights_[enemy].Queenside());
                 else if (to_sq == BitboardImpl::kInitialRookSq[enemy][QUEENSIDE])
                     castling_rights_[enemy] = CastlingRights(castling_rights_[enemy].Kingside(), false);
             }
        }
    }

    // --- 2. Set New En Passant State ---
    // If pawn moves 2 squares, set new target
    // Optimize: ManhattanDistance checks abs diff.
    if (!move.IsPromotion() && GetPiece(to_sq).GetPieceType() == PAWN && move.ManhattanDistance() == 2) {
         // Midpoint is the target
         int mid_sq = (from_sq + to_sq) / 2;
         en_passant_target_[us] = (uint8_t)mid_sq;
         hash_key_ ^= en_passant_hashes_[mid_sq];
    }
    // -----------------------------------
    
    // Update Castling Hash
    for (int c = 0; c < 4; ++c) {
        if (pre_move_castling[c] != castling_rights_[c]) {
            if (pre_move_castling[c].Kingside() ^ castling_rights_[c].Kingside()) 
                hash_key_ ^= castling_hashes_[c][KINGSIDE];
            if (pre_move_castling[c].Queenside() ^ castling_rights_[c].Queenside()) 
                hash_key_ ^= castling_hashes_[c][QUEENSIDE];
        }
    }
    
    int t = static_cast<int>(turn_.GetColor());
    UpdateTurnHash(t);
    turn_ = GetNextPlayer(turn_);
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    move_history_ptr_++;
}

inline void Board::UndoMove(const Move& move) {
    assert(move_history_ptr_ > 0);
    --move_history_ptr_;
    const auto& undo = undo_stack_[move_history_ptr_];
    
    Player turn_before = GetPreviousPlayer(turn_);
    PlayerColor us = turn_before.GetColor();

    UpdateTurnHash(static_cast<int>(turn_.GetColor()));
    turn_ = turn_before;
    UpdateTurnHash(static_cast<int>(turn_.GetColor()));

    // OPTIMIZATION: Use direct indices
    int from_sq = move.FromIndex();
    int to_sq = move.ToIndex();
    
    // --- 1. Restore En Passant State ---
    uint8_t current_ep = en_passant_target_[us];
    if (current_ep != 255) {
        hash_key_ ^= en_passant_hashes_[current_ep];
        en_passant_target_[us] = 255;
    }
    uint8_t old_ep = undo.prev_ep_target;
    en_passant_target_[us] = old_ep;
    if (old_ep != 255) hash_key_ ^= en_passant_hashes_[old_ep];
    // -----------------------------------

    // Update Castling Hash
    for (int c = 0; c < 4; ++c) {
        if (castling_rights_[c] != undo.castling_rights[c]) {
            if (castling_rights_[c].Kingside() ^ undo.castling_rights[c].Kingside()) 
                hash_key_ ^= castling_hashes_[c][KINGSIDE];
            if (castling_rights_[c].Queenside() ^ undo.castling_rights[c].Queenside()) 
                hash_key_ ^= castling_hashes_[c][QUEENSIDE];
        }
    }

    // Restore Castling Rights
    std::memcpy(castling_rights_, undo.castling_rights, sizeof(castling_rights_));

    if (move.IsCastling()) {
        int r_from_idx = -1;
        int r_to_idx = -1;
        BoardLocation from = BitboardImpl::IndexToLocation(from_sq);
        BoardLocation to = BitboardImpl::IndexToLocation(to_sq);

        if (turn_before.GetColor() == RED) {
            if (to.GetCol() > from.GetCol()) { r_from_idx = BitboardImpl::kInitialRookSq[RED][KINGSIDE]; r_to_idx = from_sq + 1; } 
            else { r_from_idx = BitboardImpl::kInitialRookSq[RED][QUEENSIDE]; r_to_idx = from_sq - 1; }
        } else if (turn_before.GetColor() == BLUE) {
             if (to.GetRow() > from.GetRow()) { r_from_idx = BitboardImpl::kInitialRookSq[BLUE][KINGSIDE]; r_to_idx = from_sq + BitboardImpl::kBoardWidth; } 
             else { r_from_idx = BitboardImpl::kInitialRookSq[BLUE][QUEENSIDE]; r_to_idx = from_sq - BitboardImpl::kBoardWidth; }
        } else if (turn_before.GetColor() == YELLOW) {
             if (to.GetCol() < from.GetCol()) { r_from_idx = BitboardImpl::kInitialRookSq[YELLOW][KINGSIDE]; r_to_idx = from_sq - 1; } 
             else { r_from_idx = BitboardImpl::kInitialRookSq[YELLOW][QUEENSIDE]; r_to_idx = from_sq + 1; }
        } else { 
             if (to.GetRow() < from.GetRow()) { r_from_idx = BitboardImpl::kInitialRookSq[GREEN][KINGSIDE]; r_to_idx = from_sq - BitboardImpl::kBoardWidth; } 
             else { r_from_idx = BitboardImpl::kInitialRookSq[GREEN][QUEENSIDE]; r_to_idx = from_sq + BitboardImpl::kBoardWidth; }
        }
        MovePiece(to_sq, from_sq); // King back
        MovePiece(r_to_idx, r_from_idx); // Rook back
        
    } else if (move.IsEnPassant()) {
        MovePiece(to_sq, from_sq); // Pawn back
        int cap_idx = BitboardImpl::LocationToIndex(move.GetEnpassantLocation());
        SetPiece(cap_idx, undo.captured_piece);
        
    } else {
        // Normal/Promo
        if (move.IsPromotion()) {
            RemovePiece(to_sq);
            SetPiece(from_sq, Piece(turn_before.GetColor(), PAWN));
        } else {
            MovePiece(to_sq, from_sq);
        }
        
        if (undo.captured_piece.Present()) {
            SetPiece(to_sq, undo.captured_piece);
        }
    }
}

}  // namespace chess

template <>
struct std::hash<chess::Move> {
  std::size_t operator()(const chess::Move& m) const {
      return std::hash<uint32_t>()(m.From().GetRawValue() | (m.To().GetRawValue() << 8));
  }
};

#endif  // _BOARD_H_