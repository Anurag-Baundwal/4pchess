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

class SimpleMove {
 public:
  SimpleMove() = default;
  SimpleMove(BoardLocation from, BoardLocation to) : from_(from), to_(to) { }
  bool Present() const { return from_.Present() && to_.Present(); }
  const BoardLocation& From() const { return from_; }
  const BoardLocation& To() const { return to_; }
  bool operator==(const SimpleMove& other) const { return from_ == other.from_ && to_ == other.to_; }
  bool operator!=(const SimpleMove& other) const { return !(*this == other); }
 private:
  BoardLocation from_;
  BoardLocation to_;
};

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

  BoardLocation From() const { return BitboardImpl::IndexToLocation(data_ & 0xFF); }
  BoardLocation To() const { return BitboardImpl::IndexToLocation((data_ >> 8) & 0xFF); }
  
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

// Simplified initialization struct to hold just indices
struct EnpassantInitialization {
    int target_indices[4] = {-1, -1, -1, -1};
};

struct UndoInfo {
    Piece captured_piece;
    CastlingRights castling_rights[4]; 
    uint8_t prev_ep_target; // Stores the EP target index for the player who moved
};

class Board {
 public:
  Board(Player turn, std::unordered_map<BoardLocation, Piece> location_to_piece,
        std::optional<std::unordered_map<Player, CastlingRights>> castling_rights = std::nullopt,
        std::optional<EnpassantInitialization> enp = std::nullopt);
  Board(const Board&) = default;

  SafetyInfo CalculateSafety(PlayerColor us) const;
  template<PlayerColor Us> SafetyInfo CalculateSafetyT() const;

  bool IsLegal(const Move& move, const SafetyInfo& safety) const;
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

  void MakeMove(const Move& move);
  void UndoMove();

  bool LastMoveWasCapture() const {
      if (move_history_ptr_ == 0) return false;
      return undo_stack_[move_history_ptr_ - 1].captured_piece.Present();
  }
  const Move& GetLastMove() const {
    return move_history_[move_history_ptr_ - 1];
  }
  int NumMoves() const { return move_history_ptr_; }
  
  std::vector<Move> Moves() const {
      return std::vector<Move>(move_history_, move_history_ + move_history_ptr_); 
  }

  void SetPlayer(const Player& player);
  void MakeNullMove();
  void UndoNullMove();
  
  // Returns raw indices
  const uint8_t* GetEnPassantTargets() const { return en_passant_target_; }

  friend class AlphaBetaPlayer;
  friend int StaticExchangeEvaluationCapture(const int[6], const Board&, const Move&);
  friend int SeeRecursive(const Board&, const int[6], int, Bitboard, Team, int);
  friend int GetLeastValuableAttacker(const Board&, int, Team, const Bitboard&, PieceType&);

  bool IsPinned(int sq, const SafetyInfo& safety) const {
      return (safety.pinned & BitboardImpl::IndexToBitboard(sq)).operator bool();
  }
 
 private:
  template<PlayerColor Us> ExtMove* GetPawnMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetKnightMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetBishopMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetRookMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetQueenMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  template<PlayerColor Us> ExtMove* GetKingMovesT(ExtMove* buffer, const SafetyInfo& safety) const;
  
  void SetPiece(const BoardLocation& location, const Piece& piece);
  void RemovePiece(const BoardLocation& location);
  void MovePiece(const BoardLocation& from, const BoardLocation& to);

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

  static constexpr int kMaxGameDepth = 512;
  
  Piece piece_on_square_[256];
  
  CastlingRights castling_rights_[4];
  
  // NEW: Store EP targets as 256-sized indices. 255 (0xFF) = No Target.
  uint8_t en_passant_target_[4];
  
  Move move_history_[kMaxGameDepth];
  UndoInfo undo_stack_[kMaxGameDepth]; 
  int move_history_ptr_ = 0;
  
  int piece_evaluation_ = 0;
  int player_piece_evaluations_[4] = {0, 0, 0, 0};

  int64_t hash_key_ = 0;
  int64_t piece_hashes_[4][6][256];
  int64_t turn_hashes_[4];
  // NEW: Zobrist hashes
  int64_t en_passant_hashes_[256];
  int64_t castling_hashes_[4][2]; // [Color][Side] (0=KS, 1=QS)
};

Team OtherTeam(Team team);
Team GetTeam(PlayerColor color);
Player GetNextPlayer(const Player& player);
Player GetPreviousPlayer(const Player& player);
Player GetPartner(const Player& player);

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

}  // namespace chess

template <>
struct std::hash<chess::Move> {
  std::size_t operator()(const chess::Move& m) const {
      // Very simple hash for uint32
      return std::hash<uint32_t>()(m.From().GetRawValue() | (m.To().GetRawValue() << 8));
  }
};

#endif  // _BOARD_H_