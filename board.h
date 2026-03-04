#ifndef _BOARD_H_
#define _BOARD_H_

// Classes for a 4-player teams chess board (chess.com variant).

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <array>
#include <cstdint>

#include "types.h"

namespace chess {

// Forward declarations
class Board;
class NNUE;
struct Accumulator;

// Move or capture. Does not include pawn promotion, en-passant, or castling.
class SimpleMove {
 public:
  SimpleMove() = default;

  SimpleMove(BoardLocation from,
             BoardLocation to)
    : from_(std::move(from)),
      to_(std::move(to))
  { }

  bool Present() const { return from_.Present() && to_.Present(); }
  const BoardLocation& From() const { return from_; }
  const BoardLocation& To() const { return to_; }

  bool operator==(const SimpleMove& other) const {
    return from_ == other.from_
        && to_ == other.to_;
  }

  bool operator!=(const SimpleMove& other) const {
    return !(*this == other);
  }

 private:
  BoardLocation from_;
  BoardLocation to_;
};

enum CastlingType {
  KINGSIDE = 0, QUEENSIDE = 1,
};

class CastlingRights {
 public:
  CastlingRights() = default;

  CastlingRights(bool kingside, bool queenside)
    : bits_(0b10000000 | (kingside << 6) | (queenside << 5)) { }

  bool Present() const { return bits_ & (1 << 7); }
  bool Kingside() const { return bits_ & (1 << 6); }
  bool Queenside() const { return bits_ & (1 << 5); }

  bool operator==(const CastlingRights& other) const {
    return bits_ == other.bits_;
  }
  bool operator!=(const CastlingRights& other) const {
    return !(*this == other);
  }

  static CastlingRights kMissingRights;

 private:
  // bit 0: presence
  // bit 1: kingside
  // bit 2: queenside
  int8_t bits_ = 0;
};

class Move {
 public:
  Move() = default;

  // Standard move
  Move(BoardLocation from, BoardLocation to,
       Piece standard_capture = Piece::kNoPiece,
       CastlingRights initial_castling_rights = CastlingRights::kMissingRights,
       CastlingRights castling_rights = CastlingRights::kMissingRights)
    : from_(std::move(from)),
      to_(std::move(to)),
      standard_capture_(standard_capture),
      initial_castling_rights_(std::move(initial_castling_rights)),
      castling_rights_(std::move(castling_rights))
  { }

  // Pawn move
  Move(BoardLocation from, BoardLocation to,
       Piece standard_capture,
       BoardLocation en_passant_location,
       Piece en_passant_capture,
       PieceType promotion_piece_type = NO_PIECE)
    : from_(std::move(from)),
      to_(std::move(to)),
      standard_capture_(standard_capture),
      promotion_piece_type_(promotion_piece_type),
      en_passant_location_(en_passant_location),
      en_passant_capture_(en_passant_capture)
  { }

  // Castling
  Move(BoardLocation from, BoardLocation to,
       SimpleMove rook_move,
       CastlingRights initial_castling_rights,
       CastlingRights castling_rights)
    : from_(std::move(from)),
      to_(std::move(to)),
      rook_move_(rook_move),
      initial_castling_rights_(std::move(initial_castling_rights)),
      castling_rights_(std::move(castling_rights))
  { }

  const BoardLocation& From() const { return from_; }
  const BoardLocation& To() const { return to_; }
  bool Present() const { return from_.Present() && to_.Present(); }
  Piece GetStandardCapture() const {
    return standard_capture_;
  }
  bool IsStandardCapture() const {
    return standard_capture_.Present();
  }
  PieceType GetPromotionPieceType() const {
    return promotion_piece_type_;
  }
  const BoardLocation GetEnpassantLocation() const {
    return en_passant_location_;
  }
  Piece GetEnpassantCapture() const {
    return en_passant_capture_;
  }
  SimpleMove GetRookMove() const { return rook_move_; }
  CastlingRights GetInitialCastlingRights() const {
    return initial_castling_rights_;
  }
  CastlingRights GetCastlingRights() const {
    return castling_rights_;
  }

  bool IsCapture() const {
    return standard_capture_.Present() || en_passant_capture_.Present();
  }
  Piece GetCapturePiece() const {
    return standard_capture_.Present() ? standard_capture_ : en_passant_capture_;
  }

  bool operator==(const Move& other) const; // Implementation in board.cc
  bool operator!=(const Move& other) const {
    return !(*this == other);
  }
  int ManhattanDistance() const;
  friend std::ostream& operator<<(std::ostream& os, const Move& move);
  std::string PrettyStr() const;
  
  // NOTE: This does not find discovered checks.
  bool DeliversCheck(Board& board);
  int SEE(Board& board, const int* piece_evaluations);
  int ApproxSEE(Board& board, const int* piece_evaluations);

  // NEW: Store the en passant target square that was cleared by this move.
  const BoardLocation& GetPreviousEnPassantTarget() const { return previous_en_passant_target_; }
  void SetPreviousEnPassantTarget(const BoardLocation& loc) { previous_en_passant_target_ = loc; }

 private:
  BoardLocation from_;  // 1
  BoardLocation to_;  // 1

  // Capture
  Piece standard_capture_; // 1

  // Promotion
  PieceType promotion_piece_type_ = NO_PIECE; // 1

  // En-passant
  BoardLocation en_passant_location_; // 1
  Piece en_passant_capture_;  // 1

  // For castling moves
  SimpleMove rook_move_; // 2

  // Castling rights before the move
  CastlingRights initial_castling_rights_; // 1

  // Castling rights after the move
  CastlingRights castling_rights_; // 1

  // Store the en passant target square that was cleared by this move.
  BoardLocation previous_en_passant_target_; // 1

  // Cached check
  // -1 means missing, 0/1 store check values
  int8_t delivers_check_ = -1; // 1

  static constexpr int kSeeNotSet = -9999999;

  // Static exchange value of the move
  int see_ = kSeeNotSet;
};

enum GameResult {
  IN_PROGRESS = 0,
  WIN_RY = 1,
  WIN_BG = 2,
  STALEMATE = 3,
};

struct MoveBuffer {
  Move* buffer = nullptr;
  size_t pos = 0;
  size_t limit = 0;

  template<class... T>
  void emplace_back(T&&... args) {
    if (pos >= limit) {
      std::cout << "Move buffer overflow" << std::endl;
      abort();
    }
    buffer[pos++] = Move(std::forward<T>(args)...);
  }
};

enum SetupType {
  MODERN,
  CLASSIC,
};

// Forward declare for friend function
std::string GenerateFENFromBoard(const Board& board);

class Board {
 // Conventions:
 // - Red is on the bottom of the board, blue on the left, yellow on top,
 //   green on the right
 // - Rows go downward from the top
 // - Columns go rightward from the left

 public:
  Board(
      Player turn,
      std::unordered_map<BoardLocation, Piece> location_to_piece,
      std::optional<std::unordered_map<Player, CastlingRights>>
        castling_rights = std::nullopt,
      // NEW: En passant state is now passed directly to the constructor.
      std::optional<std::array<BoardLocation, 4>> en_passant_targets = std::nullopt);

  Board(const Board&) = default;

  size_t GetPseudoLegalMoves2(Move* buffer, size_t limit);

  bool IsKingInCheck(const Player& player) const;
  bool IsKingInCheck(Team team) const;

  GameResult CheckWasLastMoveKingCapture() const;
  GameResult GetGameResult(); // Avoid calling during search.

  Team TeamToPlay() const;
  int PieceEvaluation() const;
  int PieceEvaluation(PlayerColor color) const;
  int MobilityEvaluation();
  int MobilityEvaluation(const Player& player);
  const Player& GetTurn() const { return turn_; }
  bool IsAttackedByTeam(
      Team team,
      const BoardLocation& location) const;

  size_t GetAttackers2(
      PlacedPiece* buffer, size_t limit,
      Team team, const BoardLocation& location) const;

  BoardLocation GetKingLocation(PlayerColor color) const;
  bool DeliversCheck(const Move& move);

  const Piece& GetPiece(
      int row, int col) const {
    // OPTIMIZATION: Use 1D array access.
    return location_to_piece_[14 * row + col];
  }
  const Piece& GetPiece(
      const BoardLocation& location) const {
    // OPTIMIZATION: Use 1D array access with raw index.
    return location_to_piece_[location.GetIndex()];
  }
  inline bool IsOnPathBetween(
      const BoardLocation& from,
      const BoardLocation& to,
      const BoardLocation& between) const;
  inline bool DiscoversCheck(
      const BoardLocation& king_location,
      const BoardLocation& move_from,
      const BoardLocation& move_to,
      Team attacking_team) const;

  int64_t HashKey() const { return hash_key_; }

  static std::shared_ptr<Board> CreateStandardSetup(SetupType setup = MODERN);
  const CastlingRights& GetCastlingRights(const Player& player) const;

  void MakeMove(const Move& move);
  void UndoMove();

  // NEW: Update NNUE Accumulator incrementally
  void UpdateAccumulator(const Move& move, const NNUE& nnue, Accumulator& acc) const;
  
  bool LastMoveWasCapture() const {
    return !moves_.empty() && moves_.back().GetStandardCapture().Present();
  }
  const Move& GetLastMove() const {
    return moves_.back();
  }
  int NumMoves() const { return moves_.size(); }
  const std::vector<Move>& Moves() { return moves_; }


  void GetPawnMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void GetKnightMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void GetBishopMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void GetRookMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void GetQueenMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void GetKingMoves2(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void AddMovesFromIncrMovement2(
      MoveBuffer& moves,
      const Piece& piece,
      const BoardLocation& from,
      int incr_row,
      int incr_col,
      CastlingRights initial_castling_rights = CastlingRights::kMissingRights,
      CastlingRights castling_rights = CastlingRights::kMissingRights) const;

  friend std::ostream& operator<<(
      std::ostream& os, const Board& board);

  // NEW: Friend function to access private members for FEN generation
  friend std::string GenerateFENFromBoard(const Board& board);

  // Use with caution: after you set the player you must reset it to its
  // original value before calling UndoMove past the current moves.
  // These functions may be used by things such as null move pruning.
  void SetPlayer(const Player& player) { turn_ = player; }
  void MakeNullMove();
  void UndoNullMove();

  bool IsLegalLocation(int row, int col) const;
  bool IsLegalLocation(const BoardLocation& location) const;

  const std::vector<std::vector<PlacedPiece>>& GetPieceList() const { return piece_list_; }
  
  // NEW: Provides a flattened array of pieces needed for NNUE root initialization
  std::vector<PlacedPiece> GetPieceListFlat() const;

  // NEW: Getter for testing en passant state.
  const BoardLocation& GetEnPassantTarget(PlayerColor color) const {
    return en_passant_target_[color];
  }

 private:
  void AddMovesFromIncrMovement(
      std::vector<Move>& moves,
      const Piece& piece,
      const BoardLocation& from,
      int incr_row,
      int incr_col,
      CastlingRights initial_castling_rights = CastlingRights::kMissingRights,
      CastlingRights castling_rights = CastlingRights::kMissingRights) const;
  void AddModernCastlingMoves(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  void AddClassicCastlingMoves(
      MoveBuffer& moves,
      const BoardLocation& from,
      const Piece& piece) const;
  int GetMaxRow() const { return 13; }
  int GetMaxCol() const { return 13; }
  std::optional<CastlingType> GetRookLocationType(
      const Player& player, const BoardLocation& location) const;
  inline void SetPiece(const BoardLocation& location,
                const Piece& piece);
  inline void RemovePiece(const BoardLocation& location);
  inline bool QueenAttacks(
      const BoardLocation& queen_loc,
      const BoardLocation& other_loc) const;
  inline bool RookAttacks(
      const BoardLocation& rook_loc,
      const BoardLocation& other_loc) const;
  inline bool BishopAttacks(
      const BoardLocation& bishop_loc,
      const BoardLocation& other_loc) const;
  inline bool KingAttacks(
      const BoardLocation& king_loc,
      const BoardLocation& other_loc) const;
  inline bool KnightAttacks(
      const BoardLocation& knight_loc,
      const BoardLocation& other_loc) const;
  inline bool PawnAttacks(
      const BoardLocation& pawn_loc,
      PlayerColor pawn_color,
      const BoardLocation& other_loc) const;

  void InitializeHash();
  void UpdatePieceHash(const Piece& piece, const BoardLocation& loc) {
    hash_key_ ^= piece_hashes_[piece.GetColor()][piece.GetPieceType()]
      [loc.GetRow()][loc.GetCol()];
  }
  void UpdateTurnHash(int turn) {
    hash_key_ ^= turn_hashes_[turn];
  }
  // NEW: Zobrist hash updates for en passant and castling rights.
  void UpdateEnPassantHash(const BoardLocation& loc) {
    hash_key_ ^= en_passant_hashes_[loc.GetRow()][loc.GetCol()];
  }
  void UpdateCastlingHash(PlayerColor color, CastlingType type) {
    hash_key_ ^= castling_hashes_[color][type];
  }

  Player turn_;

  // OPTIMIZATION: Use 1D mailbox array
  Piece location_to_piece_[196];
  // OPTIMIZATION: Index into piece_list_ for O(1) updates
  int8_t piece_indices_[196];
  std::vector<std::vector<PlacedPiece>> piece_list_;

  BoardLocation locations_[14][14];
  SetupType setup_type_;
  CastlingRights castling_rights_[4];
  // NEW: Each color has a potential en passant target square.
  BoardLocation en_passant_target_[4];

  std::vector<Move> moves_; // list of moves from beginning of game
  std::vector<Move> move_buffer_;
  int piece_evaluation_ = 0;
  int player_piece_evaluations_[4] = {0, 0, 0, 0}; // one per player

  int64_t hash_key_ = 0;
  int64_t piece_hashes_[4][6][14][14];
  int64_t turn_hashes_[4];
  // NEW: Zobrist keys for en passant and castling rights.
  int64_t en_passant_hashes_[14][14];
  int64_t castling_hashes_[4][2]; // [color][KINGSIDE/QUEENSIDE]

  BoardLocation king_locations_[4];

  size_t move_buffer_size_ = 300;
  Move move_buffer_2_[300];
};

// Helper functions (SEE)
int StaticExchangeEvaluationCapture(
    const int piece_evaluations[6],
    Board& board,
    const Move& move);

}  // namespace chess

// NEW HASH FUNCTION FOR MOVE
template <>
struct std::hash<chess::Move>
{
  std::size_t operator()(const chess::Move& m) const
  {
    std::size_t h1 = std::hash<chess::BoardLocation>()(m.From());
    std::size_t h2 = std::hash<chess::BoardLocation>()(m.To());
    std::size_t h3 = std::hash<int>()(m.GetPromotionPieceType());
    // Combine the hashes using XOR and bit shifts
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

#endif  // _BOARD_H_