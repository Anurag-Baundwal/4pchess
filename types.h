#ifndef _TYPES_H_
#define _TYPES_H_

#include <cstdint>
#include <string>
#include <vector> // Keep for potential use, though not directly used by these types alone
#include <ostream>
#include <functional> // For std::hash

namespace chess {

// --- Enums ---
enum PieceType : int8_t {
  PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5,
  NO_PIECE = 6,
};

enum PlayerColor : int8_t {
  UNINITIALIZED_PLAYER = -1,
  RED = 0, BLUE = 1, YELLOW = 2, GREEN = 3,
};

enum Team : int8_t {
  RED_YELLOW = 0, BLUE_GREEN = 1, NO_TEAM = 2, CURRENT_TEAM = 3,
};

// --- Constants ---
constexpr int kNumPieceTypes = 6;
// In centipawns
constexpr int kPieceEvaluations[6] = {
  50,     // PAWN
  300,    // KNIGHT
  400,    // BISHOP
  500,    // ROOK
  1000,   // QUEEN
  10000,  // KING (used for SEE, etc.)
};


// --- Classes ---
class Player {
 public:
  Player() : color_(UNINITIALIZED_PLAYER) { }
  explicit Player(PlayerColor color) : color_(color) { }

  PlayerColor GetColor() const { return color_; }
  Team GetTeam() const;
  bool operator==(const Player& other) const {
    return color_ == other.color_;
  }
  bool operator!=(const Player& other) const {
    return !(*this == other);
  }
  friend std::ostream& operator<<(std::ostream& os, const Player& player);

 private:
  PlayerColor color_;
};

// Player Constants
extern const Player kRedPlayer;
extern const Player kBluePlayer;
extern const Player kYellowPlayer;
extern const Player kGreenPlayer;


class BoardLocation {
 public:
  BoardLocation(); // Default constructor for kNoLocation
  BoardLocation(int8_t row, int8_t col);

  bool Present() const { return loc_ < 196; } // 14*14 = 196
  bool Missing() const { return !Present(); }
  int8_t GetRow() const { return loc_ / 14; }
  int8_t GetCol() const { return loc_ % 14; }

  BoardLocation Relative(int8_t delta_row, int8_t delta_col) const;

  bool operator==(const BoardLocation& other) const { return loc_ == other.loc_; }
  bool operator!=(const BoardLocation& other) const { return loc_ != other.loc_; }

  friend std::ostream& operator<<(std::ostream& os, const BoardLocation& location);
  std::string PrettyStr() const;

  static BoardLocation kNoLocation;

 private:
  uint8_t loc_; // Value 0-195 for valid locations, 196 for kNoLocation
};


class Piece {
 public:
  Piece(); // Default constructor for kNoPiece
  Piece(bool present, PlayerColor color, PieceType piece_type);
  Piece(PlayerColor color, PieceType piece_type);
  Piece(Player player, PieceType piece_type);

  bool Present() const { return bits_ & (1 << 7); }
  bool Missing() const { return !Present(); }
  PlayerColor GetColor() const {
    return static_cast<PlayerColor>((bits_ & 0b01100000) >> 5);
  }
  PieceType GetPieceType() const {
    return static_cast<PieceType>((bits_ & 0b00011100) >> 2);
  }

  bool operator==(const Piece& other) const { return bits_ == other.bits_; }
  bool operator!=(const Piece& other) const { return bits_ != other.bits_; }

  Player GetPlayer() const { return Player(GetColor()); }
  Team GetTeam() const;
  friend std::ostream& operator<<(std::ostream& os, const Piece& piece);

  static Piece kNoPiece;

 private:
  int8_t bits_;
};

// Piece Constants
extern const Piece kRedPawn;
extern const Piece kRedKnight;
extern const Piece kRedBishop;
extern const Piece kRedRook;
extern const Piece kRedQueen;
extern const Piece kRedKing;

extern const Piece kBluePawn;
extern const Piece kBlueKnight;
extern const Piece kBlueBishop;
extern const Piece kBlueRook;
extern const Piece kBlueQueen;
extern const Piece kBlueKing;

extern const Piece kYellowPawn;
extern const Piece kYellowKnight;
extern const Piece kYellowBishop;
extern const Piece kYellowRook;
extern const Piece kYellowQueen;
extern const Piece kYellowKing;

extern const Piece kGreenPawn;
extern const Piece kGreenKnight;
extern const Piece kGreenBishop;
extern const Piece kGreenRook;
extern const Piece kGreenQueen;
extern const Piece kGreenKing;


class PlacedPiece {
 public:
  PlacedPiece() = default;
  PlacedPiece(const BoardLocation& location, const Piece& piece)
    : location_(location), piece_(piece) { }

  const BoardLocation& GetLocation() const { return location_; }
  const Piece& GetPiece() const { return piece_; }
  friend std::ostream& operator<<(std::ostream& os, const PlacedPiece& placed_piece);

 private:
  BoardLocation location_;
  Piece piece_;
};

// --- Helper Functions Declarations ---
Team GetTeam(PlayerColor color); // Overload for just color
Player GetNextPlayer(const Player& player);
Player GetPreviousPlayer(const Player& player);
Player GetPartner(const Player& player);
Team OtherTeam(Team team);

// Internal helpers for ostream, can be in .cc if not needed externally beyond ostream
std::string ToStr(PlayerColor color);
std::string ToStr(PieceType piece_type);

}  // namespace chess


// --- Hash Specializations ---
namespace std {
template <>
struct hash<chess::Player> {
  std::size_t operator()(const chess::Player& x) const {
    return std::hash<int>()(x.GetColor());
  }
};

template <>
struct hash<chess::BoardLocation> {
  std::size_t operator()(const chess::BoardLocation& x) const {
    if (x.Missing()) { // Handle kNoLocation consistently
        return std::hash<uint8_t>()(196); 
    }
    // Use the more robust hashing from original board.h
    std::size_t hash_val = 14479 + 14593 * x.GetRow();
    hash_val += 24439 * x.GetCol();
    return hash_val;
  }
};
} // namespace std

#endif  // _TYPES_H_