#include "types.h"
#include <string> // For std::to_string
#include <stdexcept> // For std::runtime_error if needed

namespace chess {

// --- Static Member Definitions ---
BoardLocation BoardLocation::kNoLocation = BoardLocation(); // Default constructor sets loc_ to 196
Piece Piece::kNoPiece = Piece(); // Default constructor creates a non-present piece

// --- Player Constants Definitions ---
const Player kRedPlayer = Player(RED);
const Player kBluePlayer = Player(BLUE);
const Player kYellowPlayer = Player(YELLOW);
const Player kGreenPlayer = Player(GREEN);

// --- Piece Constants Definitions ---
const Piece kRedPawn(kRedPlayer, PAWN);
const Piece kRedKnight(kRedPlayer, KNIGHT);
const Piece kRedBishop(kRedPlayer, BISHOP);
const Piece kRedRook(kRedPlayer, ROOK);
const Piece kRedQueen(kRedPlayer, QUEEN);
const Piece kRedKing(kRedPlayer, KING);

const Piece kBluePawn(kBluePlayer, PAWN);
const Piece kBlueKnight(kBluePlayer, KNIGHT);
const Piece kBlueBishop(kBluePlayer, BISHOP);
const Piece kBlueRook(kBluePlayer, ROOK);
const Piece kBlueQueen(kBluePlayer, QUEEN);
const Piece kBlueKing(kBluePlayer, KING);

const Piece kYellowPawn(kYellowPlayer, PAWN);
const Piece kYellowKnight(kYellowPlayer, KNIGHT);
const Piece kYellowBishop(kYellowPlayer, BISHOP);
const Piece kYellowRook(kYellowPlayer, ROOK);
const Piece kYellowQueen(kYellowPlayer, QUEEN);
const Piece kYellowKing(kYellowPlayer, KING);

const Piece kGreenPawn(kGreenPlayer, PAWN);
const Piece kGreenKnight(kGreenPlayer, KNIGHT);
const Piece kGreenBishop(kGreenPlayer, BISHOP);
const Piece kGreenRook(kGreenPlayer, ROOK);
const Piece kGreenQueen(kGreenPlayer, QUEEN);
const Piece kGreenKing(kGreenPlayer, KING);


// --- Class Method Implementations ---

// Player
Team Player::GetTeam() const {
  return (color_ == RED || color_ == YELLOW) ? RED_YELLOW : BLUE_GREEN;
}

// BoardLocation
BoardLocation::BoardLocation() : loc_(196) {} // Explicitly define for kNoLocation init

BoardLocation::BoardLocation(int8_t row, int8_t col) {
  loc_ = (row < 0 || row >= 14 || col < 0 || col >= 14)
    ? 196 : (14 * row + col);
}

BoardLocation BoardLocation::Relative(int8_t delta_row, int8_t delta_col) const {
  if (Missing()) return BoardLocation::kNoLocation; // Cannot make relative from kNoLocation
  return BoardLocation(GetRow() + delta_row, GetCol() + delta_col);
}

std::string BoardLocation::PrettyStr() const {
  if (Missing()) return "NL"; // Or some other indicator for NoLocation
  std::string s;
  s += (char)('a' + GetCol());
  s += std::to_string(14 - GetRow());
  return s;
}

// Piece
Piece::Piece() : Piece(false, RED, NO_PIECE) {} // Explicitly define for kNoPiece init

Piece::Piece(bool present, PlayerColor color, PieceType piece_type) {
  bits_ = (((int8_t)present) << 7)
        | (((int8_t)color) << 5)
        | (((int8_t)piece_type) << 2);
}

Piece::Piece(PlayerColor color, PieceType piece_type)
  : Piece(true, color, piece_type) { }

Piece::Piece(Player player, PieceType piece_type)
  : Piece(true, player.GetColor(), piece_type) { }

Team Piece::GetTeam() const { return GetPlayer().GetTeam(); }


// --- Ostream Operator Implementations & ToStr Helpers ---
std::string ToStr(PlayerColor color) {
  switch (color) {
  case RED:    return "RED";
  case BLUE:   return "BLUE";
  case YELLOW: return "YELLOW";
  case GREEN:  return "GREEN";
  case UNINITIALIZED_PLAYER: return "UNINIT";
  default:     return "ERR_PLAYER";
  }
}

std::string ToStr(PieceType piece_type) {
  switch (piece_type) {
  case PAWN:   return "P";
  case ROOK:   return "R";
  case KNIGHT: return "N";
  case BISHOP: return "B";
  case KING:   return "K";
  case QUEEN:  return "Q";
  case NO_PIECE: return ".";
  default:     return "U";
  }
}

std::ostream& operator<<(std::ostream& os, const Player& player) {
  os << "Player(" << ToStr(player.GetColor()) << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Piece& piece) {
  if (piece.Missing()) {
    os << "NoPiece";
  } else {
    os << ToStr(piece.GetColor()) << "(" << ToStr(piece.GetPieceType()) << ")";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const BoardLocation& location) {
  if (location.Missing()) {
     os << "Loc(NL)"; // NoLocation
  } else {
     os << "Loc(" << (int)location.GetRow() << "," << (int)location.GetCol() << ")";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const PlacedPiece& placed_piece) {
  os << placed_piece.GetPiece() << "@" << placed_piece.GetLocation();
  return os;
}

// --- Helper Function Implementations ---
Team GetTeam(PlayerColor color) {
  return (color == RED || color == YELLOW) ? RED_YELLOW : BLUE_GREEN;
}

Player GetNextPlayer(const Player& player) {
  switch (player.GetColor()) {
  case RED:    return kBluePlayer;
  case BLUE:   return kYellowPlayer;
  case YELLOW: return kGreenPlayer;
  case GREEN:
  default:     return kRedPlayer; // Default or error case could be handled
  }
}

Player GetPreviousPlayer(const Player& player) {
  switch (player.GetColor()) {
  case RED:    return kGreenPlayer;
  case BLUE:   return kRedPlayer;
  case YELLOW: return kBluePlayer;
  case GREEN:
  default:     return kYellowPlayer; // Default or error case
  }
}

Player GetPartner(const Player& player) {
  switch (player.GetColor()) {
  case RED:    return kYellowPlayer;
  case BLUE:   return kGreenPlayer;
  case YELLOW: return kRedPlayer;
  case GREEN:
  default:     return kBluePlayer; // Default or error case
  }
}

Team OtherTeam(Team team) {
  return team == RED_YELLOW ? BLUE_GREEN : RED_YELLOW;
}

}  // namespace chess