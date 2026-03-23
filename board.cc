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

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#include "board.h"

namespace chess {

constexpr int kMobilityMultiplier = 5;
Piece Piece::kNoPiece = Piece();
BoardLocation BoardLocation::kNoLocation = BoardLocation();
CastlingRights CastlingRights::kMissingRights = CastlingRights();

const BoardLocation kRedInitialRookLocationKingside(13, 10);
const BoardLocation kRedInitialRookLocationQueenside(13, 3);
const BoardLocation kBlueInitialRookLocationKingside(10, 0);
const BoardLocation kBlueInitialRookLocationQueenside(3, 0);
const BoardLocation kYellowInitialRookLocationKingside(0, 3);
const BoardLocation kYellowInitialRookLocationQueenside(0, 10);
const BoardLocation kGreenInitialRookLocationKingside(3, 13);
const BoardLocation kGreenInitialRookLocationQueenside(10, 13);

const Player kRedPlayer = Player(RED);
const Player kBluePlayer = Player(BLUE);
const Player kYellowPlayer = Player(YELLOW);
const Player kGreenPlayer = Player(GREEN);

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

namespace {

int64_t rand64() {
  int32_t t0 = rand();
  int32_t t1 = rand();
  return (((int64_t)t0) << 32) + (int64_t)t1;
}


void AddPawnMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const BoardLocation& to,
    const PlayerColor color,
    const Piece capture = Piece::kNoPiece,
    const BoardLocation en_passant_location = BoardLocation::kNoLocation,
    const Piece en_passant_capture = Piece::kNoPiece) {
  bool is_promotion = false;

  constexpr int kRedPromotionRow = 3;
  constexpr int kYellowPromotionRow = 10;
  constexpr int kBluePromotionCol = 10;
  constexpr int kGreenPromotionCol = 3;

  switch (color) {
  case RED:
    is_promotion = to.GetRow() == kRedPromotionRow;
    break;
  case BLUE:
    is_promotion = to.GetCol() == kBluePromotionCol;
    break;
  case YELLOW:
    is_promotion = to.GetRow() == kYellowPromotionRow;
    break;
  case GREEN:
    is_promotion = to.GetCol() == kGreenPromotionCol;
    break;
  default:
    assert(false);
    break;
  }

  if (is_promotion) {
    moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, KNIGHT);
    moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, BISHOP);
    moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, ROOK);
    moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, QUEEN);
  } else {
    moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, NO_PIECE);
  }
}

}  // namespace

void Board::GetPawnMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {
  PlayerColor color = piece.GetColor();
  Team team = piece.GetTeam();

  // Move forward
  int delta_rows = 0;
  int delta_cols = 0;
  bool not_moved = false;
  switch (color) {
  case RED:
    delta_rows = -1;
    not_moved = from.GetRow() == 12;
    break;
  case BLUE:
    delta_cols = 1;
    not_moved = from.GetCol() == 1;
    break;
  case YELLOW:
    delta_rows = 1;
    not_moved = from.GetRow() == 1;
    break;
  case GREEN:
    delta_cols = -1;
    not_moved = from.GetCol() == 12;
    break;
  default:
    assert(false);
    break;
  }

  BoardLocation to = from.Relative(delta_rows, delta_cols);
  if (IsLegalLocation(to)) {
    Piece other_piece = GetPiece(to);
    if (other_piece.Missing()) {
      // Advance once square
      AddPawnMoves2(moves, from, to, piece.GetColor());
      // Initial move (advance 2 squares)
      if (not_moved) {
        to = from.Relative(delta_rows * 2, delta_cols * 2);
        other_piece = GetPiece(to);
        if (other_piece.Missing()) {
          AddPawnMoves2(moves, from, to, piece.GetColor());
        }
      }
    }
  }

  // Non-enpassant capture
  bool check_cols = team == RED_YELLOW;
  int capture_row, capture_col;
  for (int incr = 0; incr < 2; ++incr) {
    capture_row = from.GetRow() + delta_rows;
    capture_col = from.GetCol() + delta_cols;
    if (check_cols) {
      capture_col += incr == 0 ? -1 : 1;
    } else {
      capture_row += incr == 0 ? -1 : 1;
    }
    if (IsLegalLocation(capture_row, capture_col)) {
      auto other_piece = GetPiece(capture_row, capture_col);
      if (other_piece.Present()
          && other_piece.GetTeam() != team) {
        AddPawnMoves2(moves, from, BoardLocation(capture_row, capture_col),
            piece.GetColor(), other_piece);
      }
    }
  }

  // --- NEW EN PASSANT LOGIC ---
  for (int opp_color_idx = 0; opp_color_idx < 4; ++opp_color_idx) {
    PlayerColor opp_color = static_cast<PlayerColor>(opp_color_idx);
    if (GetTeam(opp_color) == team) {
      continue; // Can't capture own team
    }

    const BoardLocation& target_square = en_passant_target_[opp_color];
    if (target_square.Missing()) {
      continue; // No en passant opportunity from this opponent
    }

    // Check if our pawn is in a position to perform this capture
    if (PawnAttacks(from, color, target_square)) {
      // Victim pawn is one square "behind" the target square, from the perspective of the victim pawn
      BoardLocation victim_loc;
      switch (opp_color) {
        case RED:    victim_loc = target_square.Relative(-1, 0); break;
        case BLUE:   victim_loc = target_square.Relative(0, 1); break;
        case YELLOW: victim_loc = target_square.Relative(1, 0); break;
        case GREEN:  victim_loc = target_square.Relative(0, -1); break;
        default:     assert(false); break;
      }
      
      Piece victim_pawn = GetPiece(victim_loc);
      // Verify the victim pawn is still there and is the correct type and color
      if (victim_pawn.Present() && victim_pawn.GetPieceType() == PAWN && victim_pawn.GetColor() == opp_color) {
        // Check for double capture: a piece on the target square
        Piece piece_on_target = GetPiece(target_square);
        if (piece_on_target.Missing() || piece_on_target.GetTeam() != team) {
            AddPawnMoves2(moves, from, target_square, color, piece_on_target, victim_loc, victim_pawn);
        }
      }
    }
  }
}

void Board::GetKnightMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  int delta_row, delta_col;
  for (int pos_row_sign = 0; pos_row_sign < 2; ++pos_row_sign) {
    for (int abs_delta_row = 1; abs_delta_row < 3; ++abs_delta_row) {
      delta_row = pos_row_sign > 0 ? abs_delta_row : -abs_delta_row;
      for (int pos_col_sign = 0; pos_col_sign < 2; ++pos_col_sign) {
        int abs_delta_col = abs_delta_row == 1 ? 2 : 1;
        delta_col = pos_col_sign > 0 ? abs_delta_col : -abs_delta_col;
        BoardLocation to = from.Relative(delta_row, delta_col);
        if (IsLegalLocation(to)) {
          const auto capture = GetPiece(to);
          if (capture.Missing()
              || capture.GetTeam() != piece.GetTeam()) {
            moves.emplace_back(from, to, capture);
          }
        }
      }
    }
  }

}

void Board::AddMovesFromIncrMovement(
    std::vector<Move>& moves,
    const Piece& piece,
    const BoardLocation& from,
    int incr_row,
    int incr_col,
    CastlingRights initial_castling_rights,
    CastlingRights castling_rights) const {
  BoardLocation to = from.Relative(incr_row, incr_col);
  while (IsLegalLocation(to)) {
    const auto capture = GetPiece(to);
    if (capture.Missing()) {
      moves.emplace_back(from, to, Piece::kNoPiece, initial_castling_rights,
          castling_rights);
    } else {
      if (capture.GetTeam() != piece.GetTeam()) {
        moves.emplace_back(from, to, capture, initial_castling_rights,
            castling_rights);
      }
      break;
    }
    to = to.Relative(incr_row, incr_col);
  }
}

void Board::AddMovesFromIncrMovement2(
    MoveBuffer& moves,
    const Piece& piece,
    const BoardLocation& from,
    int incr_row,
    int incr_col,
    CastlingRights initial_castling_rights,
    CastlingRights castling_rights) const {
  BoardLocation to = from.Relative(incr_row, incr_col);
  while (IsLegalLocation(to)) {
    const auto capture = GetPiece(to);
    if (capture.Missing()) {
      moves.emplace_back(from, to, Piece::kNoPiece, initial_castling_rights,
          castling_rights);
    } else {
      if (capture.GetTeam() != piece.GetTeam()) {
        moves.emplace_back(from, to, capture, initial_castling_rights,
            castling_rights);
      }
      break;
    }
    to = to.Relative(incr_row, incr_col);
  }
}

void Board::GetBishopMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  for (int pos_row = 0; pos_row < 2; ++pos_row) {
    for (int pos_col = 0; pos_col < 2; ++pos_col) {
      AddMovesFromIncrMovement2(
          moves, piece, from, pos_row ? 1 : -1, pos_col ? 1 : -1);
    }
  }
}

void Board::GetRookMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  // Update castling rights
  CastlingRights initial_castling_rights;
  CastlingRights castling_rights;
  std::optional<CastlingType> castling_type = GetRookLocationType(
      piece.GetPlayer(), from);
  if (castling_type.has_value()) {
    const auto& curr_rights = castling_rights_[piece.GetColor()];
    if (curr_rights.Kingside() || curr_rights.Queenside()) {
      if (castling_type == KINGSIDE) {
        if (curr_rights.Kingside()) {
          initial_castling_rights = curr_rights;
          castling_rights = CastlingRights(false, curr_rights.Queenside());
        }
      } else {
        if (curr_rights.Queenside()) {
          initial_castling_rights = curr_rights;
          castling_rights = CastlingRights(curr_rights.Kingside(), false);
        }
      }
    }
  }

  for (int do_pos_incr = 0; do_pos_incr < 2; ++do_pos_incr) {
    int incr = do_pos_incr > 0 ? 1 : -1;
    for (int do_incr_row = 0; do_incr_row < 2; ++do_incr_row) {
      int incr_row = do_incr_row > 0 ? incr : 0;
      int incr_col = do_incr_row > 0 ? 0 : incr;
      AddMovesFromIncrMovement2(moves, piece, from, incr_row, incr_col,
          initial_castling_rights, castling_rights);
    }
  }
}

void Board::GetQueenMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {
  GetBishopMoves2(moves, from, piece);
  GetRookMoves2(moves, from, piece);
}

void Board::GetKingMoves2(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  const CastlingRights& initial_castling_rights = castling_rights_[piece.GetColor()];
  CastlingRights castling_rights(false, false);

  // --- Standard King Moves (same as before) ---
  for (int delta_row = -1; delta_row < 2; ++delta_row) {
    for (int delta_col = -1; delta_col < 2; ++delta_col) {
      if (delta_row == 0 && delta_col == 0) {
        continue;
      }
      BoardLocation to = from.Relative(delta_row, delta_col);
      if (IsLegalLocation(to)) {
        const auto capture = GetPiece(to);
        if (capture.Missing()
            || capture.GetTeam() != piece.GetTeam()) {
          moves.emplace_back(from, to, capture, initial_castling_rights,
              castling_rights);
        }
      }
    }
  }

  // --- Castling ---
  // Delegate to the appropriate helper function based on the setup type.
  if (setup_type_ == MODERN) {
    AddModernCastlingMoves(moves, from, piece);
  } else {
    AddClassicCastlingMoves(moves, from, piece);
  }
}

void Board::AddModernCastlingMoves(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  const CastlingRights& initial_castling_rights = castling_rights_[piece.GetColor()];
  CastlingRights castling_rights(false, false);
  Team other_team = OtherTeam(piece.GetTeam());
  
  for (int is_kingside = 0; is_kingside < 2; ++is_kingside) {
    bool allowed = is_kingside ? initial_castling_rights.Kingside() :
      initial_castling_rights.Queenside();
    if (allowed) {
      std::vector<BoardLocation> squares_between;
      BoardLocation rook_location;

      switch (piece.GetColor()) {
      case RED:
        if (is_kingside) {
          squares_between = { from.Relative(0, 1), from.Relative(0, 2) };
          rook_location = from.Relative(0, 3);
        } else {
          squares_between = { from.Relative(0, -1), from.Relative(0, -2), from.Relative(0, -3) };
          rook_location = from.Relative(0, -4);
        }
        break;
      case BLUE:
        if (is_kingside) {
          squares_between = { from.Relative(1, 0), from.Relative(2, 0) };
          rook_location = from.Relative(3, 0);
        } else {
          squares_between = { from.Relative(-1, 0), from.Relative(-2, 0), from.Relative(-3, 0) };
          rook_location = from.Relative(-4, 0);
        }
        break;
      case YELLOW:
        if (is_kingside) {
          squares_between = { from.Relative(0, -1), from.Relative(0, -2) };
          rook_location = from.Relative(0, -3);
        } else {
          squares_between = { from.Relative(0, 1), from.Relative(0, 2), from.Relative(0, 3) };
          rook_location = from.Relative(0, 4);
        }
        break;
      case GREEN:
        if (is_kingside) {
          squares_between = { from.Relative(-1, 0), from.Relative(-2, 0) };
          rook_location = from.Relative(-3, 0);
        } else {
          squares_between = { from.Relative(1, 0), from.Relative(2, 0), from.Relative(3, 0) };
          rook_location = from.Relative(4, 0);
        }
        break;
      default:
        assert(false);
        break;
      }

      const auto rook = GetPiece(rook_location);
      if (rook.Missing() || rook.GetPieceType() != ROOK || rook.GetTeam() != piece.GetTeam()) {
        continue;
      }

      bool piece_between = false;
      for (const auto& loc : squares_between) {
        if (GetPiece(loc).Present()) {
          piece_between = true;
          break;
        }
      }

      if (!piece_between) {
        if (!IsAttackedByTeam(other_team, squares_between[0]) && !IsAttackedByTeam(other_team, from)) {
          SimpleMove rook_move(rook_location, squares_between[0]);
          moves.emplace_back(from, squares_between[1], rook_move,
              initial_castling_rights, castling_rights);
        }
      }
    }
  }
}

void Board::AddClassicCastlingMoves(
    MoveBuffer& moves,
    const BoardLocation& from,
    const Piece& piece) const {

  const CastlingRights& initial_castling_rights = castling_rights_[piece.GetColor()];
  CastlingRights castling_rights(false, false);
  Team other_team = OtherTeam(piece.GetTeam());

  for (int is_kingside = 0; is_kingside < 2; ++is_kingside) {
    bool allowed = is_kingside ? initial_castling_rights.Kingside() :
      initial_castling_rights.Queenside();
    if (allowed) {
      std::vector<BoardLocation> squares_between;
      BoardLocation rook_location;

      switch (piece.GetColor()) {
      case RED:   // Same as modern
        if (is_kingside) {
          squares_between = { from.Relative(0, 1), from.Relative(0, 2) };
          rook_location = from.Relative(0, 3);
        } else {
          squares_between = { from.Relative(0, -1), from.Relative(0, -2), from.Relative(0, -3) };
          rook_location = from.Relative(0, -4);
        }
        break;
      case BLUE:  // Logic is inverted compared to modern
        if (is_kingside) { // This is the SHORT castle (towards row 3)
          squares_between = { from.Relative(-1, 0), from.Relative(-2, 0) };
          rook_location = from.Relative(-3, 0);
        } else { // This is the LONG castle (towards row 10)
          squares_between = { from.Relative(1, 0), from.Relative(2, 0), from.Relative(3, 0) };
          rook_location = from.Relative(4, 0);
        }
        break;
      case YELLOW: // Same as modern
        if (is_kingside) {
          squares_between = { from.Relative(0, -1), from.Relative(0, -2) };
          rook_location = from.Relative(0, -3);
        } else {
          squares_between = { from.Relative(0, 1), from.Relative(0, 2), from.Relative(0, 3) };
          rook_location = from.Relative(0, 4);
        }
        break;
      case GREEN: // Logic is inverted compared to modern
        if (is_kingside) { // This is the SHORT castle (towards row 10)
          squares_between = { from.Relative(1, 0), from.Relative(2, 0) };
          rook_location = from.Relative(3, 0);
        } else { // This is the LONG castle (towards row 3)
          squares_between = { from.Relative(-1, 0), from.Relative(-2, 0), from.Relative(-3, 0) };
          rook_location = from.Relative(-4, 0);
        }
        break;
      default:
        assert(false);
        break;
      }

      const auto rook = GetPiece(rook_location);
      if (rook.Missing() || rook.GetPieceType() != ROOK || rook.GetTeam() != piece.GetTeam()) {
        continue;
      }

      bool piece_between = false;
      for (const auto& loc : squares_between) {
        if (GetPiece(loc).Present()) {
          piece_between = true;
          break;
        }
      }

      if (!piece_between) {
        if (!IsAttackedByTeam(other_team, squares_between[0]) && !IsAttackedByTeam(other_team, from)) {
          SimpleMove rook_move(rook_location, squares_between[0]);
          moves.emplace_back(from, squares_between[1], rook_move,
              initial_castling_rights, castling_rights);
        }
      }
    }
  }
}

bool Board::RookAttacks(
    const BoardLocation& rook_loc,
    const BoardLocation& other_loc) const {
  if (rook_loc.GetRow() == other_loc.GetRow()) {
    bool piece_between = false;
    for (int col = std::min(rook_loc.GetCol(), other_loc.GetCol()) + 1;
         col < std::max(rook_loc.GetCol(), other_loc.GetCol());
         ++col) {
      if (GetPiece(rook_loc.GetRow(), col).Present()) {
        piece_between = true;
        break;
      }
    }
    if (!piece_between) {
      return true;
    }
  }
  if (rook_loc.GetCol() == other_loc.GetCol()) {
    bool piece_between = false;
    for (int row = std::min(rook_loc.GetRow(), other_loc.GetRow()) + 1;
         row < std::max(rook_loc.GetRow(), other_loc.GetRow());
         ++row) {
      if (GetPiece(row, rook_loc.GetCol()).Present()) {
        piece_between = true;
        break;
      }
    }
    if (!piece_between) {
      return true;
    }
  }
  return false;
}

bool Board::BishopAttacks(
    const BoardLocation& bishop_loc,
    const BoardLocation& other_loc) const {
  int delta_row = bishop_loc.GetRow() - other_loc.GetRow();
  int delta_col = bishop_loc.GetCol() - other_loc.GetCol();
  if (std::abs(delta_row) == std::abs(delta_col)) {
    int row;
    int col;
    int col_incr;
    int row_max;
    if (bishop_loc.GetRow() < other_loc.GetRow()) {
      row = bishop_loc.GetRow();
      col = bishop_loc.GetCol();
      row_max = other_loc.GetRow();
      col_incr = bishop_loc.GetCol() < other_loc.GetCol() ? 1 : -1;
    } else {
      row = other_loc.GetRow();
      col = other_loc.GetCol();
      row_max = bishop_loc.GetRow();
      col_incr = other_loc.GetCol() < bishop_loc.GetCol() ? 1 : -1;
    }
    row++;
    col += col_incr;
    bool piece_between = false;
    while (row < row_max) {
      if (GetPiece(row, col).Present()) {
        piece_between = true;
        break;
      }

      ++row;
      col += col_incr;
    }
    return !piece_between;
  }
  return false;
}

bool Board::QueenAttacks(
    const BoardLocation& queen_loc,
    const BoardLocation& other_loc) const {
  return RookAttacks(queen_loc, other_loc)
         || BishopAttacks(queen_loc, other_loc);
}

bool Board::KingAttacks(
    const BoardLocation& king_loc,
    const BoardLocation& other_loc) const {
  if ((std::abs(king_loc.GetRow() - other_loc.GetRow())
        + std::abs(king_loc.GetCol() - other_loc.GetCol())) < 2) {
    return true;
  }
  return false;
}

bool Board::KnightAttacks(
    const BoardLocation& knight_loc,
    const BoardLocation& other_loc) const {
  int abs_row_diff = std::abs(knight_loc.GetRow() - other_loc.GetRow());
  int abs_col_diff = std::abs(knight_loc.GetCol() - other_loc.GetCol());
  return (abs_row_diff == 1 && abs_col_diff == 2)
    || (abs_row_diff == 2 && abs_col_diff == 1);
}

bool Board::PawnAttacks(
    const BoardLocation& pawn_loc,
    PlayerColor pawn_color,
    const BoardLocation& other_loc) const {
  int row_diff = other_loc.GetRow() - pawn_loc.GetRow();
  int col_diff = other_loc.GetCol() - pawn_loc.GetCol();
  switch (pawn_color) {
  case RED:
    return row_diff == -1 && (std::abs(col_diff) == 1);
  case BLUE:
    return col_diff == 1 && (std::abs(row_diff) == 1);
  case YELLOW:
    return row_diff == 1 && (std::abs(col_diff) == 1);
  case GREEN:
    return col_diff == -1 && (std::abs(row_diff) == 1);
  default:
    assert(false);
    return false;
  }
}

size_t Board::GetAttackers2(
    PlacedPiece* buffer, size_t limit,
    Team team, const BoardLocation& location) const {
  assert(limit > 0);
  size_t pos = 0;

#define ADD_ATTACKER(row, col, piece) \
  buffer[pos++] = PlacedPiece(BoardLocation(row, col), piece); \
  if (pos == limit) { \
    return limit; \
  }

  int loc_row = location.GetRow();
  int loc_col = location.GetCol();
  bool no_team = team == NO_TEAM;

  // Rooks & queens
  for (int do_incr_row = 0; do_incr_row < 2; ++do_incr_row) {
    for (int pos_incr = 0; pos_incr < 2; ++pos_incr) {
      int row_incr = do_incr_row ? (pos_incr ? 1 : -1) : 0;
      int col_incr = do_incr_row ? 0 : (pos_incr ? 1 : -1);
      int row = loc_row + row_incr;
      int col = loc_col + col_incr;
      while (row >= 0 && row < 14 && col >= 0 && col < 14) {
        const auto piece = GetPiece(row, col);
        if (piece.Present()) {
          if ((piece.GetTeam() == team || no_team)
              && (piece.GetPieceType() == ROOK
                  || piece.GetPieceType() == QUEEN)) {
            ADD_ATTACKER(row, col, piece);
          }
          break;
        }
        row += row_incr;
        col += col_incr;
      }
    }
  }

  // Bishops & queens
  for (int pos_row = 0; pos_row < 2; ++pos_row) {
    int row_incr = pos_row ? 1 : -1;
    for (int pos_col = 0; pos_col < 2; ++pos_col) {
      int col_incr = pos_col ? 1 : -1;
      int row = loc_row + row_incr;
      int col = loc_col + col_incr;
      while (IsLegalLocation(row, col)) {
        const auto piece = GetPiece(row, col);
        if (piece.Present()) {
          if ((piece.GetTeam() == team || no_team)
              && (piece.GetPieceType() == BISHOP
                  || piece.GetPieceType() == QUEEN)) {
            ADD_ATTACKER(row, col, piece);
          }
          break;
        }
        row += row_incr;
        col += col_incr;
      }
    }
  }

#ifdef __AVX2__
  // Knights (SIMD version)
  {
    alignas(32) static const int32_t knight_d_row[8] = {1, 1, -1, -1, 2, 2, -2, -2};
    alignas(32) static const int32_t knight_d_col[8] = {2, -2, 2, -2, 1, -1, 1, -1};

    __m256i v_loc_row = _mm256_set1_epi32(loc_row);
    __m256i v_loc_col = _mm256_set1_epi32(loc_col);

    __m256i v_d_row = _mm256_load_si256((__m256i const*)knight_d_row);
    __m256i v_d_col = _mm256_load_si256((__m256i const*)knight_d_col);

    __m256i v_target_rows = _mm256_add_epi32(v_loc_row, v_d_row);
    __m256i v_target_cols = _mm256_add_epi32(v_loc_col, v_d_col);

    // Legality check
    __m256i v_3 = _mm256_set1_epi32(3);
    __m256i v_10 = _mm256_set1_epi32(10);
    
    __m256i rows_ok = _mm256_and_si256(_mm256_cmpgt_epi32(v_target_rows, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(14), v_target_rows));
    __m256i cols_ok = _mm256_and_si256(_mm256_cmpgt_epi32(v_target_cols, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(14), v_target_cols));
    __m256i bounds_ok = _mm256_and_si256(rows_ok, cols_ok);

    __m256i row_lt_3 = _mm256_cmpgt_epi32(v_3, v_target_rows);
    __m256i row_gt_10 = _mm256_cmpgt_epi32(v_target_rows, v_10);
    __m256i col_lt_3 = _mm256_cmpgt_epi32(v_3, v_target_cols);
    __m256i col_gt_10 = _mm256_cmpgt_epi32(v_target_cols, v_10);
    __m256i corner1 = _mm256_and_si256(row_lt_3, _mm256_or_si256(col_lt_3, col_gt_10));
    __m256i corner2 = _mm256_and_si256(row_gt_10, _mm256_or_si256(col_lt_3, col_gt_10));
    __m256i is_corner = _mm256_or_si256(corner1, corner2);
    
    __m256i legal_mask_vec = _mm256_andnot_si256(is_corner, bounds_ok);

    int legal_mask = _mm256_movemask_epi8(legal_mask_vec);
    if (legal_mask != 0) {
      alignas(32) int32_t target_rows[8];
      alignas(32) int32_t target_cols[8];
      _mm256_store_si256((__m256i*)target_rows, v_target_rows);
      _mm256_store_si256((__m256i*)target_cols, v_target_cols);

      for (int i = 0; i < 8; ++i) {
        if ((legal_mask >> (i * 4)) & 1) {
          const auto piece = GetPiece(target_rows[i], target_cols[i]);
          if (piece.Present()
              && (piece.GetTeam() == team || no_team)
              && piece.GetPieceType() == KNIGHT) {
            ADD_ATTACKER(target_rows[i], target_cols[i], piece);
          }
        }
      }
    }
  }
#else
  // Knights (Original scalar version)
  for (int row_less = 0; row_less < 2; ++row_less) {
    for (int pos_row = 0; pos_row < 2; ++pos_row) {
      int row = loc_row + (row_less ? (pos_row ? 1 : -1) : (pos_row ? 2: -2));
      for (int pos_col = 0; pos_col < 2; ++pos_col) {
        int col = loc_col + (row_less ? (pos_col ? 2: -2): (pos_col ? 1 : -1));
        if (IsLegalLocation(row, col)) {
          const auto piece = GetPiece(row, col);
          if (piece.Present()
              && (piece.GetTeam() == team || no_team)
              && piece.GetPieceType() == KNIGHT) {
            ADD_ATTACKER(row, col, piece);
          }
        }
      }
    }
  }
#endif

  // Pawns
  for (int pos_row = 0; pos_row < 2; ++pos_row) {
    int row = pos_row ? loc_row + 1 : loc_row - 1;
    if (row >= 0 && row < 14) {
      for (int pos_col = 0; pos_col < 2; ++pos_col) {
        int col = pos_col ? loc_col + 1 : loc_col - 1;
        if (col >= 0 && col < 14) {
          const auto piece = GetPiece(row, col);
          if (piece.Present()
              && (piece.GetTeam() == team || no_team)
              && piece.GetPieceType() == PAWN) {
            bool attacks = false;
            switch (piece.GetColor()) {
            case RED:
              if (pos_row) {
                attacks = true;
              }
              break;
            case BLUE:
              if (!pos_col) {
                attacks = true;
              }
              break;
            case YELLOW:
              if (!pos_row) {
                attacks = true;
              }
              break;
            case GREEN:
              if (pos_col) {
                attacks = true;
              }
              break;
            default:
              assert(false);
              break;
            }

            if (attacks) {
              ADD_ATTACKER(row, col, piece);
            }

          }
        }
      }
    }
  }

#ifdef __AVX2__
  // Kings (SIMD version)
  {
      alignas(32) static const int32_t king_d_row[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
      alignas(32) static const int32_t king_d_col[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

      __m256i v_loc_row = _mm256_set1_epi32(loc_row);
      __m256i v_loc_col = _mm256_set1_epi32(loc_col);

      __m256i v_d_row = _mm256_load_si256((__m256i const*)king_d_row);
      __m256i v_d_col = _mm256_load_si256((__m256i const*)king_d_col);

      __m256i v_target_rows = _mm256_add_epi32(v_loc_row, v_d_row);
      __m256i v_target_cols = _mm256_add_epi32(v_loc_col, v_d_col);

      // Legality check
      __m256i v_3 = _mm256_set1_epi32(3);
      __m256i v_10 = _mm256_set1_epi32(10);
      
      __m256i rows_ok = _mm256_and_si256(_mm256_cmpgt_epi32(v_target_rows, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(14), v_target_rows));
      __m256i cols_ok = _mm256_and_si256(_mm256_cmpgt_epi32(v_target_cols, _mm256_set1_epi32(-1)), _mm256_cmpgt_epi32(_mm256_set1_epi32(14), v_target_cols));
      __m256i bounds_ok = _mm256_and_si256(rows_ok, cols_ok);

      __m256i row_lt_3 = _mm256_cmpgt_epi32(v_3, v_target_rows);
      __m256i row_gt_10 = _mm256_cmpgt_epi32(v_target_rows, v_10);
      __m256i col_lt_3 = _mm256_cmpgt_epi32(v_3, v_target_cols);
      __m256i col_gt_10 = _mm256_cmpgt_epi32(v_target_cols, v_10);
      __m256i corner1 = _mm256_and_si256(row_lt_3, _mm256_or_si256(col_lt_3, col_gt_10));
      __m256i corner2 = _mm256_and_si256(row_gt_10, _mm256_or_si256(col_lt_3, col_gt_10));
      __m256i is_corner = _mm256_or_si256(corner1, corner2);
      
      __m256i legal_mask_vec = _mm256_andnot_si256(is_corner, bounds_ok);

      int legal_mask = _mm256_movemask_epi8(legal_mask_vec);
      if (legal_mask != 0) {
          alignas(32) int32_t target_rows[8];
          alignas(32) int32_t target_cols[8];
          _mm256_store_si256((__m256i*)target_rows, v_target_rows);
          _mm256_store_si256((__m256i*)target_cols, v_target_cols);

          for (int i = 0; i < 8; ++i) {
              if ((legal_mask >> (i * 4)) & 1) {
                  const auto piece = GetPiece(target_rows[i], target_cols[i]);
                  if (piece.Present()
                      && (piece.GetTeam() == team || no_team)
                      && piece.GetPieceType() == KING) {
                      ADD_ATTACKER(target_rows[i], target_cols[i], piece);
                  }
              }
          }
      }
  }
#else
  // Kings (Original scalar version)
  for (int delta_row = -1; delta_row < 2; ++delta_row) {
    int row = loc_row + delta_row;
    for (int delta_col = -1; delta_col < 2; ++delta_col) {
      if (delta_row == 0 && delta_col == 0) {
        continue;
      }
      int col = loc_col + delta_col;
      if (IsLegalLocation(row, col)) {
        const auto piece = GetPiece(row, col);
        if (piece.Present()
            && (piece.GetTeam() == team || no_team)
            && piece.GetPieceType() == KING) {
          ADD_ATTACKER(row, col, piece);
        }
      }
    }
  }
#endif

#undef ADD_ATTACKER

  return pos;
}

bool Board::IsAttackedByTeam(Team team, const BoardLocation& location) const {
  PlacedPiece attackers[1];
  size_t pos = GetAttackers2(attackers, 1, team, location);
  return pos > 0;

  // auto attackers = GetAttackers(team, location, /*return_early=*/true);
  // return attackers.size() > 0;
}

bool Board::IsOnPathBetween(
    const BoardLocation& from,
    const BoardLocation& to,
    const BoardLocation& between) const {
  int delta_row = from.GetRow() - to.GetRow();
  int delta_col = from.GetCol() - to.GetCol();
  int delta_row_between = from.GetRow() - between.GetRow();
  int delta_col_between = from.GetCol() - between.GetCol();
  return delta_row * delta_col_between == delta_col * delta_row_between;
}

bool Board::DiscoversCheck(
    const BoardLocation& king_location,
    const BoardLocation& move_from,
    const BoardLocation& move_to,
    Team attacking_team) const {
  int delta_row = move_from.GetRow() - king_location.GetRow();
  int delta_col = move_from.GetCol() - king_location.GetCol();
  if (std::abs(delta_row) != std::abs(delta_col)
      && delta_row != 0
      && delta_col != 0) {
    return false;
  }

  int incr_col = delta_col == 0 ? 0 : delta_col > 0 ? 1 : -1;
  int incr_row = delta_row == 0 ? 0 : delta_row > 0 ? 1 : -1;
  int row = king_location.GetRow() + incr_row;
  int col = king_location.GetCol() + incr_col;
  while (IsLegalLocation(row, col)) {
    if (row != move_from.GetRow() || col != move_from.GetCol()) {
      if (row == move_to.GetRow() && col == move_to.GetCol()) {
        return false;
      }
      const auto piece = GetPiece(row, col);
      if (piece.Present()) {
        if (piece.GetTeam() == attacking_team) {
          if (delta_row == 0 || delta_col == 0) {
            if (piece.GetPieceType() == QUEEN
                || piece.GetPieceType() == ROOK) {
              return true;
            }
          } else {
            if (piece.GetPieceType() == QUEEN
                || piece.GetPieceType() == BISHOP) {
              return true;
            }
          }
        }
        break;
      }
    }

    row += incr_row;
    col += incr_col;
  }
  return false;
}

size_t Board::GetPseudoLegalMoves2(Move* buffer, size_t limit) {
  MoveBuffer move_buffer;
  move_buffer.buffer = buffer;
  move_buffer.limit = limit;

  BoardLocation king_location = GetKingLocation(turn_.GetColor());
  if (!king_location.Present()) {
    return 0;
  }

  for (const auto& placed_piece : piece_list_[turn_.GetColor()]) {
    const auto& location = placed_piece.GetLocation();
    const auto& piece = placed_piece.GetPiece();
    switch (piece.GetPieceType()) {
      case PAWN:
        GetPawnMoves2(move_buffer, location, piece);
        break;
      case KNIGHT:
        GetKnightMoves2(move_buffer, location, piece);
        break;
      case BISHOP:
        GetBishopMoves2(move_buffer, location, piece);
        break;
      case ROOK:
        GetRookMoves2(move_buffer, location, piece);
        break;
      case QUEEN:
        GetQueenMoves2(move_buffer, location, piece);
        break;
      case KING:
        GetKingMoves2(move_buffer, location, piece);
        king_location = location;
        break;
      default:
       assert(false);
    }
  }

  return move_buffer.pos;
}

GameResult Board::GetGameResult() {
  if (!GetKingLocation(turn_.GetColor()).Present()) {
    // other team won
    return turn_.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
  }
  Player player = turn_;

  size_t num_moves = GetPseudoLegalMoves2(move_buffer_2_, move_buffer_size_);
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
  // No legal moves
  PlayerColor color = player.GetColor();
  if (color == RED || color == YELLOW) {
    return WIN_BG;
  }
  return WIN_RY;
}

bool Board::IsKingInCheck(const Player& player) const {
  const auto king_location = GetKingLocation(player.GetColor());

  if (!king_location.Present()) {
    return false;
  }

  return IsAttackedByTeam(OtherTeam(player.GetTeam()), king_location);
}

bool Board::IsKingInCheck(Team team) const {
  if (team == RED_YELLOW) {
    return IsKingInCheck(Player(RED)) || IsKingInCheck(Player(YELLOW));
  }
  return IsKingInCheck(Player(BLUE)) || IsKingInCheck(Player(GREEN));
}

GameResult Board::CheckWasLastMoveKingCapture() const {
  // King captured last move
  if (!moves_.empty()) {
    const auto& last_move = moves_.back();
    const auto capture = last_move.GetCapturePiece();
    if (capture.Present() && capture.GetPieceType() == KING) {
      return capture.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
    }
  }
  return IN_PROGRESS;
}

void Board::SetPiece(
    const BoardLocation& location,
    const Piece& piece) {
  // OPTIMIZATION: O(1) piece list update
  PlayerColor color = piece.GetColor();
  auto& placed_pieces = piece_list_[color];
  const int new_index = placed_pieces.size();
  
  placed_pieces.emplace_back(location, piece);

  location_to_piece_[location.GetIndex()] = piece;
  piece_indices_[location.GetIndex()] = new_index;
  // END OPTIMIZATION

  UpdatePieceHash(piece, location);
  // Update king location
  if (piece.GetPieceType() == KING) {
    king_locations_[piece.GetColor()] = location;
  }
  // Update piece eval
  int piece_eval = kPieceEvaluations[piece.GetPieceType()];
  if (piece.GetTeam() == RED_YELLOW) {
    piece_evaluation_ += piece_eval;
  } else {
    piece_evaluation_ -= piece_eval;
  }
  player_piece_evaluations_[piece.GetColor()] += piece_eval;
}

void Board::RemovePiece(const BoardLocation& location) {
  const auto piece = GetPiece(location);
  assert(piece.Present());

  // OPTIMIZATION: O(1) piece list removal (swap and pop)
  PlayerColor color = piece.GetColor();
  int index_to_remove = piece_indices_[location.GetIndex()];
  auto& placed_pieces = piece_list_[color];
  
  const PlacedPiece& last_piece = placed_pieces.back();
  placed_pieces[index_to_remove] = last_piece;
  piece_indices_[last_piece.GetLocation().GetIndex()] = index_to_remove;
  placed_pieces.pop_back();
  // END OPTIMIZATION

  UpdatePieceHash(piece, location);
  location_to_piece_[location.GetIndex()] = Piece();
  piece_indices_[location.GetIndex()] = -1;
  
  // Update king location
  if (piece.GetPieceType() == KING) {
    king_locations_[piece.GetColor()] = BoardLocation::kNoLocation;
  }
  // Update piece eval
  int piece_eval = kPieceEvaluations[piece.GetPieceType()];
  if (piece.GetTeam() == RED_YELLOW) {
    piece_evaluation_ -= piece_eval;
  } else {
    piece_evaluation_ += piece_eval;
  }
  player_piece_evaluations_[piece.GetColor()] -= piece_eval;
}

void Board::InitializeHash() {
  for (int color = 0; color < 4; color++) {
    for (const auto& placed_piece : piece_list_[color]) {
      UpdatePieceHash(placed_piece.GetPiece(), placed_piece.GetLocation());
    }
  }
  UpdateTurnHash(static_cast<int>(turn_.GetColor()));

  // NEW: Hash in initial castling rights and en passant state.
  for (int color = 0; color < 4; ++color) {
    const auto& rights = castling_rights_[color];
    if (rights.Kingside()) {
      UpdateCastlingHash(static_cast<PlayerColor>(color), KINGSIDE);
    }
    if (rights.Queenside()) {
      UpdateCastlingHash(static_cast<PlayerColor>(color), QUEENSIDE);
    }
    const auto& ep_target = en_passant_target_[color];
    if (ep_target.Present()) {
      UpdateEnPassantHash(ep_target);
    }
  }
}

void Board::MakeMove(const Move& move) {
  // Cases:
  // 1. Move
  // 2. Capture
  // 3. En-passant
  // 4. Promotion
  // 5. Castling (rights, rook move)

  // --- NEW HASHING LOGIC (Part 1: Clear old state) ---
  Move move_to_store = move;
  PlayerColor current_color = turn_.GetColor();
  
  // 1. Clear expiring en passant square for the current player
  BoardLocation old_en_passant_target = en_passant_target_[current_color];
  move_to_store.SetPreviousEnPassantTarget(old_en_passant_target);
  if (old_en_passant_target.Present()) {
    UpdateEnPassantHash(old_en_passant_target);
  }
  en_passant_target_[current_color] = BoardLocation::kNoLocation;
  
  // 2. Update hash for castling rights changes
  const auto initial_castling_rights = move.GetInitialCastlingRights();
  const auto final_castling_rights = move.GetCastlingRights();
  if (final_castling_rights.Present()) {
      if (initial_castling_rights.Kingside() != final_castling_rights.Kingside()) {
          UpdateCastlingHash(current_color, KINGSIDE);
      }
      if (initial_castling_rights.Queenside() != final_castling_rights.Queenside()) {
          UpdateCastlingHash(current_color, QUEENSIDE);
      }
      castling_rights_[current_color] = final_castling_rights;
  }
  // --- END HASHING LOGIC (Part 1) ---


  const auto piece = GetPiece(move.From());

  // Capture
  const auto standard_capture = GetPiece(move.To());
  if (standard_capture.Present()) {
    RemovePiece(move.To());
  }

  if (piece.Missing()) {
    std::cout
      << "move"
      << " from: " << move.From()
      << " to: " << move.To()
      << " turn: " << turn_
      << std::endl;
    std::cout << *this << std::endl;
    abort();
  }
  assert(piece.Present());

  RemovePiece(move.From());
  const auto promotion_piece_type = move.GetPromotionPieceType();
  if (promotion_piece_type != NO_PIECE) { // Promotion
    SetPiece(
        move.To(),
        Piece(turn_.GetColor(), promotion_piece_type));
  } else { // Move
    SetPiece(move.To(), piece);
  }

  // --- NEW HASHING LOGIC (Part 2: Set new state) ---
  // 3. Set new en passant square if pawn pushed two squares
  if (piece.GetPieceType() == PAWN && move.ManhattanDistance() == 2) {
    int dr = move.To().GetRow() - move.From().GetRow();
    int dc = move.To().GetCol() - move.From().GetCol();
    BoardLocation new_en_passant_target = move.From().Relative(dr / 2, dc / 2);
    en_passant_target_[current_color] = new_en_passant_target;
    UpdateEnPassantHash(new_en_passant_target);
  }
  // --- END HASHING LOGIC (Part 2) ---


  // En-passant
  const auto enpassant_location = move.GetEnpassantLocation();
  if (enpassant_location.Present()) {
    RemovePiece(enpassant_location);
  } else {
    // Castling
    const auto rook_move = move.GetRookMove();
    if (rook_move.Present()) {
      const auto rook = GetPiece(rook_move.From());
      assert(rook.Present());
      RemovePiece(rook_move.From());
      SetPiece(rook_move.To(), rook);
    }
    // Castling: rights update is now handled by the hashing logic above
  }

  int t = static_cast<int>(turn_.GetColor());
  UpdateTurnHash(t);
  UpdateTurnHash((t+1)%4);

  turn_ = GetNextPlayer(turn_);
  moves_.push_back(move_to_store); // Push the copy with history
}

void Board::UndoMove() {
  // Cases:
  // 1. Move
  // 2. Capture
  // 3. En-passant
  // 4. Promotion
  // 5. Castling (rights, rook move)

  assert(!moves_.empty());
  const Move& move = moves_.back();
  Player turn_before = GetPreviousPlayer(turn_);
  PlayerColor color_before = turn_before.GetColor();

  // --- NEW HASHING LOGIC (Undo Part 1: Clear new state) ---
  // 1. Undo new en passant square creation
  const auto piece_moved_after = GetPiece(move.To());
  const auto piece_moved_before = move.GetPromotionPieceType() != NO_PIECE ? Piece(color_before, PAWN) : piece_moved_after;
  if (piece_moved_before.GetPieceType() == PAWN && move.ManhattanDistance() == 2) {
      BoardLocation new_en_passant_target = en_passant_target_[color_before];
      if (new_en_passant_target.Present()) {
          UpdateEnPassantHash(new_en_passant_target);
      }
      en_passant_target_[color_before] = BoardLocation::kNoLocation;
  }
  
  // 2. Undo castling rights changes
  const auto initial_castling_rights = move.GetInitialCastlingRights();
  const auto final_castling_rights = move.GetCastlingRights();
  if (final_castling_rights.Present()) {
      if (initial_castling_rights.Kingside() != final_castling_rights.Kingside()) {
          UpdateCastlingHash(color_before, KINGSIDE);
      }
      if (initial_castling_rights.Queenside() != final_castling_rights.Queenside()) {
          UpdateCastlingHash(color_before, QUEENSIDE);
      }
      castling_rights_[color_before] = initial_castling_rights;
  }
  // --- END HASHING LOGIC (Undo Part 1) ---


  const BoardLocation& to = move.To();
  const BoardLocation& from = move.From();

  // Move the piece back.
  const auto piece = GetPiece(to);
  if (piece.Missing()) {
    std::cout << "piece missing in UndoMove" << std::endl;
    std::cout << *this << std::endl;
    abort();
  }

  RemovePiece(to);
  const auto promotion_piece_type = move.GetPromotionPieceType();
  if (promotion_piece_type != NO_PIECE) {
    // Handle promotions
    SetPiece(from, Piece(turn_before.GetColor(), PAWN));
  } else {
    SetPiece(from, piece);
  }

  // Place back captured pieces
  const auto standard_capture = move.GetStandardCapture();
  if (standard_capture.Present()) {
    SetPiece(to, standard_capture);
  }

  // Place back en-passant pawns
  const auto enpassant_location = move.GetEnpassantLocation();
  if (enpassant_location.Present()) {
    SetPiece(enpassant_location,
             move.GetEnpassantCapture());
  } else {
    // Castling: rook move
    const auto rook_move = move.GetRookMove();
    if (rook_move.Present()) {
      RemovePiece(rook_move.To());
      SetPiece(rook_move.From(), Piece(turn_before.GetColor(), ROOK));
    }
    // Castling: rights update is handled by hashing logic above
  }

  // --- NEW HASHING LOGIC (Undo Part 2: Restore old state) ---
  // 3. Restore the previously cleared en passant square for this player
  BoardLocation old_en_passant_target = move.GetPreviousEnPassantTarget();
  if (old_en_passant_target.Present()) {
    UpdateEnPassantHash(old_en_passant_target);
  }
  en_passant_target_[color_before] = old_en_passant_target;
  // --- END HASHING LOGIC (Undo Part 2) ---

  turn_ = turn_before;
  moves_.pop_back();
  int t = static_cast<int>(turn_.GetColor());
  UpdateTurnHash(t);
  UpdateTurnHash((t+1)%4);
}

BoardLocation Board::GetKingLocation(PlayerColor color) const {
  return king_locations_[color];
}

Team Board::TeamToPlay() const {
  return GetTeam(GetTurn().GetColor());
}

int Board::PieceEvaluation() const {
  assert(player_piece_evaluations_[RED]
       + player_piece_evaluations_[YELLOW]
       - player_piece_evaluations_[BLUE]
       - player_piece_evaluations_[GREEN]
       == piece_evaluation_);
  return piece_evaluation_;
}

int Board::PieceEvaluation(PlayerColor color) const {
  return player_piece_evaluations_[color];
}

int Board::MobilityEvaluation(const Player& player) {
  Player turn = turn_;
  turn_ = player;
  int mobility = 0;
  size_t num_moves = GetPseudoLegalMoves2(move_buffer_2_, move_buffer_size_);
  int player_mobility = (int) num_moves;

  if (player.GetTeam() == RED_YELLOW) {
    mobility += player_mobility;
  } else {
    mobility -= player_mobility;
  }

  mobility *= kMobilityMultiplier;

  turn_ = turn;
  return mobility;
}

int Board::MobilityEvaluation() {
  Player turn = turn_;

  int mobility = 0;
  for (int player_color = 0; player_color < 4; ++player_color) {
    turn_ = Player(static_cast<PlayerColor>(player_color));
    size_t num_moves = GetPseudoLegalMoves2(move_buffer_2_, move_buffer_size_);
    int player_mobility = (int) num_moves;

    if (turn_.GetTeam() == RED_YELLOW) {
      mobility += player_mobility;
    } else {
      mobility -= player_mobility;
    }
  }

  mobility *= kMobilityMultiplier;

  turn_ = turn;
  return mobility;
}

Board::Board(
    Player turn,
    std::unordered_map<BoardLocation, Piece> location_to_piece,
    std::optional<std::unordered_map<Player, CastlingRights>> castling_rights,
    std::optional<std::array<BoardLocation, 4>> en_passant_targets)
  : turn_(std::move(turn))
    {

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
  // NEW: Initialize en_passant_target_
  for (int i = 0; i < 4; ++i) {
      en_passant_target_[i] = BoardLocation::kNoLocation;
  }
  if (en_passant_targets.has_value()) {
      for (int i = 0; i < 4; ++i) {
          en_passant_target_[i] = (*en_passant_targets)[i];
      }
  }
  
  move_buffer_.reserve(1000);

  // OPTIMIZATION: Initialize 1D arrays
  for (int i = 0; i < 196; ++i) {
      location_to_piece_[i] = Piece();
      piece_indices_[i] = -1;
  }

  for (int i = 0; i < 14; ++i) {
    for (int j = 0; j < 14; ++j) {
      locations_[i][j] = BoardLocation(i, j);
    }
  }

  for (int i = 0; i < 4; i++) {
    piece_list_.push_back(std::vector<PlacedPiece>());
    piece_list_[i].reserve(16);
    king_locations_[i] = BoardLocation::kNoLocation;
  }

  // --- START OF THE FIX ---

  // 1. First, populate our internal 1D array from the (non-deterministic) input map.
  // This gives us fast, O(1) access to any piece by its location.
  for (const auto& it : location_to_piece) {
    location_to_piece_[it.first.GetIndex()] = it.second;
  }

  // 2. Now, iterate over the board in a FIXED, DETERMINISTIC order (row-by-row, col-by-col)
  // to populate the piece_list_ vectors. This guarantees the initial order is always the same.
  for (int r = 0; r < 14; ++r) {
    for (int c = 0; c < 14; ++c) {
      Piece piece = location_to_piece_[r * 14 + c];
      if (piece.Present()) {
        BoardLocation location(r, c);
        PlayerColor color = piece.GetColor();

        // Add the piece to its color's list
        auto& placed_pieces = piece_list_[color];
        piece_indices_[location.GetIndex()] = placed_pieces.size();
        placed_pieces.emplace_back(location, piece);

        // Also update evaluations and king locations while we're here
        PieceType piece_type = piece.GetPieceType();
        if (piece.GetTeam() == RED_YELLOW) {
          piece_evaluation_ += kPieceEvaluations[static_cast<int>(piece_type)];
        } else {
          piece_evaluation_ -= kPieceEvaluations[static_cast<int>(piece_type)];
        }
        player_piece_evaluations_[piece.GetColor()] += kPieceEvaluations[static_cast<int>(piece_type)];
        if (piece.GetPieceType() == KING) {
          king_locations_[color] = location;
        }
      }
    }
  }

  // --- END OF THE FIX ---


  // Deduce the setup type based on the initial position of the Blue King.
  // In the classic setup, the Blue King is on a8 (row 6), in modern it's on a7 (row 7).
  const auto& blue_king_loc = king_locations_[BLUE];
  if (blue_king_loc.Present() && blue_king_loc.GetRow() == 6) {
    setup_type_ = CLASSIC;
  } else {
    setup_type_ = MODERN;
  }

  struct {
    bool operator()(const PlacedPiece& a, const PlacedPiece& b) {
      // this doesn't need to be fast.
      int piece_move_order_scores[6];
      piece_move_order_scores[PAWN] = 1;
      piece_move_order_scores[KNIGHT] = 2;
      piece_move_order_scores[BISHOP] = 3;
      piece_move_order_scores[ROOK] = 4;
      piece_move_order_scores[QUEEN] = 5;
      piece_move_order_scores[KING] = 0;

      int order_a = piece_move_order_scores[a.GetPiece().GetPieceType()];
      int order_b = piece_move_order_scores[b.GetPiece().GetPieceType()];
      
      // --- FIX PART 2: Add a tie-breaker to the sort ---
      if (order_a != order_b) {
        return order_a < order_b;
      }
      // If piece types are the same, sort by location to guarantee a stable order.
      return a.GetLocation().GetIndex() < b.GetLocation().GetIndex();
      // --- END FIX PART 2 ---
    }
  } customLess;

  for (auto& placed_pieces : piece_list_) {
    std::sort(placed_pieces.begin(), placed_pieces.end(), customLess);
    // OPTIMIZATION: After sorting, we need to update our piece_indices_ array to match.
    for(size_t i = 0; i < placed_pieces.size(); ++i) {
        piece_indices_[placed_pieces[i].GetLocation().GetIndex()] = i;
    }
  }

  // Initialize hashes for each piece at each location, and each turn
  std::srand(958829);
  for (int color = 0; color < 4; color++) {
    turn_hashes_[color] = rand64();
  }
  for (int color = 0; color < 4; color++) {
    for (int piece_type = 0; piece_type < 6; piece_type++) {
      for (int row = 0; row < 14; row++) {
        for (int col = 0; col < 14; col++) {
          piece_hashes_[color][piece_type][row][col] = rand64();
        }
      }
    }
  }

  // NEW: Initialize hashes for en passant and castling rights
  for (int row = 0; row < 14; ++row) {
    for (int col = 0; col < 14; ++col) {
        en_passant_hashes_[row][col] = rand64();
    }
  }
  for (int color = 0; color < 4; ++color) {
      castling_hashes_[color][KINGSIDE] = rand64();
      castling_hashes_[color][QUEENSIDE] = rand64();
  }

  InitializeHash();
}

inline Team GetTeam(PlayerColor color) {
  return (color == RED || color == YELLOW) ? RED_YELLOW : BLUE_GREEN;
}

Player GetNextPlayer(const Player& player) {
  switch (player.GetColor()) {
  case RED:
    return kBluePlayer;
  case BLUE:
    return kYellowPlayer;
  case YELLOW:
    return kGreenPlayer;
  case GREEN:
  default:
    return kRedPlayer;
  }
}

Player GetPartner(const Player& player) {
  switch (player.GetColor()) {
  case RED:
    return kYellowPlayer;
  case BLUE:
    return kGreenPlayer;
  case YELLOW:
    return kRedPlayer;
  case GREEN:
  default:
    return kBluePlayer;
  }
}

Player GetPreviousPlayer(const Player& player) {
  switch (player.GetColor()) {
  case RED:
    return kGreenPlayer;
  case BLUE:
    return kRedPlayer;
  case YELLOW:
    return kBluePlayer;
  case GREEN:
  default:
    return kYellowPlayer;
  }
}

std::shared_ptr<Board> Board::CreateStandardSetup(SetupType setup) {
  std::unordered_map<BoardLocation, Piece> location_to_piece;
  std::unordered_map<Player, CastlingRights> castling_rights;

  // for modern setup
  std::vector<PieceType> piece_types_modern = {
    ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK,
  };
  // for classic setup
  std::vector<PieceType> piece_types_classic = {
    ROOK, KNIGHT, BISHOP, KING, QUEEN, BISHOP, KNIGHT, ROOK,
  };

  std::vector<PlayerColor> player_colors = {RED, BLUE, YELLOW, GREEN};

  for (const PlayerColor& color : player_colors) {
    Player player(color);
    castling_rights[player] = CastlingRights(true, true);

    // logic to select the correct piece layout based on the setup
    const auto& piece_types = (setup == CLASSIC && (color == BLUE || color == GREEN))
                              ? piece_types_classic
                              : piece_types_modern;

    BoardLocation piece_location;
    int delta_row = 0;
    int delta_col = 0;
    int pawn_offset_row = 0;
    int pawn_offset_col = 0;

    switch (color) {
    case RED:
      piece_location = BoardLocation(13, 3);
      delta_col = 1;
      pawn_offset_row = -1;
      break;
    case BLUE:
      piece_location = BoardLocation(3, 0);
      delta_row = 1;
      pawn_offset_col = 1;
      break;
    case YELLOW:
      piece_location = BoardLocation(0, 10);
      delta_col = -1;
      pawn_offset_row = 1;
      break;
    case GREEN:
      piece_location = BoardLocation(10, 13);
      delta_row = -1;
      pawn_offset_col = -1;
      break;
    default:
      assert(false);
      break;
    }

    for (const PieceType piece_type : piece_types) {
      BoardLocation pawn_location = piece_location.Relative(
          pawn_offset_row, pawn_offset_col);
      location_to_piece[piece_location] = Piece(player.GetColor(), piece_type);
      location_to_piece[pawn_location] = Piece(player.GetColor(), PAWN);
      piece_location = piece_location.Relative(delta_row, delta_col);
    }
  }

  return std::make_shared<Board>(
      Player(RED), std::move(location_to_piece), std::move(castling_rights));
}

int Move::ManhattanDistance() const {
  return std::abs(from_.GetRow() - to_.GetRow())
       + std::abs(from_.GetCol() - to_.GetCol());
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

std::string ToStr(PieceType piece_type) {
  switch (piece_type) {
  case PAWN:
    return "P";
  case ROOK:
    return "R";
  case KNIGHT:
    return "N";
  case BISHOP:
    return "B";
  case KING:
    return "K";
  case QUEEN:
    return "Q";
  default:
    return "U";
  }
}

}  // namespace

std::ostream& operator<<(
    std::ostream& os, const Piece& piece) {
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
  os << "Loc(" << (int)location.GetRow() << ", " << (int)location.GetCol() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Move& move) {
  os << "Move(" << move.From() << " -> " << move.To() << ")";
  return os;
}

std::ostream& operator<<(
    std::ostream& os, const Board& board) {
  for (int i = 0; i < 14; i++) {
    for (int j = 0; j < 14; j++) {
      if (board.IsLegalLocation(BoardLocation(i, j))) {
        const auto piece = board.GetPiece(i, j);
        if (piece.Missing()) {
          os << ".";
        } else {
          os << ToStr(piece.GetPieceType());
        }
      } else {
        os << " ";
      }
    }
    os << std::endl;
  }

  os << "Turn: " << board.turn_ << std::endl;

  os << "All moves: " << std::endl;
  for (const auto& move : board.moves_) {
    os << move << std::endl;
  }
  return os;
}

const CastlingRights& Board::GetCastlingRights(const Player& player) const {
  return castling_rights_[player.GetColor()];
}

std::optional<CastlingType> Board::GetRookLocationType(
    const Player& player, const BoardLocation& location) const {
  switch (player.GetColor()) {
  case RED:
    if (location == kRedInitialRookLocationKingside) return KINGSIDE;
    if (location == kRedInitialRookLocationQueenside) return QUEENSIDE;
    break;
  case BLUE:
    if (setup_type_ == CLASSIC) {
        if (location == kBlueInitialRookLocationKingside) return QUEENSIDE;
        if (location == kBlueInitialRookLocationQueenside) return KINGSIDE;
    } else { // MODERN
        if (location == kBlueInitialRookLocationKingside) return KINGSIDE;
        if (location == kBlueInitialRookLocationQueenside) return QUEENSIDE;
    }
    break;
  case YELLOW:
    if (location == kYellowInitialRookLocationKingside) return KINGSIDE;
    if (location == kYellowInitialRookLocationQueenside) return QUEENSIDE;
    break;
  case GREEN:
    if (setup_type_ == CLASSIC) {
        if (location == kGreenInitialRookLocationKingside) return QUEENSIDE;
        if (location == kGreenInitialRookLocationQueenside) return KINGSIDE;
    } else { // MODERN
        if (location == kGreenInitialRookLocationKingside) return KINGSIDE;
        if (location == kGreenInitialRookLocationQueenside) return QUEENSIDE;
    }
    break;
  default:
    assert(false);
    break;
  }
  return std::nullopt;
}

Team OtherTeam(Team team) {
  return team == RED_YELLOW ? BLUE_GREEN : RED_YELLOW;
}

std::string BoardLocation::PrettyStr() const {
  std::string s;
  s += ('a' + GetCol());
  s += std::to_string(14 - GetRow());
  return s;
}

std::string Move::PrettyStr() const {
  std::string s = from_.PrettyStr() + "-" + to_.PrettyStr();
  if (promotion_piece_type_ != NO_PIECE) {
    s += "=" + ToStr(promotion_piece_type_);
  }
  return s;
}

bool Board::DeliversCheck(const Move& move) {
  int color = GetTurn().GetColor();
  Piece piece = GetPiece(move.From());

  bool checks = false;

  for (int add = 1; add < 4; add += 2) {
    int other = (color + add) % 4;
    auto king_loc = GetKingLocation(static_cast<PlayerColor>(other));
    if (king_loc.Present()) {
      if (king_loc == move.To()) {
        checks = true;
        break;
      }
      switch (piece.GetPieceType()) {
      case PAWN:
        checks = PawnAttacks(move.To(), piece.GetColor(), king_loc);
        break;
      case KNIGHT:
        checks = KnightAttacks(move.To(), king_loc);
        break;
      case BISHOP:
        checks = BishopAttacks(move.To(), king_loc);
        break;
      case ROOK:
        checks = RookAttacks(move.To(), king_loc);
        break;
      case QUEEN:
        checks = QueenAttacks(move.To(), king_loc);
        break;
      default:
        break;
      }
      if (checks) {
        break;
      }
    }
  }

  return checks;
}

void Board::MakeNullMove() {
  int t = static_cast<int>(turn_.GetColor());
  UpdateTurnHash(t);
  UpdateTurnHash((t+1)%4);

  turn_ = GetNextPlayer(turn_);
}

void Board::UndoNullMove() {
  turn_ = GetPreviousPlayer(turn_);

  int t = static_cast<int>(turn_.GetColor());
  UpdateTurnHash(t);
  UpdateTurnHash((t+1)%4);
}

bool Move::DeliversCheck(Board& board) {
  if (delivers_check_ < 0) {
    delivers_check_ = board.DeliversCheck(*this);
  }
  return delivers_check_;
}

int Move::SEE(Board& board,
               const int* piece_evaluations) {
  if (see_ == kSeeNotSet) {
    see_ = StaticExchangeEvaluationCapture(piece_evaluations, board, *this);
  }
  return see_;
}

int Move::ApproxSEE(Board& board, const int* piece_evaluations) {
  const auto capture = GetCapturePiece();
  const auto piece = board.GetPiece(From());
  int captured_val = piece_evaluations[capture.GetPieceType()];
  int attacker_val = piece_evaluations[piece.GetPieceType()];
  return captured_val - attacker_val;
}

int StaticExchangeEvaluationCapture(
    const int piece_evaluations[6],
    Board& board,
    const Move& move) {

  BoardLocation target = move.To();
  int initial_capture_val = piece_evaluations[move.GetCapturePiece().GetPieceType()];

  // Make the move to set up the board for the ensuing exchange
  board.MakeMove(move);
  PlayerColor current_turn = board.GetTurn().GetColor();

  // Gather all attackers on the target square
  constexpr size_t kLimit = 16;
  PlacedPiece attackers_buffer[kLimit];
  // NO_TEAM gets attackers from all 4 players
  size_t num_attackers = board.GetAttackers2(attackers_buffer, kLimit, NO_TEAM, target);

  // Use plain C-arrays (stack allocation) instead of std::vector for performance!
  int attackers[4][kLimit];
  int attacker_counts[4] = {0, 0, 0, 0};

  // Separate attackers by specific player, not by team!
  for (size_t i = 0; i < num_attackers; ++i) {
    PlayerColor pc = attackers_buffer[i].GetPiece().GetColor();
    attackers[pc][attacker_counts[pc]++] = piece_evaluations[attackers_buffer[i].GetPiece().GetPieceType()];
  }

  // Sort each player's attackers descending so we can pop_back() the cheapest piece
  for (int i = 0; i < 4; ++i) {
    if (attacker_counts[i] > 1) {
      std::sort(attackers[i], attackers[i] + attacker_counts[i], std::greater<int>());
    }
  }

  // Track the sequence of material gains
  int gains[32]; 
  int num_gains = 0;
  gains[num_gains++] = initial_capture_val;

  const Piece& piece_on_square = board.GetPiece(target);
  int current_piece_val = piece_evaluations[piece_on_square.GetPieceType()];
  Team current_owner_team = piece_on_square.GetTeam();

  PlayerColor p = current_turn;
  int passes = 0;

  // Simulate the exchange sequence strictly enforcing 4PC turn order
  while (passes < 4 && num_gains < 32) {
    // If the player's team already owns the square, or they have no pieces left, they pass.
    if (GetTeam(p) == current_owner_team || attacker_counts[p] == 0) {
      passes++;
      p = static_cast<PlayerColor>((p + 1) % 4);
      continue;
    }

    // Player p captures!
    passes = 0; // Reset consecutive passes
    int attacker_val = attackers[p][--attacker_counts[p]]; 

    gains[num_gains++] = current_piece_val;
    
    // The attacking piece is now the one sitting on the square
    current_piece_val = attacker_val;
    current_owner_team = GetTeam(p);

    p = static_cast<PlayerColor>((p + 1) % 4);
  }

  board.UndoMove();

  // Minimax evaluation backwards through the capture sequence.
  // Because teams never capture their own pieces, the captures strictly 
  // alternate between the two teams, meaning standard 1v1 SEE math applies perfectly!
  while (--num_gains > 0) {
    gains[num_gains - 1] -= std::max(0, gains[num_gains]);
  }

  return gains[0];
}


}  // namespace chess