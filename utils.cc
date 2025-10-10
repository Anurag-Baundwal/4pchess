#include "utils.h"

#include <tuple>
#include <cctype>
#include <iostream>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "board.h"


namespace chess {

std::vector<std::string> SplitStrOnWhitespace(const std::string& x) {
  std::stringstream ss(x);
  std::string part;
  std::vector<std::string> parts;
  while (ss >> part) {
    parts.push_back(part);
  }
  return parts;
}

std::vector<std::string> SplitStr(std::string s, std::string delimiter) {
  size_t pos_start = 0;
  size_t pos_end;
  size_t delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

std::optional<int> ParseInt(const std::string& input) {
  try {
    return std::stoi(input);
  } catch(std::exception const& ex) {
    return std::nullopt;
  }
}

std::optional<std::vector<bool>> ParseCastlingAvailability(
    const std::string& fen_substr) {
  std::vector<std::string> parts = SplitStr(fen_substr, ",");
  if (parts.size() != 4) {
    return std::nullopt;
  }
  std::vector<bool> availability;
  availability.reserve(4);
  for (const auto& part : parts) {
    if (part == "0") {
      availability.push_back(false);
    } else if (part == "1") {
      availability.push_back(true);
    } else {
      return std::nullopt;
    }
  }

  return availability;
}

std::optional<BoardLocation> ParseEnpLocation(const std::string& enp) {
  size_t pos = enp.find(':');
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  std::string to = enp.substr(pos + 1);
  if (!to.empty() && to[to.size() - 1] == '\'') {
    to = to.substr(0, to.size() - 1);
  }
  if (to.size() < 2 || to.size() > 3) {
    return std::nullopt;
  }
  int col = to[0] - 'a';
  if (col < 0 || col > 13) {
    return std::nullopt;
  }
  int row = to[1] - '0';
  if (row < 0 || row > 9) {
    return std::nullopt;
  }
  if (to.size() > 2) {
    int digit = to[2] - '0';
    if (digit < 0 || digit > 9) {
      return std::nullopt;
    }
    row = 10 * row + digit;
  }
  row = 14 - row;  // transform to 0-13
  return BoardLocation(row, col);
}

std::shared_ptr<Board> ParseBoardFromFEN(const std::string& fen) {
  std::vector<std::string> parts = SplitStr(fen, "-");
  if (parts.size() < 7 || parts.size() > 8) {
    return nullptr;  // invalid format
  }

  // Not used for teams chess: eliminated players, points
  // Also, currently we don't use the halfmove clock since we don't expect the
  // 50-move rule to apply in real games.

  // 0: Player to move
  // 1: Eliminated players (unused)
  // 2-3: Castling rights
  // 4: Points (unused)
  // 5: Halfmove clock (unused)
  // 6 (optional?): En-passant
  // 7: Piece placement

  const auto& player_str = parts[0];
  const auto& castling_availability_kingside = parts[2];
  const auto& castling_availability_queenside = parts[3];
  const auto& piece_placement = parts.back();

  std::string enpassant;
  if (parts.size() == 8) {
    enpassant = parts[6];
  }

  // Parse player
  if (player_str.size() != 1) {
    return nullptr;  // invalid format
  }
  char pchar = player_str[0];
  Player player;
  switch (pchar) {
  case 'R':
    player = Player(RED);
    break;
  case 'B':
    player = Player(BLUE);
    break;
  case 'Y':
    player = Player(YELLOW);
    break;
  case 'G':
    player = Player(GREEN);
    break;
  default:
    return nullptr;  // invalid format
  }

  // Parse castling availability
  std::optional<std::vector<bool>> kingside = ParseCastlingAvailability(
      castling_availability_kingside);
  if (!kingside.has_value()) {
    return nullptr;  // invalid format
  }
  std::optional<std::vector<bool>> queenside = ParseCastlingAvailability(
      castling_availability_queenside);
  if (!queenside.has_value()) {
    return nullptr;  // invalid format
  }

  std::unordered_map<Player, CastlingRights> castling_rights;
  for (int player_color = 0; player_color < 4; player_color++) {
    Player pl(static_cast<PlayerColor>(player_color));
    castling_rights[pl] = CastlingRights((*kingside)[player_color],
                                         (*queenside)[player_color]);
  }

  // Parse enpassant
  EnpassantInitialization enp;
  if (!enpassant.empty()) {
    size_t lbrace_pos = enpassant.find('(');
    size_t rbrace_pos = enpassant.rfind(')');
    if (lbrace_pos == std::string::npos || rbrace_pos == std::string::npos) {
      // invalid enpassant string
      return nullptr;
    }
    auto parts = SplitStr(
        enpassant.substr(lbrace_pos + 1, rbrace_pos - lbrace_pos), ",");
    if (parts.size() != 4) { // invalid
      return nullptr;
    }
    for (int i = 0; i < 4; i++) {
      auto enp_location = ParseEnpLocation(parts[i]);
      if (enp_location.has_value()) {
        BoardLocation& to = *enp_location;
        int from_row = to.GetRow();
        int from_col = to.GetCol();
        switch (static_cast<PlayerColor>(i)) {
        case RED:
          from_row += 2;
          break;
        case BLUE:
          from_col -= 2;
          break;
        case YELLOW:
          from_row -= 2;
          break;
        case GREEN:
          from_col += 2;
          break;
        default:
          break;
        }
        enp.enp_moves[i] = Move(BoardLocation(from_row, from_col), to);
      }
    }
  }

  // Parse piece placement
  std::vector<std::string> rows = SplitStr(piece_placement, "/");
  if (rows.size() != 14) {
    return nullptr;  // invalid format
  }
  std::unordered_map<BoardLocation, Piece> location_to_piece;
  for (size_t row = 0; row < rows.size(); row++) {
    std::vector<std::string> cols = SplitStr(rows[row], ",");
    int col = 0;
    for (const auto& col_str : cols) {

      if (col_str.empty()) {
        return nullptr;  // invalid format
      }

      char ch = col_str[0];
      if (ch == 'r' || ch == 'b' || ch == 'y' || ch == 'g') {
        // Parse piece
        if (col_str.size() != 2) {
          return nullptr;  // invalid format
        }
        BoardLocation location(row, col);

        PlayerColor player_color;
        switch (ch) {
        case 'r':
          player_color = RED;
          break;
        case 'b':
          player_color = BLUE;
          break;
        case 'y':
          player_color = YELLOW;
          break;
        case 'g':
          player_color = GREEN;
          break;
        default:
          return nullptr;  // invalid format
        }

        PieceType piece_type;
        switch (col_str[1]) {
        case 'P':
          piece_type = PAWN;
          break;
        case 'R':
          piece_type = ROOK;
          break;
        case 'N':
          piece_type = KNIGHT;
          break;
        case 'B':
          piece_type = BISHOP;
          break;
        case 'K':
          piece_type = KING;
          break;
        case 'Q':
          piece_type = QUEEN;
          break;
        default:
          return nullptr;  // invalid format
        }

        Player player(player_color);
        Piece piece(player, piece_type);

        location_to_piece[location] = piece;

        col++;
      } else if (ch == 'x') {
        // parse empty square
        col += 1;
      } else {
        // Parse empty spaces
        std::optional<int> num_empty = ParseInt(col_str);
        if (!num_empty.has_value() || *num_empty <= 0) {
          return nullptr;  // invalid format
        }
        col += *num_empty;
      }

    }
  }

  return std::make_shared<Board>(
      std::move(player), std::move(location_to_piece),
      std::move(castling_rights), std::move(enp));
}

void SendInfoMessage(const std::string& message) {
  std::cout << "info string " << message << std::endl;
}

void SendInvalidCommandMessage(const std::string& line) {
  SendInfoMessage("invalid command: '" + line + "'");
}

namespace {

std::optional<std::tuple<size_t, BoardLocation>> ParseLocation(
    const std::string& move_str, size_t start) {
  // Skip '-' and 'x'
  if (start < move_str.size() && (move_str[start] == '-'
                                  || move_str[start] == 'x')) {
    start++;
  }
  if (move_str.size() < start + 2) {
    return std::nullopt;
  }

  // Skip piece name, if present
  char c = move_str[start];
  if (c == 'K' || c == 'Q' || c == 'N' || c == 'B' || c == 'R') {
    start++;
  }

  int col = move_str[start] - 'a';
  if (col < 0 || col >= 14) {
    return std::nullopt;
  }
  start++;
  int row = move_str[start] - '0';
  if (row < 0 || row >= 10) {
    return std::nullopt;
  }
  start++;

  if (start < move_str.size() && std::isdigit(move_str[start])) {
    int digit = move_str[start] - '0';
    row = 10 * row + digit;
    start++;
  }
  // transform 1-14 upwards to 0-13 downwards
  row = 14 - row;
  return std::make_tuple(start, BoardLocation(row, col));
}

std::optional<std::tuple<size_t, PieceType>> ParsePromotion(
    const std::string& move_str, size_t start) {
  if (start >= move_str.size()) {
    return std::make_tuple(start, NO_PIECE);
  }
  if (start < move_str.size() && move_str[start] == '=') {
    start++;
  }
  if (start >= move_str.size()) {
    return std::nullopt;
  }
  char c = move_str[start];
  switch (c) {
  case 'N':
  case 'n':
    return std::make_tuple(start + 1, KNIGHT);
  case 'B':
  case 'b':
    return std::make_tuple(start + 1, BISHOP);
  case 'R':
  case 'r':
    return std::make_tuple(start + 1, ROOK);
  case 'Q':
  case 'q':
    return std::make_tuple(start + 1, QUEEN);
  default:
    break;
  }

  // unrecognized piece type
  return std::nullopt;
}

}  // namespace

std::optional<Move> ParseMove(Board& board, const std::string& move_str_ref) {
  std::string move_str = move_str_ref;
  if (!move_str.empty() && move_str[move_str.size() - 1] == '+') {
    move_str = move_str.substr(0, move_str.size() - 1);
  }
  // Expected move string format: <letter-col><number-row>
  auto from = ParseLocation(move_str, 0);
  if (!from.has_value()) {
    return std::nullopt;
  }
  auto to = ParseLocation(move_str, std::get<0>(*from));
  if (!to.has_value()) {
    return std::nullopt;
  }
  auto promotion = ParsePromotion(move_str, std::get<0>(*to));
  if (!promotion.has_value()) {
    return std::nullopt;
  }

  BoardLocation from_loc = std::get<1>(*from);
  BoardLocation to_loc = std::get<1>(*to);
  PieceType promotion_piece_type = std::get<1>(*promotion);

  Move moves[300];
  size_t num_moves = board.GetPseudoLegalMoves2(moves, 300);

  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = moves[i];
    if (move.From() == from_loc && move.To() == to_loc
        && move.GetPromotionPieceType() == promotion_piece_type) {
      return move;
    }
  }
  return std::nullopt;
}

namespace {

// Helper to convert a piece to its FEN character representation.
std::string PieceToFENChar(const Piece& piece) {
  std::string s;
  switch (piece.GetColor()) {
    case RED:    s += 'r'; break;
    case BLUE:   s += 'b'; break;
    case YELLOW: s += 'y'; break;
    case GREEN:  s += 'g'; break;
    default:     return "";
  }
  switch (piece.GetPieceType()) {
    case PAWN:   s += 'P'; break;
    case KNIGHT: s += 'N'; break;
    case BISHOP: s += 'B'; break;
    case ROOK:   s += 'R'; break;
    case QUEEN:  s += 'Q'; break;
    case KING:   s += 'K'; break;
    default:     return "";
  }
  return s;
}

// Helper to format an en passant location for the FEN string.
std::string LocationToEnpStr(const BoardLocation& victim_loc, PlayerColor pawn_color) {
  if (victim_loc.Missing()) {
    return "";
  }
  
  // Calculate the destination square based on the color of the pawn that just moved
  BoardLocation dest_loc;
  switch (pawn_color) {
    case RED:    dest_loc = victim_loc.Relative(1, 0); break;
    case BLUE:   dest_loc = victim_loc.Relative(0, -1); break;
    case YELLOW: dest_loc = victim_loc.Relative(-1, 0); break;
    case GREEN:  dest_loc = victim_loc.Relative(0, 1); break;
    default:     return ""; // Should not happen
  }

  std::stringstream victim_ss;
  victim_ss << (char)('a' + victim_loc.GetCol()) << (14 - victim_loc.GetRow());

  std::stringstream dest_ss;
  dest_ss << (char)('a' + dest_loc.GetCol()) << (14 - dest_loc.GetRow());
  
  return dest_ss.str() + ":" + victim_ss.str();
}

} // namespace


std::string GenerateFENFromBoard(const Board& board) {
  std::stringstream fen;

  // 1. Player to move
  switch (board.GetTurn().GetColor()) {
    case RED:    fen << 'R'; break;
    case BLUE:   fen << 'B'; break;
    case YELLOW: fen << 'Y'; break;
    case GREEN:  fen << 'G'; break;
    default:     fen << '?'; break; // Should not happen
  }
  fen << '-';

  // 2. Eliminated players (not tracked, so always 0)
  fen << "0,0,0,0-";

  // 3. Castling rights (Kingside)
  for (int i = 0; i < 4; ++i) {
    fen << board.GetCastlingRights(Player(static_cast<PlayerColor>(i))).Kingside();
    if (i < 3) fen << ",";
  }
  fen << '-';

  // 4. Castling rights (Queenside)
  for (int i = 0; i < 4; ++i) {
    fen << board.GetCastlingRights(Player(static_cast<PlayerColor>(i))).Queenside();
    if (i < 3) fen << ",";
  }
  fen << '-';

  // 5. Points (not tracked, so always 0)
  fen << "0,0,0,0-";

  // 6. Halfmove clock (not tracked, so always 0)
  fen << "0-";
  
  // 7. En Passant
  const auto& enp_init = board.GetEnpassantInitialization();
  bool any_enp = false;
  for(int i = 0; i < 4; ++i) {
    if (enp_init.enp_moves[i].has_value()) {
      any_enp = true;
      break;
    }
  }
  if (any_enp) {
    fen << "{'enPassant':('";
    for (int i = 0; i < 4; ++i) {
      if (enp_init.enp_moves[i].has_value()) {
        fen << LocationToEnpStr(enp_init.enp_moves[i]->To(), static_cast<PlayerColor>(i));
      }
      fen << (i < 3 ? "','": "')}-");
    }
  }


  // 8. Piece placement
  for (int r = 0; r < 14; ++r) {
    int empty_squares = 0;
    for (int c = 0; c < 14; ++c) {
      BoardLocation loc(r, c);
      if (!board.IsLegalLocation(loc)) {
        if (empty_squares > 0) {
          fen << empty_squares << ",";
          empty_squares = 0;
        }
        fen << "x,";
        continue;
      }
      
      const Piece& piece = board.GetPiece(loc);
      if (piece.Missing()) {
        empty_squares++;
      } else {
        if (empty_squares > 0) {
          fen << empty_squares << ",";
          empty_squares = 0;
        }
        fen << PieceToFENChar(piece) << ",";
      }
    }
    if (empty_squares > 0) {
      fen << empty_squares;
    }
    // Remove trailing comma if it exists
    std::string row_str = fen.str();
    if (row_str.back() == ',') {
        fen.seekp(-1, std::ios_base::end);
    }
    
    if (r < 13) {
      fen << "/";
    }
  }

  // The stringstream might have a trailing comma on the last row, clean it up.
  std::string final_fen = fen.str();
  if (final_fen.back() == ',') {
      final_fen.pop_back();
  }

  return final_fen;
}

}  // namespace chess

