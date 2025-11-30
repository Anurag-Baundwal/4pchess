#ifndef _UTILS_H_
#define _UTILS_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "board.h"

namespace chess {

// Parses a board from a Forsyth-Edwards Notation (FEN) string.
std::shared_ptr<Board> ParseBoardFromFEN(const std::string& fen);

// Generates a FEN string from the current board state.
std::string GenerateFENFromBoard(const Board& board);

// Parses a move from a string in the format "e2e4" or "e2e4q".
std::optional<Move> ParseMove(Board& board, const std::string& move_str);

// Helper string manipulation functions
std::vector<std::string> SplitStr(std::string s, std::string delimiter);
std::vector<std::string> SplitStrOnWhitespace(const std::string& x);
std::optional<int> ParseInt(const std::string& input);

// Messaging helpers
void SendInfoMessage(const std::string& message);
void SendInvalidCommandMessage(const std::string& line);

}  // namespace chess

#endif  // _UTILS_H_