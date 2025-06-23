#include <gtest/gtest.h>
#include "gmock/gmock.h"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

// You must have a utils.h that is compatible with the bitboard version of Board.
// It needs to have a ParseBoardFromFEN and ParseMove function that works with the new API.
#include "utils.h" 
#include "FastUint256.h"
#include "board.h"


namespace chess {

using Loc = BoardLocation;
using ::testing::UnorderedElementsAre;

namespace {

std::vector<Move> GetMoves(Board& board, PieceType piece_type) {
  std::vector<Move> moves;
  Move move_buffer[300];
  size_t num_moves = board.GetPseudoLegalMoves2(move_buffer, 300);

  for (size_t i = 0; i < num_moves; ++i) {
    const auto& move = move_buffer[i];
    // A move must have a 'from' location to get the piece.
    if (!move.From().Present()) continue;
    
    const auto& piece = board.GetPiece(move.From());
    if (piece.GetPieceType() == piece_type) {
      moves.push_back(move);
    }
  }
  return moves;
}

std::optional<Move> FindMove(
    Board& board, const BoardLocation& from, const BoardLocation& to) {
  Move moves[300];
  size_t num_moves = board.GetPseudoLegalMoves2(moves, 300);
  for (size_t i = 0; i < num_moves; i++) {
    const auto& move = moves[i];
    // This is a simple check; for promotions, you'd need to check promotion type too.
    if (move.From() == from && move.To() == to) {
      return move;
    }
  }
  return std::nullopt;
}

// Helper for DeliversCheck test
Move MakeMoveForCheckTest(const Board& board, BoardLocation from, BoardLocation to) {
  return Move(from, to, board.GetPiece(to));
}


}  // namespace

TEST(BoardLocationTest, Properties) {
  BoardLocation x(0, 0);
  BoardLocation y(1, 2);
  BoardLocation z(1, 2);

  EXPECT_NE(x, y);
  EXPECT_EQ(y, z);
  EXPECT_EQ(x.GetRow(), 0);
  EXPECT_EQ(x.GetCol(), 0);
  EXPECT_EQ(y.GetRow(), 1);
  EXPECT_EQ(y.GetCol(), 2);
}

TEST(PlayerTest, Properties) {
  Player red(RED);
  Player blue(BLUE);
  Player red2(RED);

  EXPECT_EQ(red.GetColor(), RED);
  EXPECT_EQ(red.GetTeam(), RED_YELLOW);
  EXPECT_EQ(blue.GetColor(), BLUE);
  EXPECT_EQ(blue.GetTeam(), BLUE_GREEN);

  EXPECT_EQ(red, red2);
  EXPECT_NE(red, blue);
}

TEST(BoardTest, GetLegalMoves_King) {
  std::shared_ptr<Board> board;
  std::vector<Move> moves;

  // normal move -- all directions
  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,5,rK,4,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,1,rB,rN,rR,x,x,x");
  ASSERT_NE(board, nullptr);

  moves = GetMoves(*board, KING);
  EXPECT_THAT(
      moves,
      UnorderedElementsAre(
        ParseMove(*board, "h5-i4"),
        ParseMove(*board, "h5-i5"),
        ParseMove(*board, "h5-i6"),
        ParseMove(*board, "h5-h4"),
        ParseMove(*board, "h5-h6"),
        ParseMove(*board, "h5-g4"),
        ParseMove(*board, "h5-g5"),
        ParseMove(*board, "h5-g6")));

  // castling: kingside & queenside (Red player)
  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,2,yK,3,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/1,bP,10,gP,1/1,bP,10,gP,1/1,bP,10,gP,gK/bK,bP,10,gP,1/1,bP,10,gP,1/1,bP,10,gP,1/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,3,rK,2,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  moves = GetMoves(*board, KING);
  EXPECT_THAT(
      moves,
      UnorderedElementsAre(
        ParseMove(*board, "h1-i1"),
        ParseMove(*board, "h1-j1"),   // Kingside castle
        ParseMove(*board, "h1-g1"),
        ParseMove(*board, "h1-f1"))); // Queenside castle

  // castling: all colors (Blue player)
  board = ParseBoardFromFEN("B-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,2,yK,3,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/1,bP,10,gP,1/1,bP,10,gP,1/1,bP,10,gP,gK/bK,bP,10,gP,1/1,bP,10,gP,1/1,bP,10,gP,1/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,3,rK,2,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  moves = GetMoves(*board, KING);
  EXPECT_THAT(
      moves,
      UnorderedElementsAre(
        ParseMove(*board, "a7-a6"),
        ParseMove(*board, "a7-a5"),   // Kingside castle
        ParseMove(*board, "a7-a8"),
        ParseMove(*board, "a7-a9"))); // Queenside castle

  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,2,yK,3,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/1,bP,10,gP,1/1,bP,10,gP,1/1,bP,4,bR,5,gP,gK/bK,bP,10,gP,1/1,bP,10,gP,1/1,bP,10,gP,1/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,1,rP,rP,rP,rP,x,x,x/x,x,x,rR,3,rK,2,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  moves = GetMoves(*board, KING);
  EXPECT_THAT(
      moves,
      UnorderedElementsAre(
        ParseMove(*board, "h1-i1"),
        ParseMove(*board, "h1-j1"), // Kingside castle
        // pseudo legal moves into check
        ParseMove(*board, "h1-g1"),
        ParseMove(*board, "h1-g2")));

  // castling not allowed while in check
  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,2,yK,3,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/1,bP,10,gP,1/1,bP,10,gP,1/1,bP,4,bR,5,gP,gK/bK,bP,10,gP,1/1,bP,10,gP,1/1,bP,1,bB,8,gP,1/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,1,rP,rP,rP,rP,x,x,x/x,x,x,rR,3,rK,2,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  moves = GetMoves(*board, KING);
  // King is in check, so no castling moves are legal.
  EXPECT_THAT(
      moves,
      UnorderedElementsAre(
        ParseMove(*board, "h1-i1"),
        ParseMove(*board, "h1-g1"),
        ParseMove(*board, "h1-g2")));
}

TEST(HelperFunctionsTest, OtherTeam) {
  EXPECT_EQ(OtherTeam(RED_YELLOW), BLUE_GREEN);
  EXPECT_EQ(OtherTeam(BLUE_GREEN), RED_YELLOW);
}

TEST(BoardTest, KeyTest) {
  auto board = Board::CreateStandardSetup();
  int64_t h0 = board->HashKey();

  auto move1_opt = ParseMove(*board, "h2-h3");
  ASSERT_TRUE(move1_opt.has_value());
  board->MakeMove(*move1_opt);
  int64_t h1 = board->HashKey();

  auto move2_opt = ParseMove(*board, "b7-c7");
  ASSERT_TRUE(move2_opt.has_value());
  board->MakeMove(*move2_opt);
  int64_t h2 = board->HashKey();

  auto move3_opt = ParseMove(*board, "g13-g12");
  ASSERT_TRUE(move3_opt.has_value());
  board->MakeMove(*move3_opt);
  int64_t h3 = board->HashKey();

  auto move4_opt = ParseMove(*board, "m8-l8");
  ASSERT_TRUE(move4_opt.has_value());
  board->MakeMove(*move4_opt);
  int64_t h4 = board->HashKey();

  auto move5_opt = ParseMove(*board, "g1-m7");
  ASSERT_TRUE(move5_opt.has_value());
  board->MakeMove(*move5_opt);
  int64_t h5 = board->HashKey();
  
  auto move6_opt = ParseMove(*board, "a8-g2");
  ASSERT_TRUE(move6_opt.has_value());
  board->MakeMove(*move6_opt);

  board->UndoMove();
  int64_t h5_1 = board->HashKey();
  board->UndoMove();
  int64_t h4_1 = board->HashKey();
  board->UndoMove();
  int64_t h3_1 = board->HashKey();
  board->UndoMove();
  int64_t h2_1 = board->HashKey();
  board->UndoMove();
  int64_t h1_1 = board->HashKey();
  board->UndoMove();
  int64_t h0_1 = board->HashKey();

  EXPECT_EQ(h0, h0_1);
  EXPECT_EQ(h1, h1_1);
  EXPECT_EQ(h2, h2_1);
  EXPECT_EQ(h3, h3_1);
  EXPECT_EQ(h4, h4_1);
  EXPECT_EQ(h5, h5_1);

  EXPECT_NE(h0, h1);
  EXPECT_NE(h0, h2);
  EXPECT_NE(h0, h3);
  EXPECT_NE(h0, h5);
}

TEST(BoardTest, KeyTest_NullMove) {
  auto board = Board::CreateStandardSetup();
  int64_t h0 = board->HashKey();
  board->MakeNullMove();
  int64_t h1 = board->HashKey();
  board->MakeNullMove();

  board->UndoNullMove();
  int64_t h1_1 = board->HashKey();
  board->UndoNullMove();
  int64_t h0_1 = board->HashKey();

  EXPECT_EQ(h0, h0_1);
  EXPECT_EQ(h1, h1_1);
  EXPECT_NE(h0, h1);
}

TEST(BoardTest, IsKingInCheck) {
  auto board = Board::CreateStandardSetup();
  
  // FIX: Corrected algebraic notation
  auto move_opt = ParseMove(*board, "h2-h3");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);

  move_opt = ParseMove(*board, "b7-c7");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);

  move_opt = ParseMove(*board, "g13-g12");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  
  move_opt = ParseMove(*board, "m8-l8");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  
  // Qxm7
  move_opt = ParseMove(*board, "g1-m7");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);

  EXPECT_TRUE(board->IsKingInCheck(Player(GREEN)));

  board->UndoMove();

  // Qj4
  move_opt = ParseMove(*board, "g1-j4");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  
  // Qd5
  move_opt = ParseMove(*board, "a8-d5");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  
  // Qxb8
  move_opt = ParseMove(*board, "h14-b8"); 
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  EXPECT_TRUE(board->IsKingInCheck(Player(BLUE)));

  // Qxh13
  move_opt = ParseMove(*board, "n7-h13");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);
  EXPECT_TRUE(board->IsKingInCheck(Player(YELLOW)));
}

TEST(BoardTest, IsKingInCheckFalse) {
  auto board = Board::CreateStandardSetup();

  EXPECT_FALSE(board->IsKingInCheck(Player(RED)));
  EXPECT_FALSE(board->IsKingInCheck(Player(BLUE)));
  EXPECT_FALSE(board->IsKingInCheck(Player(YELLOW)));
  EXPECT_FALSE(board->IsKingInCheck(Player(GREEN)));

  // FIX: Corrected algebraic notation
  auto move_opt = ParseMove(*board, "h2-h3");
  ASSERT_TRUE(move_opt.has_value());
  board->MakeMove(*move_opt);

  EXPECT_FALSE(board->IsKingInCheck(Player(RED)));
  EXPECT_FALSE(board->IsKingInCheck(Player(BLUE)));
  EXPECT_FALSE(board->IsKingInCheck(Player(YELLOW)));
  EXPECT_FALSE(board->IsKingInCheck(Player(GREEN)));
}

TEST(BoardTest, MakeAndUndoMoves) {
  auto board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,0,1,1-0,0,0,0-2-x,x,x,yR,yN,1,yK,1,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,2,bP,8,gP,1/bQ,bP,9,gP,1,gK/bK,bP,bP,1,yQ,7,gP,1/bB,11,gP,gB/bN,1,bP,6,gB,2,gP,gN/3,bR,1,rP,6,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,1,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,1,rB,rQ,rK,1,rN,rR,x,x,x");
  ASSERT_NE(board, nullptr);

  int64_t hash = board->HashKey();
  
  auto move1 = FindMove(*board, Loc(12, 3), Loc(11, 3));
  ASSERT_TRUE(move1.has_value());
  board->MakeMove(*move1);
  
  auto move2 = FindMove(*board, Loc(6, 0), Loc(5, 1));
  ASSERT_TRUE(move2.has_value());
  board->MakeMove(*move2);

  auto move3 = FindMove(*board, Loc(0, 4), Loc(2, 5));
  ASSERT_TRUE(move3.has_value());
  board->MakeMove(*move3);

  board->MakeNullMove();

  auto move4 = FindMove(*board, Loc(13, 6), Loc(10, 3));
  ASSERT_TRUE(move4.has_value());
  board->MakeMove(*move4);

  auto move5 = FindMove(*board, Loc(9, 2), Loc(10, 3));
  ASSERT_TRUE(move5.has_value());
  board->MakeMove(*move5);

  auto move6 = FindMove(*board, Loc(7, 4), Loc(9, 2));
  ASSERT_TRUE(move6.has_value());
  board->MakeMove(*move6);

  board->UndoMove();
  board->UndoMove();
  board->UndoMove();
  board->UndoNullMove();
  board->UndoMove();
  board->UndoMove();
  board->UndoMove();

  EXPECT_EQ(hash, board->HashKey());
}

TEST(BoardTest, DeliversCheck) {
  auto board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,4,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  Move move;

  move = MakeMoveForCheckTest(*board, Loc(13, 6), Loc(7, 12));
  EXPECT_TRUE(board->DeliversCheck(move));
  EXPECT_TRUE(move.DeliversCheck(*board));
  EXPECT_TRUE(move.DeliversCheck(*board)); // Test caching

  move = MakeMoveForCheckTest(*board, Loc(13, 8), Loc(7, 2));
  EXPECT_FALSE(board->DeliversCheck(move));
  EXPECT_FALSE(move.DeliversCheck(*board));

  move = MakeMoveForCheckTest(*board, Loc(13, 4), Loc(11, 3));
  EXPECT_FALSE(board->DeliversCheck(move));

  move = MakeMoveForCheckTest(*board, Loc(13, 7), Loc(12, 7));
  EXPECT_FALSE(board->DeliversCheck(move));

  move = MakeMoveForCheckTest(*board, Loc(12, 5), Loc(11, 5));
  EXPECT_FALSE(board->DeliversCheck(move));

  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,1,rB,2,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,1,rN,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  move = MakeMoveForCheckTest(*board, Loc(11, 4), Loc(8, 1));
  EXPECT_TRUE(board->DeliversCheck(move));
  
  board = ParseBoardFromFEN("R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,1,yP,yP,yP,yP,x,x,x/x,x,x,3,yP,4,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,2,rN,7,gP,gB/bQ,bP,9,gP,1,gK/bK,1,bP,9,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,1,rB,2,rP,3,x,x,x/x,x,x,rP,rP,rP,rP,1,rP,rP,rP,x,x,x/x,x,x,rR,1,rB,rQ,rK,1,rN,rR,x,x,x");
  ASSERT_NE(board, nullptr);
  move = MakeMoveForCheckTest(*board, Loc(5, 4), Loc(6, 2));
  EXPECT_TRUE(board->DeliversCheck(move));
}

TEST(BoardTest, Promotions) {
  auto board = ParseBoardFromFEN("Y-0,0,0,0-0,0,0,1-0,0,1,1-0,0,0,0-0-{'enPassant':('','c8:d8','','')}-x,x,x,1,yN,1,yK,2,yN,yR,x,x,x/x,x,x,1,yP,yP,3,yP,yP,x,x,x/x,x,x,3,yP,1,yP,2,x,x,x/bR,bP,5,yP,4,gP,gR/1,bP,10,gP,gN/bB,bP,10,gP,1/bK,2,bP,7,gP,1,gK/4,rR,7,gP,1/11,gP,2/1,bP,1,yP,9,gN/1,bP,8,gP,1,gP,gR/x,x,x,rP,1,rN,1,rP,gB,2,x,x,x/x,x,x,2,rP,rP,1,rP,2,x,x,x/x,x,x,4,rK,3,x,x,x");
  ASSERT_NE(board, nullptr);
  auto move_or = ParseMove(*board, "d5-d4=Q");
  int64_t h0 = board->HashKey();
  int piece_eval = board->PieceEvaluation();

  ASSERT_TRUE(move_or.has_value());

  board->MakeMove(*move_or);
  auto piece = board->GetPiece(Loc(10, 3));
  int64_t h1 = board->HashKey();
  int piece_eval2 = board->PieceEvaluation();

  EXPECT_TRUE(piece.Present());
  EXPECT_EQ(piece.GetColor(), YELLOW);
  EXPECT_EQ(piece.GetPieceType(), QUEEN);
  EXPECT_NE(h0, h1);
  EXPECT_NE(piece_eval, piece_eval2);

  // Undo promotion
  board->UndoMove();
  EXPECT_EQ(h0, board->HashKey());
  EXPECT_EQ(piece_eval, board->PieceEvaluation());
  auto pawn_piece = board->GetPiece(Loc(9, 3));
  EXPECT_TRUE(pawn_piece.Present());
  EXPECT_EQ(pawn_piece.GetPieceType(), PAWN);
}

}  // namespace chess

// To run tests, use the following command:
// bazel test //:board_test --test_output=streamed