#ifndef SYZYGY_TB_H
#define SYZYGY_TB_H

#include "types.h"
#include "board.h"
#include <string>

namespace Syzygy {

// Initializes Syzygy tablebases from a directory path string
bool init(const std::string& path);

// Returns the max piece count of available tablebases (e.g., 5 or 6, or 0 if none)
int max_pieces();

// Probe WDL during search. Returns true if hit, writing WDL score to out_score.
bool probe_wdl(const Board& board, Value& out_score, int ply);

// Probe Root DTZ/WDL. Returns true if hit, writing best move to out_move and score to out_score.
bool probe_root(const Board& board, Move& out_move, Value& out_score);

// Filters root_moves in-place to keep only tablebase-optimal moves (by WDL and DTZ) so search can refine them.
bool filter_root_moves(const Board& board, MoveList& root_moves, Value& out_score);

} // namespace Syzygy

#endif // SYZYGY_TB_H
