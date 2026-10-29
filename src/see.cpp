#include "see.h"
#include <algorithm>

namespace SEE {

Value evaluate(const Board& board, Move m) {
    Square from = m.from_sq();
    Square to = m.to_sq();

    Piece moving_piece = board.piece_at(from);
    Piece captured_piece = board.piece_at(to);

    if (m.is_en_passant()) {
        captured_piece = make_piece(~board.side_to_move(), PAWN);
    }

    PieceType moving_pt = piece_type(moving_piece);
    PieceType captured_pt = (captured_piece != NO_PIECE) ? piece_type(captured_piece) : NO_PIECE_TYPE;

    Value gain[32];
    int d = 0;
    gain[d] = PIECE_VALUES[captured_pt];

    Bitboard occ = board.occupied_bb();
    occ ^= square_bb(from);

    Color side = ~board.side_to_move();

    while (true) {
        d++;
        gain[d] = PIECE_VALUES[moving_pt] - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) {
            break;
        }

        Square next_from = SQ_NONE;
        moving_pt = board.least_valuable_attacker(to, side, occ, next_from);
        if (moving_pt == NO_PIECE_TYPE || next_from == SQ_NONE) {
            break;
        }

        occ ^= square_bb(next_from);
        side = ~side;
    }

    while (--d > 0) {
        gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    }

    return gain[0];
}

} // namespace SEE
