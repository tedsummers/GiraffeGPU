#ifndef SEE_H
#define SEE_H

#include "types.h"
#include "board.h"

namespace SEE {
    const Value PIECE_VALUES[PIECE_TYPE_NB] = {
        0, 100, 325, 330, 500, 900, 20000
    };

    Value evaluate(const Board& board, Move m);
}

#endif // SEE_H
