#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <cstdint>
#include "types.h"

namespace Zobrist {
    extern uint64_t PIECES[PIECE_NB][64];
    extern uint64_t CASTLING[16];
    extern uint64_t EP[65]; // 0-63 or 64 for SQ_NONE
    extern uint64_t SIDE;

    void init();
}

#endif // ZOBRIST_H
