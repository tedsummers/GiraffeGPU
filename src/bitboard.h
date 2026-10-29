#ifndef BITBOARD_H
#define BITBOARD_H

#include "types.h"
#include <cstdint>

inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

inline Square lsb(Bitboard b) {
    return static_cast<Square>(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
    return static_cast<Square>(63 - __builtin_clzll(b));
}

inline Square pop_lsb(Bitboard &b) {
    Square s = lsb(b);
    b &= b - 1;
    return s;
}

inline Bitboard square_bb(Square sq) {
    return 1ULL << sq;
}

namespace Bitboards {
    extern Bitboard FILE_MASKS[8];
    extern Bitboard RANK_MASKS[8];
    extern Bitboard PAWN_ATTACKS[2][64];
    extern Bitboard KNIGHT_ATTACKS[64];
    extern Bitboard KING_ATTACKS[64];
    extern Bitboard BETWEEN_BB[64][64];
    extern Bitboard LINE_BB[64][64];

    void init();

    Bitboard bishop_attacks(Square sq, Bitboard occupied);
    Bitboard rook_attacks(Square sq, Bitboard occupied);
    Bitboard queen_attacks(Square sq, Bitboard occupied);
}

#endif // BITBOARD_H
