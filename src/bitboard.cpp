#include "bitboard.h"
#include "magic_moves.h"
#include <vector>

namespace Bitboards {

Bitboard FILE_MASKS[8];
Bitboard RANK_MASKS[8];
Bitboard PAWN_ATTACKS[2][64];
Bitboard KNIGHT_ATTACKS[64];
Bitboard KING_ATTACKS[64];
Bitboard BETWEEN_BB[64][64];
Bitboard LINE_BB[64][64];

void init() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    // Initialize Pradu Cannan magic moves (from Giraffe)
    initmagicmoves();

    // Files and Ranks
    for (int i = 0; i < 8; ++i) {
        Bitboard file = 0x0101010101010101ULL << i;
        Bitboard rank = 0x00000000000000FFULL << (i * 8);
        FILE_MASKS[i] = file;
        RANK_MASKS[i] = rank;
    }

    // Pawn attacks
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        int r = square_rank(sq), f = square_file(sq);
        // White pawns move +rank
        Bitboard wAttacks = 0;
        if (r < 7) {
            if (f > 0) wAttacks |= square_bb(make_square(static_cast<File>(f - 1), static_cast<Rank>(r + 1)));
            if (f < 7) wAttacks |= square_bb(make_square(static_cast<File>(f + 1), static_cast<Rank>(r + 1)));
        }
        PAWN_ATTACKS[WHITE][sq] = wAttacks;

        // Black pawns move -rank
        Bitboard bAttacks = 0;
        if (r > 0) {
            if (f > 0) bAttacks |= square_bb(make_square(static_cast<File>(f - 1), static_cast<Rank>(r - 1)));
            if (f < 7) bAttacks |= square_bb(make_square(static_cast<File>(f + 1), static_cast<Rank>(r - 1)));
        }
        PAWN_ATTACKS[BLACK][sq] = bAttacks;
    }

    // Knight attacks
    const int knight_dr[8] = { 1, 2,  2,  1, -1, -2, -2, -1 };
    const int knight_df[8] = { 2, 1, -1, -2, -2, -1,  1,  2 };
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        int r = square_rank(sq), f = square_file(sq);
        Bitboard att = 0;
        for (int i = 0; i < 8; ++i) {
            int nr = r + knight_dr[i];
            int nf = f + knight_df[i];
            if (nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7) {
                att |= square_bb(make_square(static_cast<File>(nf), static_cast<Rank>(nr)));
            }
        }
        KNIGHT_ATTACKS[sq] = att;
    }

    // King attacks
    const int king_dr[8] = { 1, 1, 1, 0, 0, -1, -1, -1 };
    const int king_df[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        int r = square_rank(sq), f = square_file(sq);
        Bitboard att = 0;
        for (int i = 0; i < 8; ++i) {
            int nr = r + king_dr[i];
            int nf = f + king_df[i];
            if (nr >= 0 && nr <= 7 && nf >= 0 && nf <= 7) {
                att |= square_bb(make_square(static_cast<File>(nf), static_cast<Rank>(nr)));
            }
        }
        KING_ATTACKS[sq] = att;
    }

    // Between and Line bitboards
    for (Square sq1 = SQ_A1; sq1 <= SQ_H8; sq1 = static_cast<Square>(sq1 + 1)) {
        for (Square sq2 = SQ_A1; sq2 <= SQ_H8; sq2 = static_cast<Square>(sq2 + 1)) {
            BETWEEN_BB[sq1][sq2] = 0;
            LINE_BB[sq1][sq2] = 0;

            if (sq1 == sq2) continue;

            Bitboard b1 = square_bb(sq1);
            Bitboard b2 = square_bb(sq2);

            Bitboard rAttacks = rook_attacks(sq1, 0);
            Bitboard bAttacks = bishop_attacks(sq1, 0);

            if (rAttacks & b2) {
                LINE_BB[sq1][sq2] = (rook_attacks(sq1, 0) & rook_attacks(sq2, 0)) | b1 | b2;
                BETWEEN_BB[sq1][sq2] = rook_attacks(sq1, b2) & rook_attacks(sq2, b1);
            } else if (bAttacks & b2) {
                LINE_BB[sq1][sq2] = (bishop_attacks(sq1, 0) & bishop_attacks(sq2, 0)) | b1 | b2;
                BETWEEN_BB[sq1][sq2] = bishop_attacks(sq1, b2) & bishop_attacks(sq2, b1);
            }
        }
    }
}

Bitboard bishop_attacks(Square sq, Bitboard occupied) {
    return Bmagic(sq, occupied);
}

Bitboard rook_attacks(Square sq, Bitboard occupied) {
    return Rmagic(sq, occupied);
}

Bitboard queen_attacks(Square sq, Bitboard occupied) {
    return Bmagic(sq, occupied) | Rmagic(sq, occupied);
}

} // namespace Bitboards
