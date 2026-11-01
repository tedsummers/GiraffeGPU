#include "giraffe_features.h"
#include "see.h"
#include <cmath>
#include <tuple>
#include <algorithm>

namespace Giraffe {

static const int GIRAFFE_MAT[PIECE_TYPE_NB] = {
    0, 100, 325, 330, 500, 900, 20000
};

static inline float norm_coord(int x) {
    return 0.1429f * static_cast<float>(x);
}

static inline float norm_count(int x, float max_val) {
    return static_cast<float>(x) / max_val;
}

bool AttackMaps::is_safe(PieceType pt, Color c, Square sq) const {
    PieceType oppAtk = (c == WHITE) ? blackLVA[sq] : whiteLVA[sq];
    PieceType friendAtk = (c == WHITE) ? whiteLVA[sq] : blackLVA[sq];
    uint8_t numFriend = (c == WHITE) ? whiteNumAtk[sq] : blackNumAtk[sq];

    if (oppAtk == NO_PIECE_TYPE) {
        return true;
    }
    if (GIRAFFE_MAT[oppAtk] < GIRAFFE_MAT[pt]) {
        return false;
    }
    return (friendAtk != NO_PIECE_TYPE) && (numFriend > 1);
}

void compute_attack_maps(const Board& board, AttackMaps& maps) {
    Bitboard occ = board.occupied_bb();

    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        // White attackers
        Bitboard wAtk = board.attackers_to(sq, WHITE, occ);
        maps.whiteNumAtk[sq] = static_cast<uint8_t>(popcount(wAtk));
        Square dummy;
        maps.whiteLVA[sq] = board.least_valuable_attacker(sq, WHITE, occ, dummy);

        // Black attackers
        Bitboard bAtk = board.attackers_to(sq, BLACK, occ);
        maps.blackNumAtk[sq] = static_cast<uint8_t>(popcount(bAtk));
        maps.blackLVA[sq] = board.least_valuable_attacker(sq, BLACK, occ, dummy);

        // Control values
        if (maps.whiteLVA[sq] == NO_PIECE_TYPE) {
            maps.whiteCtrl[sq] = 0.0f;
        } else {
            float val = static_cast<float>(GIRAFFE_MAT[KING] + GIRAFFE_MAT[KING] / 2 - GIRAFFE_MAT[maps.whiteLVA[sq]]);
            maps.whiteCtrl[sq] = val / (static_cast<float>(GIRAFFE_MAT[KING]) * 2.0f);
        }

        if (maps.blackLVA[sq] == NO_PIECE_TYPE) {
            maps.blackCtrl[sq] = 0.0f;
        } else {
            float val = static_cast<float>(GIRAFFE_MAT[KING] + GIRAFFE_MAT[KING] / 2 - GIRAFFE_MAT[maps.blackLVA[sq]]);
            maps.blackCtrl[sq] = val / (static_cast<float>(GIRAFFE_MAT[KING]) * 2.0f);
        }
    }
}

// Helpers for pushing features into the vector
struct FeatureWriter {
    float* data;
    int idx;

    FeatureWriter(float* d) : data(d), idx(0) {}

    inline void push(float val) {
        data[idx++] = val;
    }
    inline void push_bool(bool b) {
        data[idx++] = b ? 1.0f : 0.0f;
    }
    inline void push_coords(bool exists, Square sq, bool must_exist = false) {
        if (!must_exist) {
            push_bool(exists);
        }
        if (exists) {
            push(norm_coord(square_file(sq)));
            push(norm_coord(square_rank(sq)));
        } else {
            push(0.0f);
            push(0.0f);
        }
    }
    inline void push_threat(Square sq, bool exists, const AttackMaps& maps) {
        if (exists) {
            push(maps.whiteCtrl[sq]);
            push(maps.blackCtrl[sq]);
        } else {
            push(0.0f);
            push(0.0f);
        }
    }
};

static void push_pawns_for_color(FeatureWriter& w, const Board& board, Color c, const AttackMaps& maps) {
    Bitboard pawns = board.piece_type_bb(PAWN, c);
    std::tuple<bool, Square> slots[8];
    for (int i = 0; i < 8; ++i) std::get<0>(slots[i]) = false;

    // First pass: assign pawns to their files if slot free
    Bitboard unassigned = 0;
    while (pawns) {
        Square sq = pop_lsb(pawns);
        int f = square_file(sq);
        if (!std::get<0>(slots[f])) {
            std::get<0>(slots[f]) = true;
            std::get<1>(slots[f]) = sq;
        } else {
            unassigned |= square_bb(sq);
        }
    }

    // Second pass: assign remaining pawns to nearest open slot
    while (unassigned) {
        Square sq = pop_lsb(unassigned);
        int f = square_file(sq);
        int best_dist = 999;
        int best_slot = 0;
        for (int i = 0; i < 8; ++i) {
            if (!std::get<0>(slots[i])) {
                int dist = std::abs(f - i);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_slot = i;
                }
            }
        }
        std::get<0>(slots[best_slot]) = true;
        std::get<1>(slots[best_slot]) = sq;
    }

    for (int i = 0; i < 8; ++i) {
        bool exists = std::get<0>(slots[i]);
        Square sq = std::get<1>(slots[i]);
        w.push_coords(exists, sq);
        w.push_threat(sq, exists, maps);
    }
}

static void push_sliding_attacks(FeatureWriter& w, const Board& board, Square sq, PieceType pt,
                                 Color c, bool exists, const AttackMaps& maps) {
    int safe_moves = 0;
    int f_start = square_file(sq);
    int r_start = square_rank(sq);

    // Orthogonal rays for Rook and Queen
    if (pt == ROOK || pt == QUEEN) {
        const int df[4] = { 1, -1, 0, 0 };
        const int dr[4] = { 0, 0, 1, -1 };
        for (int dir = 0; dir < 4; ++dir) {
            int count = 0;
            int f = f_start + df[dir];
            int r = r_start + dr[dir];
            while (f >= 0 && f <= 7 && r >= 0 && r <= 7 && exists) {
                count++;
                Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
                if (maps.is_safe(pt, c, s)) safe_moves++;
                if (board.piece_at(s) != NO_PIECE) break;
                f += df[dir];
                r += dr[dir];
            }
            w.push(norm_count(count, 7.0f));
        }
    }

    // Diagonal rays for Bishop and Queen
    if (pt == BISHOP || pt == QUEEN) {
        const int df[4] = { 1, -1, 1, -1 };
        const int dr[4] = { 1, 1, -1, -1 };
        for (int dir = 0; dir < 4; ++dir) {
            int count = 0;
            int f = f_start + df[dir];
            int r = r_start + dr[dir];
            while (f >= 0 && f <= 7 && r >= 0 && r <= 7 && exists) {
                count++;
                Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
                if (maps.is_safe(pt, c, s)) safe_moves++;
                if (board.piece_at(s) != NO_PIECE) break;
                f += df[dir];
                r += dr[dir];
            }
            w.push(norm_count(count, 7.0f));
        }
    }

    // Knight jumps
    if (pt == KNIGHT) {
        const int df[8] = { 2, -2, 1, -1, -2, 2, -1, 1 };
        const int dr[8] = { 1, 1, 2, 2, -1, -1, -2, -2 };
        for (int i = 0; i < 8; ++i) {
            int f = f_start + df[i];
            int r = r_start + dr[i];
            if (f >= 0 && f <= 7 && r >= 0 && r <= 7 && exists) {
                Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
                if (maps.is_safe(pt, c, s)) safe_moves++;
            }
        }
    }

    w.push(norm_count(exists ? safe_moves : 0, 16.0f));
}

static void push_pair_pieces(FeatureWriter& w, const Board& board, PieceType pt, Color c, const AttackMaps& maps) {
    Bitboard pieces = board.piece_type_bb(pt, c);
    int count = popcount(pieces);

    bool firstExists = false, secondExists = false;
    Square firstSq = SQ_NONE, secondSq = SQ_NONE;

    if (count == 1) {
        Square sq = pop_lsb(pieces);
        if (square_file(sq) < 4) {
            firstExists = true;
            firstSq = sq;
        } else {
            secondExists = true;
            secondSq = sq;
        }
    } else if (count >= 2) {
        firstSq = pop_lsb(pieces);
        secondSq = pop_lsb(pieces);
        firstExists = true;
        secondExists = true;
        if (square_file(firstSq) > square_file(secondSq)) {
            std::swap(firstSq, secondSq);
        }
    }

    // First piece
    w.push_coords(firstExists, firstSq);
    push_sliding_attacks(w, board, firstSq, pt, c, firstExists, maps);
    w.push_threat(firstSq, firstExists, maps);

    // Second piece
    w.push_coords(secondExists, secondSq);
    push_sliding_attacks(w, board, secondSq, pt, c, secondExists, maps);
    w.push_threat(secondSq, secondExists, maps);
}

void extract_features(const Board& board, float* out_features) {
    FeatureWriter w(out_features);
    AttackMaps maps;
    compute_attack_maps(board, maps);

    // 1. Global features (Group 0)
    w.push(norm_count(board.piece_type_count(QUEEN, WHITE), 1.0f));
    w.push(norm_count(board.piece_type_count(ROOK, WHITE), 2.0f));
    w.push(norm_count(board.piece_type_count(BISHOP, WHITE), 2.0f));
    w.push(norm_count(board.piece_type_count(KNIGHT, WHITE), 2.0f));
    w.push(norm_count(board.piece_type_count(PAWN, WHITE), 8.0f));
    w.push(norm_count(board.piece_type_count(QUEEN, BLACK), 1.0f));
    w.push(norm_count(board.piece_type_count(ROOK, BLACK), 2.0f));
    w.push(norm_count(board.piece_type_count(BISHOP, BLACK), 2.0f));
    w.push(norm_count(board.piece_type_count(KNIGHT, BLACK), 2.0f));
    w.push(norm_count(board.piece_type_count(PAWN, BLACK), 8.0f));

    // Side to move
    w.push_bool(board.side_to_move() == WHITE);

    // King coordinates and castling rights
    Square wkPos = board.king_square(WHITE);
    Square bkPos = board.king_square(BLACK);

    w.push_coords(true, wkPos, true);
    w.push_bool(board.castling_rights() & WHITE_OO);
    w.push_bool(board.castling_rights() & WHITE_OOO);

    w.push_coords(true, bkPos, true);
    w.push_bool(board.castling_rights() & BLACK_OO);
    w.push_bool(board.castling_rights() & BLACK_OOO);

    // 2. Pawns
    push_pawns_for_color(w, board, WHITE, maps);
    push_pawns_for_color(w, board, BLACK, maps);

    // 3. Queens
    // White Queen
    Bitboard wQueens = board.piece_type_bb(QUEEN, WHITE);
    bool wQExists = (wQueens != 0);
    Square wQSq = wQExists ? lsb(wQueens) : SQ_NONE;
    w.push_coords(wQExists, wQSq);
    push_sliding_attacks(w, board, wQSq, QUEEN, WHITE, wQExists, maps);
    w.push_threat(wQSq, wQExists, maps);

    // Black Queen
    Bitboard bQueens = board.piece_type_bb(QUEEN, BLACK);
    bool bQExists = (bQueens != 0);
    Square bQSq = bQExists ? lsb(bQueens) : SQ_NONE;
    w.push_coords(bQExists, bQSq);
    push_sliding_attacks(w, board, bQSq, QUEEN, BLACK, bQExists, maps);
    w.push_threat(bQSq, bQExists, maps);

    // 4. Rooks
    push_pair_pieces(w, board, ROOK, WHITE, maps);
    w.push_bool(board.castling_rights() & WHITE_OO);
    w.push_bool(board.castling_rights() & WHITE_OOO);

    push_pair_pieces(w, board, ROOK, BLACK, maps);
    w.push_bool(board.castling_rights() & BLACK_OO);
    w.push_bool(board.castling_rights() & BLACK_OOO);

    // 5. Bishops
    push_pair_pieces(w, board, BISHOP, WHITE, maps);
    push_pair_pieces(w, board, BISHOP, BLACK, maps);

    // 6. Knights
    push_pair_pieces(w, board, KNIGHT, WHITE, maps);
    push_pair_pieces(w, board, KNIGHT, BLACK, maps);

    // 7. Square Attack Maps (64 squares * 2)
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        w.push(maps.whiteCtrl[sq]);
        w.push(maps.blackCtrl[sq]);
    }

    assert(w.idx == NUM_FEATURES);
}

} // namespace Giraffe
