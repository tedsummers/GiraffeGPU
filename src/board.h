#ifndef BOARD_H
#define BOARD_H

#include "types.h"
#include "bitboard.h"
#include "zobrist.h"
#include <vector>
#include <string>
#include <cassert>

struct StateInfo {
    uint64_t key;
    Piece captured;
    Square ep_square;
    uint8_t castling_rights;
    int halfmove_clock;
};

class Board {
public:
    Board();
    explicit Board(const std::string& fen);

    bool set_fen(const std::string& fen);
    std::string to_fen() const;

    inline Piece piece_at(Square sq) const { return square_piece_[sq]; }
    inline Bitboard piece_bb(Piece p) const { return pieces_bb_[p]; }
    inline Bitboard piece_type_bb(PieceType pt, Color c) const { return pieces_bb_[make_piece(c, pt)]; }
    inline Bitboard color_bb(Color c) const { return color_bb_[c]; }
    inline Bitboard occupied_bb() const { return color_bb_[WHITE] | color_bb_[BLACK]; }
    inline Color side_to_move() const { return stm_; }
    inline uint8_t castling_rights() const { return castling_rights_; }
    inline Square ep_square() const { return ep_square_; }
    inline int halfmove_clock() const { return halfmove_clock_; }
    inline int fullmove_number() const { return fullmove_number_; }
    inline uint64_t key() const { return key_; }

    inline Square king_square(Color c) const {
        Bitboard b = pieces_bb_[make_piece(c, KING)];
        return b ? lsb(b) : SQ_NONE;
    }

    inline int piece_count(Piece p) const {
        return popcount(pieces_bb_[p]);
    }

    inline int piece_type_count(PieceType pt, Color c) const {
        return popcount(pieces_bb_[make_piece(c, pt)]);
    }

    bool is_square_attacked(Square sq, Color by_color) const;
    Bitboard attackers_to(Square sq, Color by_color, Bitboard occ) const;
    PieceType least_valuable_attacker(Square sq, Color by_color, Bitboard occ, Square& out_from) const;

    bool in_check() const {
        Square k = king_square(stm_);
        return (k != SQ_NONE) && is_square_attacked(k, ~stm_);
    }

    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_draw() const;
    bool is_repetition() const;

    bool make_move(Move m);
    void unmake_move(Move m);

    void make_null_move();
    void unmake_null_move();

    void generate_legal_moves(std::vector<Move>& moves) const;
    void generate_captures(std::vector<Move>& moves) const;

    bool is_legal_move(Move m) const;
    bool gives_check(Move m) const;
    Move parse_move_uci(const std::string& uci) const;

    void print() const;

private:
    void clear();
    void update_key();

    Bitboard pieces_bb_[PIECE_NB];
    Bitboard color_bb_[COLOR_NB];
    Piece square_piece_[64];

    Color stm_;
    uint8_t castling_rights_;
    Square ep_square_;
    int halfmove_clock_;
    int fullmove_number_;
    uint64_t key_;

    std::vector<StateInfo> history_;
};

#endif // BOARD_H
