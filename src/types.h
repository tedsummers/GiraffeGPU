#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>
#include <algorithm>
#include <iostream>

typedef uint64_t Bitboard;
typedef int32_t Value;

constexpr Value VALUE_ZERO      = 0;
constexpr Value VALUE_DRAW      = 0;
constexpr Value VALUE_MATE      = 30000;
constexpr Value VALUE_INFINITE  = 32000;
constexpr Value VALUE_MATE_THRESHOLD = 28000;

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1,
    COLOR_NB = 2
};

inline Color operator~(Color c) {
    return static_cast<Color>(c ^ 1);
}

enum PieceType : uint8_t {
    NO_PIECE_TYPE = 0,
    PAWN   = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK   = 4,
    QUEEN  = 5,
    KING   = 6,
    PIECE_TYPE_NB = 7
};

enum Piece : uint8_t {
    NO_PIECE = 0,
    W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
    B_PAWN = 7, B_KNIGHT = 8, B_BISHOP = 9, B_ROOK = 10, B_QUEEN = 11, B_KING = 12,
    PIECE_NB = 13
};

inline Piece make_piece(Color c, PieceType pt) {
    return static_cast<Piece>((c == WHITE) ? pt : (pt + 6));
}

inline PieceType piece_type(Piece p) {
    return static_cast<PieceType>((p > 6) ? (p - 6) : p);
}

inline Color piece_color(Piece p) {
    return (p <= 6) ? WHITE : BLACK;
}

enum Square : uint8_t {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    SQ_NONE = 64
};

enum File : uint8_t {
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB
};

enum Rank : uint8_t {
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB
};

inline File square_file(Square sq) {
    return static_cast<File>(sq & 7);
}

inline Rank square_rank(Square sq) {
    return static_cast<Rank>(sq >> 3);
}

inline Square make_square(File f, Rank r) {
    return static_cast<Square>((r << 3) | f);
}

enum CastlingRights : uint8_t {
    NO_CASTLING = 0,
    WHITE_OO   = 1,
    WHITE_OOO  = 2,
    BLACK_OO   = 4,
    BLACK_OOO  = 8,
    WHITE_CASTLING = WHITE_OO | WHITE_OOO,
    BLACK_CASTLING = BLACK_OO | BLACK_OOO,
    ALL_CASTLING   = WHITE_CASTLING | BLACK_CASTLING
};

// Move Representation:
// 16 bits:
// bits 0-5:   from square (0..63)
// bits 6-11:  to square   (0..63)
// bits 12-13: promotion piece type - 2 (0=KNIGHT, 1=BISHOP, 2=ROOK, 3=QUEEN)
// bits 14-15: move type (00=NORMAL, 01=PROMOTION, 10=EN_PASSANT, 11=CASTLING)

enum MoveType : uint16_t {
    MOVE_NORMAL     = 0 << 14,
    MOVE_PROMOTION  = 1 << 14,
    MOVE_EN_PASSANT = 2 << 14,
    MOVE_CASTLING   = 3 << 14
};

struct Move {
    uint16_t data;

    Move() : data(0) {}
    explicit Move(uint16_t d) : data(d) {}
    Move(Square from, Square to, MoveType type = MOVE_NORMAL, PieceType promo = KNIGHT) {
        data = static_cast<uint16_t>(from | (to << 6) | type | (((promo >= KNIGHT && promo <= QUEEN) ? (promo - KNIGHT) : 0) << 12));
    }

    inline Square from_sq() const { return static_cast<Square>(data & 0x3F); }
    inline Square to_sq() const { return static_cast<Square>((data >> 6) & 0x3F); }
    inline MoveType type() const { return static_cast<MoveType>(data & 0xC000); }
    inline PieceType promotion_type() const {
        return (type() == MOVE_PROMOTION) ? static_cast<PieceType>(((data >> 12) & 0x3) + KNIGHT) : NO_PIECE_TYPE;
    }
    inline bool is_promotion() const { return (data & 0xC000) == MOVE_PROMOTION; }
    inline bool is_castling() const { return (data & 0xC000) == MOVE_CASTLING; }
    inline bool is_en_passant() const { return (data & 0xC000) == MOVE_EN_PASSANT; }
    inline bool is_null() const { return data == 0; }

    inline bool operator==(const Move& other) const { return data == other.data; }
    inline bool operator!=(const Move& other) const { return data != other.data; }

    std::string to_uci() const {
        if (is_null()) return "0000";
        Square f = from_sq();
        Square t = to_sq();
        std::string s;
        s += static_cast<char>('a' + square_file(f));
        s += static_cast<char>('1' + square_rank(f));
        s += static_cast<char>('a' + square_file(t));
        s += static_cast<char>('1' + square_rank(t));
        if (is_promotion()) {
            switch (promotion_type()) {
                case KNIGHT: s += 'n'; break;
                case BISHOP: s += 'b'; break;
                case ROOK:   s += 'r'; break;
                case QUEEN:  s += 'q'; break;
                default:     s += 'q'; break;
            }
        }
        return s;
    }
};

const Move MOVE_NONE = Move(0);

#endif // TYPES_H
