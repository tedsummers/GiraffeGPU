#include "board.h"
#include <sstream>
#include <iostream>
#include <cctype>

Board::Board() {
    Bitboards::init();
    Zobrist::init();
    clear();
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Board::Board(const std::string& fen) {
    Bitboards::init();
    Zobrist::init();
    clear();
    set_fen(fen);
}

void Board::clear() {
    for (int p = 0; p < PIECE_NB; ++p) pieces_bb_[p] = 0;
    color_bb_[WHITE] = 0;
    color_bb_[BLACK] = 0;
    for (int sq = 0; sq < 64; ++sq) square_piece_[sq] = NO_PIECE;

    stm_ = WHITE;
    castling_rights_ = NO_CASTLING;
    ep_square_ = SQ_NONE;
    halfmove_clock_ = 0;
    fullmove_number_ = 1;
    key_ = 0;
    history_.clear();
}

void Board::update_key() {
    key_ = 0;
    for (Square sq = SQ_A1; sq <= SQ_H8; sq = static_cast<Square>(sq + 1)) {
        Piece p = square_piece_[sq];
        if (p != NO_PIECE) {
            key_ ^= Zobrist::PIECES[p][sq];
        }
    }
    if (stm_ == BLACK) {
        key_ ^= Zobrist::SIDE;
    }
    key_ ^= Zobrist::CASTLING[castling_rights_ & 0xF];
    key_ ^= Zobrist::EP[ep_square_];
}

bool Board::set_fen(const std::string& fen) {
    clear();
    std::istringstream iss(fen);
    std::string piece_placement, active_color, castling, en_passant;
    int halfmove = 0, fullmove = 1;

    if (!(iss >> piece_placement >> active_color)) {
        return false;
    }

    iss >> castling;
    iss >> en_passant;
    iss >> halfmove >> fullmove;

    // Piece placement
    Square sq = SQ_A8;
    for (char c : piece_placement) {
        if (c == '/') {
            sq = static_cast<Square>(sq - 16);
        } else if (std::isdigit(c)) {
            sq = static_cast<Square>(sq + (c - '0'));
        } else {
            Piece p = NO_PIECE;
            switch (c) {
                case 'P': p = W_PAWN; break;
                case 'N': p = W_KNIGHT; break;
                case 'B': p = W_BISHOP; break;
                case 'R': p = W_ROOK; break;
                case 'Q': p = W_QUEEN; break;
                case 'K': p = W_KING; break;
                case 'p': p = B_PAWN; break;
                case 'n': p = B_KNIGHT; break;
                case 'b': p = B_BISHOP; break;
                case 'r': p = B_ROOK; break;
                case 'q': p = B_QUEEN; break;
                case 'k': p = B_KING; break;
                default: break;
            }
            if (p != NO_PIECE) {
                square_piece_[sq] = p;
                pieces_bb_[p] |= square_bb(sq);
                color_bb_[piece_color(p)] |= square_bb(sq);
                sq = static_cast<Square>(sq + 1);
            }
        }
    }

    // Active color
    stm_ = (active_color == "b") ? BLACK : WHITE;

    // Castling rights
    castling_rights_ = NO_CASTLING;
    if (castling.find('K') != std::string::npos) castling_rights_ |= WHITE_OO;
    if (castling.find('Q') != std::string::npos) castling_rights_ |= WHITE_OOO;
    if (castling.find('k') != std::string::npos) castling_rights_ |= BLACK_OO;
    if (castling.find('q') != std::string::npos) castling_rights_ |= BLACK_OOO;

    // En passant
    if (en_passant != "-" && en_passant.length() >= 2) {
        File f = static_cast<File>(en_passant[0] - 'a');
        Rank r = static_cast<Rank>(en_passant[1] - '1');
        if (f >= FILE_A && f <= FILE_H && r >= RANK_1 && r <= RANK_8) {
            ep_square_ = make_square(f, r);
        } else {
            ep_square_ = SQ_NONE;
        }
    } else {
        ep_square_ = SQ_NONE;
    }

    halfmove_clock_ = halfmove;
    fullmove_number_ = fullmove;

    update_key();
    return true;
}

std::string Board::to_fen() const {
    std::string fen;
    for (int r = 7; r >= 0; --r) {
        int empty_count = 0;
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            Piece p = square_piece_[sq];
            if (p == NO_PIECE) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    fen += std::to_string(empty_count);
                    empty_count = 0;
                }
                const char piece_chars[] = ".PNBRQKpnbrqk";
                fen += piece_chars[p];
            }
        }
        if (empty_count > 0) {
            fen += std::to_string(empty_count);
        }
        if (r > 0) fen += '/';
    }

    fen += (stm_ == WHITE) ? " w " : " b ";

    std::string castling_str;
    if (castling_rights_ & WHITE_OO) castling_str += 'K';
    if (castling_rights_ & WHITE_OOO) castling_str += 'Q';
    if (castling_rights_ & BLACK_OO) castling_str += 'k';
    if (castling_rights_ & BLACK_OOO) castling_str += 'q';
    if (castling_str.empty()) castling_str = "-";
    fen += castling_str;

    fen += ' ';
    if (ep_square_ != SQ_NONE) {
        fen += static_cast<char>('a' + square_file(ep_square_));
        fen += static_cast<char>('1' + square_rank(ep_square_));
    } else {
        fen += '-';
    }

    fen += " " + std::to_string(halfmove_clock_) + " " + std::to_string(fullmove_number_);
    return fen;
}

bool Board::is_square_attacked(Square sq, Color by_color) const {
    if (sq >= 64) return false;
    Bitboard occ = occupied_bb();
    // Pawn attacks: if a pawn of by_color were on sq, where would it attack?
    // Invert color for attack source
    Color defender_color = ~by_color;
    if (Bitboards::PAWN_ATTACKS[defender_color][sq] & pieces_bb_[make_piece(by_color, PAWN)]) {
        return true;
    }
    if (Bitboards::KNIGHT_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KNIGHT)]) {
        return true;
    }
    if (Bitboards::bishop_attacks(sq, occ) & (pieces_bb_[make_piece(by_color, BISHOP)] | pieces_bb_[make_piece(by_color, QUEEN)])) {
        return true;
    }
    if (Bitboards::rook_attacks(sq, occ) & (pieces_bb_[make_piece(by_color, ROOK)] | pieces_bb_[make_piece(by_color, QUEEN)])) {
        return true;
    }
    if (Bitboards::KING_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KING)]) {
        return true;
    }
    return false;
}

Bitboard Board::attackers_to(Square sq, Color by_color, Bitboard occ) const {
    if (sq >= 64) return 0;
    Bitboard attackers = 0;
    Color defender_color = ~by_color;
    attackers |= (Bitboards::PAWN_ATTACKS[defender_color][sq] & pieces_bb_[make_piece(by_color, PAWN)]);
    attackers |= (Bitboards::KNIGHT_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KNIGHT)]);
    attackers |= (Bitboards::bishop_attacks(sq, occ) & (pieces_bb_[make_piece(by_color, BISHOP)] | pieces_bb_[make_piece(by_color, QUEEN)]));
    attackers |= (Bitboards::rook_attacks(sq, occ) & (pieces_bb_[make_piece(by_color, ROOK)] | pieces_bb_[make_piece(by_color, QUEEN)]));
    attackers |= (Bitboards::KING_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KING)]);
    return attackers;
}

PieceType Board::least_valuable_attacker(Square sq, Color by_color, Bitboard occ, Square& out_from) const {
    if (sq >= 64) {
        out_from = SQ_NONE;
        return NO_PIECE_TYPE;
    }
    Color defender_color = ~by_color;

    // Pawns
    Bitboard pawns = Bitboards::PAWN_ATTACKS[defender_color][sq] & pieces_bb_[make_piece(by_color, PAWN)] & occ;
    if (pawns) {
        out_from = lsb(pawns);
        return PAWN;
    }

    // Knights
    Bitboard knights = Bitboards::KNIGHT_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KNIGHT)] & occ;
    if (knights) {
        out_from = lsb(knights);
        return KNIGHT;
    }

    // Bishops
    Bitboard bishops = Bitboards::bishop_attacks(sq, occ) & pieces_bb_[make_piece(by_color, BISHOP)] & occ;
    if (bishops) {
        out_from = lsb(bishops);
        return BISHOP;
    }

    // Rooks
    Bitboard rooks = Bitboards::rook_attacks(sq, occ) & pieces_bb_[make_piece(by_color, ROOK)] & occ;
    if (rooks) {
        out_from = lsb(rooks);
        return ROOK;
    }

    // Queens
    Bitboard queens = Bitboards::queen_attacks(sq, occ) & pieces_bb_[make_piece(by_color, QUEEN)] & occ;
    if (queens) {
        out_from = lsb(queens);
        return QUEEN;
    }

    // King
    Bitboard king = Bitboards::KING_ATTACKS[sq] & pieces_bb_[make_piece(by_color, KING)] & occ;
    if (king) {
        out_from = lsb(king);
        return KING;
    }

    out_from = SQ_NONE;
    return NO_PIECE_TYPE;
}

bool Board::make_move(Move m) {
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece moving = square_piece_[from];
    Piece captured = square_piece_[to];
    Color us = stm_;
    Color them = ~us;

    if (m.is_en_passant()) {
        captured = make_piece(them, PAWN);
    }

    // Save state
    StateInfo st;
    st.key = key_;
    st.captured = captured;
    st.ep_square = ep_square_;
    st.castling_rights = castling_rights_;
    st.halfmove_clock = halfmove_clock_;
    history_.push_back(st);

    // Update halfmove clock
    if (piece_type(moving) == PAWN || captured != NO_PIECE) {
        halfmove_clock_ = 0;
    } else {
        halfmove_clock_++;
    }

    if (us == BLACK) {
        fullmove_number_++;
    }

    // Remove old EP from hash
    key_ ^= Zobrist::EP[ep_square_];
    ep_square_ = SQ_NONE;

    // Handle moving piece
    key_ ^= Zobrist::PIECES[moving][from];
    pieces_bb_[moving] ^= square_bb(from);
    color_bb_[us] ^= square_bb(from);
    square_piece_[from] = NO_PIECE;

    // Handle capture
    if (captured != NO_PIECE && !m.is_en_passant()) {
        key_ ^= Zobrist::PIECES[captured][to];
        pieces_bb_[captured] ^= square_bb(to);
        color_bb_[them] ^= square_bb(to);
    } else if (m.is_en_passant()) {
        Square cap_sq = make_square(square_file(to), square_rank(from));
        key_ ^= Zobrist::PIECES[captured][cap_sq];
        pieces_bb_[captured] ^= square_bb(cap_sq);
        color_bb_[them] ^= square_bb(cap_sq);
        square_piece_[cap_sq] = NO_PIECE;
    }

    // Handle normal / promotion / castling
    Piece placed_piece = moving;
    if (m.is_promotion()) {
        placed_piece = make_piece(us, m.promotion_type());
    }

    key_ ^= Zobrist::PIECES[placed_piece][to];
    pieces_bb_[placed_piece] |= square_bb(to);
    color_bb_[us] |= square_bb(to);
    square_piece_[to] = placed_piece;

    // Castling rook move
    if (m.is_castling()) {
        Square r_from, r_to;
        if (to == SQ_G1) { r_from = SQ_H1; r_to = SQ_F1; }
        else if (to == SQ_C1) { r_from = SQ_A1; r_to = SQ_D1; }
        else if (to == SQ_G8) { r_from = SQ_H8; r_to = SQ_F8; }
        else { r_from = SQ_A8; r_to = SQ_D8; }

        Piece r_piece = make_piece(us, ROOK);
        key_ ^= Zobrist::PIECES[r_piece][r_from];
        key_ ^= Zobrist::PIECES[r_piece][r_to];
        pieces_bb_[r_piece] ^= (square_bb(r_from) | square_bb(r_to));
        color_bb_[us] ^= (square_bb(r_from) | square_bb(r_to));
        square_piece_[r_from] = NO_PIECE;
        square_piece_[r_to] = r_piece;
    }

    // En passant square update for 2-step pawn push
    if (piece_type(moving) == PAWN) {
        int r_from = square_rank(from);
        int r_to = square_rank(to);
        if (std::abs(r_to - r_from) == 2) {
            ep_square_ = make_square(square_file(from), static_cast<Rank>((r_from + r_to) / 2));
            key_ ^= Zobrist::EP[ep_square_];
        }
    }

    // Update castling rights
    key_ ^= Zobrist::CASTLING[castling_rights_ & 0xF];

    // King move removes all castling for this side
    if (moving == W_KING) castling_rights_ &= ~WHITE_CASTLING;
    if (moving == B_KING) castling_rights_ &= ~BLACK_CASTLING;

    // Rook moves
    if (from == SQ_A1 || to == SQ_A1) castling_rights_ &= ~WHITE_OOO;
    if (from == SQ_H1 || to == SQ_H1) castling_rights_ &= ~WHITE_OO;
    if (from == SQ_A8 || to == SQ_A8) castling_rights_ &= ~BLACK_OOO;
    if (from == SQ_H8 || to == SQ_H8) castling_rights_ &= ~BLACK_OO;

    key_ ^= Zobrist::CASTLING[castling_rights_ & 0xF];

    // Toggle STM
    stm_ = them;
    key_ ^= Zobrist::SIDE;

    // Verify move does not leave our king in check
    if (is_square_attacked(king_square(us), them)) {
        unmake_move(m);
        return false;
    }

    return true;
}

void Board::unmake_move(Move m) {
    stm_ = ~stm_;
    Color us = stm_;
    Color them = ~us;

    assert(!history_.empty());
    const StateInfo& st = history_.back();

    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece moving = square_piece_[to];
    if (m.is_promotion()) {
        moving = make_piece(us, PAWN);
    }

    // Restore to square
    pieces_bb_[square_piece_[to]] ^= square_bb(to);
    color_bb_[us] ^= square_bb(to);
    square_piece_[to] = NO_PIECE;

    // Put moving piece back on from
    pieces_bb_[moving] |= square_bb(from);
    color_bb_[us] |= square_bb(from);
    square_piece_[from] = moving;

    // Restore captured piece
    if (st.captured != NO_PIECE && !m.is_en_passant()) {
        pieces_bb_[st.captured] |= square_bb(to);
        color_bb_[them] |= square_bb(to);
        square_piece_[to] = st.captured;
    } else if (m.is_en_passant()) {
        Square cap_sq = make_square(square_file(to), square_rank(from));
        pieces_bb_[st.captured] |= square_bb(cap_sq);
        color_bb_[them] |= square_bb(cap_sq);
        square_piece_[cap_sq] = st.captured;
    }

    // Castling rook revert
    if (m.is_castling()) {
        Square r_from, r_to;
        if (to == SQ_G1) { r_from = SQ_H1; r_to = SQ_F1; }
        else if (to == SQ_C1) { r_from = SQ_A1; r_to = SQ_D1; }
        else if (to == SQ_G8) { r_from = SQ_H8; r_to = SQ_F8; }
        else { r_from = SQ_A8; r_to = SQ_D8; }

        Piece r_piece = make_piece(us, ROOK);
        pieces_bb_[r_piece] ^= (square_bb(r_from) | square_bb(r_to));
        color_bb_[us] ^= (square_bb(r_from) | square_bb(r_to));
        square_piece_[r_to] = NO_PIECE;
        square_piece_[r_from] = r_piece;
    }

    key_ = st.key;
    ep_square_ = st.ep_square;
    castling_rights_ = st.castling_rights;
    halfmove_clock_ = st.halfmove_clock;
    if (us == BLACK) fullmove_number_--;

    history_.pop_back();
}

void Board::make_null_move() {
    StateInfo st;
    st.key = key_;
    st.captured = NO_PIECE;
    st.ep_square = ep_square_;
    st.castling_rights = castling_rights_;
    st.halfmove_clock = halfmove_clock_;
    history_.push_back(st);

    key_ ^= Zobrist::EP[ep_square_];
    ep_square_ = SQ_NONE;

    stm_ = ~stm_;
    key_ ^= Zobrist::SIDE;
    halfmove_clock_++;
}

void Board::unmake_null_move() {
    stm_ = ~stm_;
    const StateInfo& st = history_.back();
    key_ = st.key;
    ep_square_ = st.ep_square;
    castling_rights_ = st.castling_rights;
    halfmove_clock_ = st.halfmove_clock;
    history_.pop_back();
}

void Board::generate_legal_moves(std::vector<Move>& moves) const {
    moves.clear();
    Color us = stm_;
    Color them = ~us;
    Bitboard occ = occupied_bb();
    Bitboard friendly = color_bb_[us];
    Bitboard enemies = color_bb_[them];
    Bitboard empty = ~occ;

    // Pawns
    Bitboard pawns = pieces_bb_[make_piece(us, PAWN)];
    Rank start_rank = (us == WHITE) ? RANK_2 : RANK_7;
    Rank promo_rank = (us == WHITE) ? RANK_7 : RANK_2;
    int fwd = (us == WHITE) ? 8 : -8;

    while (pawns) {
        Square sq = pop_lsb(pawns);
        Rank r = square_rank(sq);
        Square one_step = static_cast<Square>(sq + fwd);

        // Single push
        if (empty & square_bb(one_step)) {
            if (r == promo_rank) {
                moves.emplace_back(sq, one_step, MOVE_PROMOTION, QUEEN);
                moves.emplace_back(sq, one_step, MOVE_PROMOTION, ROOK);
                moves.emplace_back(sq, one_step, MOVE_PROMOTION, BISHOP);
                moves.emplace_back(sq, one_step, MOVE_PROMOTION, KNIGHT);
            } else {
                moves.emplace_back(sq, one_step, MOVE_NORMAL);
                // Double push
                if (r == start_rank) {
                    Square two_step = static_cast<Square>(one_step + fwd);
                    if (empty & square_bb(two_step)) {
                        moves.emplace_back(sq, two_step, MOVE_NORMAL);
                    }
                }
            }
        }

        // Captures
        Bitboard att = Bitboards::PAWN_ATTACKS[us][sq];
        Bitboard targets = att & enemies;
        while (targets) {
            Square t = pop_lsb(targets);
            if (r == promo_rank) {
                moves.emplace_back(sq, t, MOVE_PROMOTION, QUEEN);
                moves.emplace_back(sq, t, MOVE_PROMOTION, ROOK);
                moves.emplace_back(sq, t, MOVE_PROMOTION, BISHOP);
                moves.emplace_back(sq, t, MOVE_PROMOTION, KNIGHT);
            } else {
                moves.emplace_back(sq, t, MOVE_NORMAL);
            }
        }

        // En passant
        if (ep_square_ != SQ_NONE && (att & square_bb(ep_square_))) {
            moves.emplace_back(sq, ep_square_, MOVE_EN_PASSANT);
        }
    }

    // Knights
    Bitboard knights = pieces_bb_[make_piece(us, KNIGHT)];
    while (knights) {
        Square sq = pop_lsb(knights);
        Bitboard att = Bitboards::KNIGHT_ATTACKS[sq] & ~friendly;
        while (att) {
            Square t = pop_lsb(att);
            moves.emplace_back(sq, t, MOVE_NORMAL);
        }
    }

    // Bishops
    Bitboard bishops = pieces_bb_[make_piece(us, BISHOP)];
    while (bishops) {
        Square sq = pop_lsb(bishops);
        Bitboard att = Bitboards::bishop_attacks(sq, occ) & ~friendly;
        while (att) {
            Square t = pop_lsb(att);
            moves.emplace_back(sq, t, MOVE_NORMAL);
        }
    }

    // Rooks
    Bitboard rooks = pieces_bb_[make_piece(us, ROOK)];
    while (rooks) {
        Square sq = pop_lsb(rooks);
        Bitboard att = Bitboards::rook_attacks(sq, occ) & ~friendly;
        while (att) {
            Square t = pop_lsb(att);
            moves.emplace_back(sq, t, MOVE_NORMAL);
        }
    }

    // Queens
    Bitboard queens = pieces_bb_[make_piece(us, QUEEN)];
    while (queens) {
        Square sq = pop_lsb(queens);
        Bitboard att = Bitboards::queen_attacks(sq, occ) & ~friendly;
        while (att) {
            Square t = pop_lsb(att);
            moves.emplace_back(sq, t, MOVE_NORMAL);
        }
    }

    // King
    Square ksq = king_square(us);
    if (ksq != SQ_NONE) {
        Bitboard att = Bitboards::KING_ATTACKS[ksq] & ~friendly;
        while (att) {
            Square t = pop_lsb(att);
            moves.emplace_back(ksq, t, MOVE_NORMAL);
        }

        // Castling
        if (!in_check()) {
            if (us == WHITE) {
                if ((castling_rights_ & WHITE_OO) && !(occ & (square_bb(SQ_F1) | square_bb(SQ_G1)))) {
                    if (!is_square_attacked(SQ_F1, BLACK) && !is_square_attacked(SQ_G1, BLACK)) {
                        moves.emplace_back(SQ_E1, SQ_G1, MOVE_CASTLING);
                    }
                }
                if ((castling_rights_ & WHITE_OOO) && !(occ & (square_bb(SQ_D1) | square_bb(SQ_C1) | square_bb(SQ_B1)))) {
                    if (!is_square_attacked(SQ_D1, BLACK) && !is_square_attacked(SQ_C1, BLACK)) {
                        moves.emplace_back(SQ_E1, SQ_C1, MOVE_CASTLING);
                    }
                }
            } else {
                if ((castling_rights_ & BLACK_OO) && !(occ & (square_bb(SQ_F8) | square_bb(SQ_G8)))) {
                    if (!is_square_attacked(SQ_F8, WHITE) && !is_square_attacked(SQ_G8, WHITE)) {
                        moves.emplace_back(SQ_E8, SQ_G8, MOVE_CASTLING);
                    }
                }
                if ((castling_rights_ & BLACK_OOO) && !(occ & (square_bb(SQ_D8) | square_bb(SQ_C8) | square_bb(SQ_B8)))) {
                    if (!is_square_attacked(SQ_D8, WHITE) && !is_square_attacked(SQ_C8, WHITE)) {
                        moves.emplace_back(SQ_E8, SQ_C8, MOVE_CASTLING);
                    }
                }
            }
        }
    }

    // Filter legal moves
    Board* non_const_this = const_cast<Board*>(this);
    size_t legal_count = 0;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (non_const_this->make_move(moves[i])) {
            non_const_this->unmake_move(moves[i]);
            moves[legal_count++] = moves[i];
        }
    }
    moves.resize(legal_count);
}

void Board::generate_captures(std::vector<Move>& moves) const {
    std::vector<Move> all_moves;
    generate_legal_moves(all_moves);
    moves.clear();
    for (const auto& m : all_moves) {
        if (square_piece_[m.to_sq()] != NO_PIECE || m.is_en_passant() || m.is_promotion()) {
            moves.push_back(m);
        }
    }
}

bool Board::is_legal_move(Move m) const {
    std::vector<Move> moves;
    generate_legal_moves(moves);
    for (const auto& legal_m : moves) {
        if (legal_m == m) return true;
    }
    return false;
}

Move Board::parse_move_uci(const std::string& uci) const {
    if (uci.length() < 4) return MOVE_NONE;
    Square from = make_square(static_cast<File>(uci[0] - 'a'), static_cast<Rank>(uci[1] - '1'));
    Square to   = make_square(static_cast<File>(uci[2] - 'a'), static_cast<Rank>(uci[3] - '1'));

    PieceType promo = NO_PIECE_TYPE;
    if (uci.length() >= 5) {
        switch (uci[4]) {
            case 'n': promo = KNIGHT; break;
            case 'b': promo = BISHOP; break;
            case 'r': promo = ROOK; break;
            case 'q': promo = QUEEN; break;
            default: break;
        }
    }

    std::vector<Move> moves;
    generate_legal_moves(moves);
    for (const auto& m : moves) {
        if (m.from_sq() == from && m.to_sq() == to) {
            if (m.is_promotion()) {
                if (m.promotion_type() == promo) return m;
            } else {
                return m;
            }
        }
    }
    return MOVE_NONE;
}

bool Board::is_checkmate() const {
    if (!in_check()) return false;
    std::vector<Move> moves;
    generate_legal_moves(moves);
    return moves.empty();
}

bool Board::is_stalemate() const {
    if (in_check()) return false;
    std::vector<Move> moves;
    generate_legal_moves(moves);
    return moves.empty();
}

bool Board::is_draw() const {
    if (halfmove_clock_ >= 100) return true;
    if (is_repetition()) return true;
    // Insufficient material
    if (popcount(occupied_bb()) == 2) return true; // KK
    if (popcount(occupied_bb()) == 3) {
        if (pieces_bb_[W_KNIGHT] || pieces_bb_[B_KNIGHT] || pieces_bb_[W_BISHOP] || pieces_bb_[B_BISHOP]) {
            return true;
        }
    }
    return false;
}

bool Board::is_repetition() const {
    int count = 0;
    for (int i = static_cast<int>(history_.size()) - 2; i >= 0; i -= 2) {
        if (history_[i].key == key_) {
            count++;
            if (count >= 2) return true; // 3-fold repetition
        }
    }
    return false;
}

bool Board::gives_check(Move m) const {
    Color us = stm_;
    Color them = ~us;
    Square ksq = king_square(them);
    if (ksq == SQ_NONE) return false;

    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece moving = square_piece_[from];
    PieceType pt = piece_type(moving);

    Bitboard occ = (occupied_bb() ^ square_bb(from)) | square_bb(to);

    // Direct check from moving piece
    Bitboard attacks = 0;
    PieceType check_pt = m.is_promotion() ? m.promotion_type() : pt;
    switch (check_pt) {
        case KNIGHT: attacks = Bitboards::KNIGHT_ATTACKS[to]; break;
        case BISHOP: attacks = Bitboards::bishop_attacks(to, occ); break;
        case ROOK:   attacks = Bitboards::rook_attacks(to, occ); break;
        case QUEEN:  attacks = Bitboards::queen_attacks(to, occ); break;
        case PAWN:   attacks = Bitboards::PAWN_ATTACKS[us][to]; break;
        default: break;
    }
    if (attacks & square_bb(ksq)) return true;

    // Discovered check
    Bitboard diag_slider = pieces_bb_[make_piece(us, BISHOP)] | pieces_bb_[make_piece(us, QUEEN)];
    if ((diag_slider & ~square_bb(from)) && (Bitboards::bishop_attacks(ksq, occ) & diag_slider & ~square_bb(from))) {
        return true;
    }

    Bitboard orth_slider = pieces_bb_[make_piece(us, ROOK)] | pieces_bb_[make_piece(us, QUEEN)];
    if ((orth_slider & ~square_bb(from)) && (Bitboards::rook_attacks(ksq, occ) & orth_slider & ~square_bb(from))) {
        return true;
    }

    return false;
}

void Board::print() const {
    std::cout << "\n +---+---+---+---+---+---+---+---+\n";
    for (int r = 7; r >= 0; --r) {
        std::cout << " |";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            Piece p = square_piece_[sq];
            const char piece_chars[] = ".PNBRQKpnbrqk";
            std::cout << " " << piece_chars[p] << " |";
        }
        std::cout << " " << (r + 1) << "\n +---+---+---+---+---+---+---+---+\n";
    }
    std::cout << "   a   b   c   d   e   f   g   h\n\n";
    std::cout << "FEN: " << to_fen() << "\n";
    std::cout << "Key: 0x" << std::hex << key_ << std::dec << "\n";
    std::cout << "Side to move: " << (stm_ == WHITE ? "White" : "Black") << "\n\n";
}
