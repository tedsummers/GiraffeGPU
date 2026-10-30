#include "syzygy_tb.h"
#include "syzygy/tbprobe.h"
#include <iostream>
#include <mutex>

namespace Syzygy {

static std::mutex tb_mutex;
static bool tb_initialized = false;
static std::string tb_current_path = "";

bool init(const std::string& path) {
    std::lock_guard<std::mutex> lock(tb_mutex);
    if (path.empty()) {
        TB_LARGEST = 0;
        tb_initialized = false;
        tb_current_path = "";
        return true;
    }
    tb_initialized = tb_init(path.c_str());
    if (tb_initialized) {
        tb_current_path = path;
    }
    return tb_initialized;
}

int max_pieces() {
    return static_cast<int>(TB_LARGEST);
}

bool probe_wdl(const Board& board, Value& out_score, int ply) {
    if (TB_LARGEST == 0) return false;

    // Tablebases only apply when castling rights are gone
    if (board.castling_rights() != 0) return false;

    int total_pieces = popcount(board.occupied_bb());
    if (total_pieces > static_cast<int>(TB_LARGEST)) return false;

    uint64_t white = board.color_bb(WHITE);
    uint64_t black = board.color_bb(BLACK);
    uint64_t kings   = board.piece_bb(W_KING)   | board.piece_bb(B_KING);
    uint64_t queens  = board.piece_bb(W_QUEEN)  | board.piece_bb(B_QUEEN);
    uint64_t rooks   = board.piece_bb(W_ROOK)   | board.piece_bb(B_ROOK);
    uint64_t bishops = board.piece_bb(W_BISHOP) | board.piece_bb(B_BISHOP);
    uint64_t knights = board.piece_bb(W_KNIGHT) | board.piece_bb(B_KNIGHT);
    uint64_t pawns   = board.piece_bb(W_PAWN)   | board.piece_bb(B_PAWN);

    unsigned rule50 = board.halfmove_clock();
    unsigned ep = (board.ep_square() != SQ_NONE) ? static_cast<unsigned>(board.ep_square()) : 0;
    bool turn = (board.side_to_move() == WHITE);

    unsigned res = tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                                rule50, 0, ep, turn);

    if (res == TB_RESULT_FAILED) return false;

    switch (res) {
        case TB_LOSS:
            out_score = -VALUE_MATE + 100 + ply;
            return true;
        case TB_BLESSED_LOSS:
            out_score = -10; // 50-move draw in practice
            return true;
        case TB_DRAW:
            out_score = VALUE_DRAW;
            return true;
        case TB_CURSED_WIN:
            out_score = 10;
            return true;
        case TB_WIN:
            out_score = VALUE_MATE - 100 - ply;
            return true;
        default:
            return false;
    }
}

bool probe_root(const Board& board, Move& out_move, Value& out_score) {
    if (TB_LARGEST == 0) return false;
    if (board.castling_rights() != 0) return false;

    int total_pieces = popcount(board.occupied_bb());
    if (total_pieces > static_cast<int>(TB_LARGEST)) return false;

    uint64_t white = board.color_bb(WHITE);
    uint64_t black = board.color_bb(BLACK);
    uint64_t kings   = board.piece_bb(W_KING)   | board.piece_bb(B_KING);
    uint64_t queens  = board.piece_bb(W_QUEEN)  | board.piece_bb(B_QUEEN);
    uint64_t rooks   = board.piece_bb(W_ROOK)   | board.piece_bb(B_ROOK);
    uint64_t bishops = board.piece_bb(W_BISHOP) | board.piece_bb(B_BISHOP);
    uint64_t knights = board.piece_bb(W_KNIGHT) | board.piece_bb(B_KNIGHT);
    uint64_t pawns   = board.piece_bb(W_PAWN)   | board.piece_bb(B_PAWN);

    unsigned rule50 = board.halfmove_clock();
    unsigned ep = (board.ep_square() != SQ_NONE) ? static_cast<unsigned>(board.ep_square()) : 0;
    bool turn = (board.side_to_move() == WHITE);

    unsigned results[TB_MAX_MOVES];
    unsigned res = tb_probe_root(white, black, kings, queens, rooks, bishops, knights, pawns,
                                 rule50, 0, ep, turn, results);

    if (res == TB_RESULT_FAILED || res == 0) return false;

    unsigned best_res = results[0];
    unsigned from = TB_GET_FROM(best_res);
    unsigned to   = TB_GET_TO(best_res);
    unsigned promotes = TB_GET_PROMOTES(best_res);

    MoveType mt = MOVE_NORMAL;
    PieceType promo = KNIGHT;
    if (promotes == TB_PROMOTES_QUEEN) { mt = MOVE_PROMOTION; promo = QUEEN; }
    else if (promotes == TB_PROMOTES_ROOK) { mt = MOVE_PROMOTION; promo = ROOK; }
    else if (promotes == TB_PROMOTES_BISHOP) { mt = MOVE_PROMOTION; promo = BISHOP; }
    else if (promotes == TB_PROMOTES_KNIGHT) { mt = MOVE_PROMOTION; promo = KNIGHT; }
    else if (TB_GET_EP(best_res)) mt = MOVE_EN_PASSANT;

    out_move = Move(static_cast<Square>(from), static_cast<Square>(to), mt, promo);

    unsigned wdl = TB_GET_WDL(best_res);
    int dtz = TB_GET_DTZ(best_res);

    switch (wdl) {
        case TB_LOSS:
            out_score = -VALUE_MATE + 100 + dtz;
            break;
        case TB_BLESSED_LOSS:
            out_score = -10;
            break;
        case TB_DRAW:
            out_score = VALUE_DRAW;
            break;
        case TB_CURSED_WIN:
            out_score = 10;
            break;
        case TB_WIN:
            out_score = VALUE_MATE - 100 - dtz;
            break;
        default:
            out_score = VALUE_DRAW;
            break;
    }

    return true;
}

} // namespace Syzygy
