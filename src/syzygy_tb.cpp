#include "syzygy_tb.h"
#include "syzygy/tbprobe.h"
#include <iostream>
#include <mutex>

namespace Syzygy {

static std::mutex tb_mutex;
static bool tb_initialized = false;
static std::string tb_current_path = "";

static Move move_from_result(unsigned result) {
    unsigned promotes = TB_GET_PROMOTES(result);
    MoveType type = MOVE_NORMAL;
    PieceType promo = KNIGHT;
    if (promotes == TB_PROMOTES_QUEEN) { type = MOVE_PROMOTION; promo = QUEEN; }
    else if (promotes == TB_PROMOTES_ROOK) { type = MOVE_PROMOTION; promo = ROOK; }
    else if (promotes == TB_PROMOTES_BISHOP) { type = MOVE_PROMOTION; promo = BISHOP; }
    else if (promotes == TB_PROMOTES_KNIGHT) { type = MOVE_PROMOTION; promo = KNIGHT; }
    else if (TB_GET_EP(result)) type = MOVE_EN_PASSANT;
    return Move(static_cast<Square>(TB_GET_FROM(result)),
                static_cast<Square>(TB_GET_TO(result)), type, promo);
}

static int wdl_rank(unsigned wdl) {
    switch (wdl) {
        case TB_WIN: return 4;
        case TB_CURSED_WIN: return 3;
        case TB_DRAW: return 2;
        case TB_BLESSED_LOSS: return 1;
        default: return 0;
    }
}

static Value score_from_result(unsigned result, int ply) {
    switch (TB_GET_WDL(result)) {
        case TB_LOSS: return -VALUE_MATE + 100 + ply;
        case TB_BLESSED_LOSS: return -10;
        case TB_DRAW: return VALUE_DRAW;
        case TB_CURSED_WIN: return 10;
        case TB_WIN: return VALUE_MATE - 100 - ply;
        default: return VALUE_DRAW;
    }
}

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
    // The bundled Fathom probe keeps internal caches that are not guaranteed
    // to be thread-safe; search helpers may call this concurrently.
    std::lock_guard<std::mutex> lock(tb_mutex);
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
    std::lock_guard<std::mutex> lock(tb_mutex);
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

    // Do not blindly use results[0]. The root API returns one result per
    // legal move and the first entry is not guaranteed to be best.
    unsigned best_res = TB_RESULT_FAILED;
    int best_rank = -1;
    unsigned best_dtz = 0xFFF;
    for (unsigned i = 0; i < TB_MAX_MOVES && results[i] != TB_RESULT_FAILED; ++i) {
        unsigned candidate = results[i];
        int rank = wdl_rank(TB_GET_WDL(candidate));
        unsigned dtz = TB_GET_DTZ(candidate);
        if (best_res == TB_RESULT_FAILED || rank > best_rank ||
            (rank == best_rank && dtz < best_dtz)) {
            best_res = candidate;
            best_rank = rank;
            best_dtz = dtz;
        }
    }

    if (best_res == TB_RESULT_FAILED) return false;
    out_move = move_from_result(best_res);
    out_score = score_from_result(best_res, 0);

    return true;
}

} // namespace Syzygy
