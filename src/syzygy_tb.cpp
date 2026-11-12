#include "syzygy_tb.h"
#include "syzygy/tbprobe.h"
#include <iostream>
#include <mutex>

namespace Syzygy {

static std::mutex tb_mutex;
static bool tb_initialized = false;
static std::string tb_current_path = "";
static int tb_probe_limit = 7;

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

constexpr Value VALUE_TB_WIN = 27000;

static Value score_from_result(unsigned result, int ply) {
    switch (TB_GET_WDL(result)) {
        case TB_LOSS: return -VALUE_TB_WIN + ply;
        case TB_BLESSED_LOSS: return -10;
        case TB_DRAW: return VALUE_DRAW;
        case TB_CURSED_WIN: return 10;
        case TB_WIN: return VALUE_TB_WIN - ply;
        default: return VALUE_DRAW;
    }
}

bool init(const std::string& path) {
    std::lock_guard<std::mutex> lock(tb_mutex);
    if (path.empty() || path == "<empty>") {
        TB_LARGEST = 0;
        tb_initialized = false;
        tb_current_path = "";
        return true;
    }
    if (tb_initialized && path == tb_current_path && TB_LARGEST > 0) {
        return true;
    }
    std::string effective_path = path;
#ifdef _WIN32
    // When running GiraffeGPU.exe under Wine on Linux, translate Unix paths (/run/media/...) to Z:\...
    if (!path.empty() && path[0] == '/') {
        std::string wine_path = "Z:" + path;
        for (char& c : wine_path) {
            if (c == '/') c = '\\';
        }
        effective_path = path + ";" + wine_path;
    }
#endif
    tb_initialized = tb_init(effective_path.c_str());
    if (tb_initialized) {
        tb_current_path = path;
    }
    return tb_initialized;
}

int max_pieces() {
    return static_cast<int>(TB_LARGEST);
}

void set_probe_limit(int limit) {
    tb_probe_limit = std::max(0, limit);
}

int probe_limit() {
    return tb_probe_limit;
}

bool probe_wdl(const Board& board, Value& out_score, int ply) {
    if (TB_LARGEST == 0) return false;
    if (board.castling_rights() != 0) return false;

    int total_pieces = popcount(board.occupied_bb());
    if (total_pieces > static_cast<int>(TB_LARGEST) || total_pieces > tb_probe_limit) return false;

    std::lock_guard<std::mutex> lock(tb_mutex);

    uint64_t white = board.color_bb(WHITE);
    uint64_t black = board.color_bb(BLACK);
    uint64_t kings   = board.piece_bb(W_KING)   | board.piece_bb(B_KING);
    uint64_t queens  = board.piece_bb(W_QUEEN)  | board.piece_bb(B_QUEEN);
    uint64_t rooks   = board.piece_bb(W_ROOK)   | board.piece_bb(B_ROOK);
    uint64_t bishops = board.piece_bb(W_BISHOP) | board.piece_bb(B_BISHOP);
    uint64_t knights = board.piece_bb(W_KNIGHT) | board.piece_bb(B_KNIGHT);
    uint64_t pawns   = board.piece_bb(W_PAWN)   | board.piece_bb(B_PAWN);

    unsigned ep = (board.ep_square() != SQ_NONE) ? static_cast<unsigned>(board.ep_square()) : 0;
    bool turn = (board.side_to_move() == WHITE);

    // Note: Fathom's tb_probe_wdl wrapper rejects rule50 != 0 immediately; pass 0 for WDL lookup!
    unsigned res = tb_probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns,
                                0, 0, ep, turn);

    if (res == TB_RESULT_FAILED) return false;

    switch (res) {
        case TB_LOSS:
            out_score = -VALUE_TB_WIN + ply;
            return true;
        case TB_BLESSED_LOSS:
            out_score = (board.halfmove_clock() > 0) ? VALUE_DRAW : Value(-10);
            return true;
        case TB_DRAW:
            out_score = VALUE_DRAW;
            return true;
        case TB_CURSED_WIN:
            out_score = (board.halfmove_clock() > 0) ? VALUE_DRAW : Value(10);
            return true;
        case TB_WIN:
            out_score = VALUE_TB_WIN - ply;
            return true;
        default:
            return false;
    }
}

bool filter_root_moves(const Board& board, MoveList& root_moves, Value& out_score) {
    if (TB_LARGEST == 0 || board.castling_rights() != 0) return false;
    int total_pieces = popcount(board.occupied_bb());
    if (total_pieces > static_cast<int>(TB_LARGEST) || total_pieces > tb_probe_limit) return false;

    std::lock_guard<std::mutex> lock(tb_mutex);

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

    if (res == TB_RESULT_FAILED || res == 0 || res == TB_RESULT_CHECKMATE || res == TB_RESULT_STALEMATE) {
        return false;
    }

    int best_rank = -1;
    unsigned best_dtz = 0xFFF;
    unsigned best_res = TB_RESULT_FAILED;

    for (unsigned i = 0; i < TB_MAX_MOVES && results[i] != TB_RESULT_FAILED; ++i) {
        unsigned candidate = results[i];
        int rank = wdl_rank(TB_GET_WDL(candidate));
        unsigned dtz = TB_GET_DTZ(candidate);
        if (best_res == TB_RESULT_FAILED || rank > best_rank ||
            (rank == best_rank && ((best_rank >= 3 && dtz < best_dtz) || (best_rank <= 1 && dtz > best_dtz)))) {
            best_res = candidate;
            best_rank = rank;
            best_dtz = dtz;
        }
    }

    if (best_res == TB_RESULT_FAILED) return false;

    MoveList filtered;
    // Keep moves that preserve the optimal WDL rank and are within 1 DTZ step of the minimum DTZ
    // (Syzygy DTZ values can be off by 1 ply; allowing +1 lets search break ties for the most natural move)
    for (unsigned i = 0; i < TB_MAX_MOVES && results[i] != TB_RESULT_FAILED; ++i) {
        unsigned candidate = results[i];
        int rank = wdl_rank(TB_GET_WDL(candidate));
        unsigned dtz = TB_GET_DTZ(candidate);
        bool keep = false;
        if (rank == best_rank) {
            if (best_rank >= 3) {
                keep = (dtz == best_dtz);
            } else if (best_rank == 2) {
                keep = true;
            } else {
                keep = (dtz == best_dtz);
            }
        }
        if (keep) {
            Move m = move_from_result(candidate);
            for (size_t k = 0; k < root_moves.size(); ++k) {
                if (root_moves[k] == m) {
                    filtered.push_back(root_moves[k]);
                    break;
                }
            }
        }
    }

    if (!filtered.empty()) {
        // Filter out redundant underpromotions when a Queen promotion from/to the same square is also optimal
        MoveList deduped;
        for (size_t i = 0; i < filtered.size(); ++i) {
            Move m = filtered[i];
            if (m.is_promotion() && m.promotion_type() != QUEEN) {
                bool has_queen_promo = false;
                for (size_t j = 0; j < filtered.size(); ++j) {
                    if (filtered[j].is_promotion() && filtered[j].promotion_type() == QUEEN &&
                        filtered[j].from_sq() == m.from_sq() && filtered[j].to_sq() == m.to_sq()) {
                        has_queen_promo = true;
                        break;
                    }
                }
                if (has_queen_promo) continue;
            }
            deduped.push_back(m);
        }
        root_moves = deduped.empty() ? filtered : deduped;
        out_score = score_from_result(best_res, 0);
        return true;
    }
    return false;
}

bool probe_root(const Board& board, Move& out_move, Value& out_score) {
    std::lock_guard<std::mutex> lock(tb_mutex);
    if (TB_LARGEST == 0) return false;
    if (board.castling_rights() != 0) return false;

    int total_pieces = popcount(board.occupied_bb());
    if (total_pieces > static_cast<int>(TB_LARGEST) || total_pieces > tb_probe_limit) return false;

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
