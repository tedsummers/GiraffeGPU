#include "search.h"
#include "see.h"
#include <iostream>
#include <algorithm>
#include <cstring>

Search g_search;

Search::Search()
    : stop_flag_(false)
    , searching_(false)
    , nodes_(0)
    , num_threads_(2) // Default to 2 threads for dual-GPU acceleration
    , allocated_time_ms_(-1)
{
    main_worker_.id = 0;
    main_worker_.seldepth = 0;
    std::memset(main_worker_.killer_moves, 0, sizeof(main_worker_.killer_moves));
    std::memset(main_worker_.history, 0, sizeof(main_worker_.history));
    std::memset(main_worker_.pv_table, 0, sizeof(main_worker_.pv_table));
    std::memset(main_worker_.pv_length, 0, sizeof(main_worker_.pv_length));
}

Search::~Search() {
    stop();
}

void Search::set_threads(int num_threads) {
    num_threads_ = std::max(1, std::min(num_threads, 64));
}

void Search::stop() {
    stop_flag_.store(true);
    for (auto& t : helper_threads_) {
        if (t.joinable()) t.join();
    }
    helper_threads_.clear();
}

bool Search::check_time() {
    if (stop_flag_.load()) return true;

    if ((nodes_ & 2047) == 0 && allocated_time_ms_ > 0) {
        auto now = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        if (elapsed >= allocated_time_ms_) {
            stop_flag_.store(true);
            return true;
        }
    }
    return false;
}

void Search::order_moves(const Board& board, const SearchWorker& w, std::vector<Move>& moves, Move tt_move, int ply) {
    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(moves.size());

    for (const auto& m : moves) {
        int score = 0;
        if (m == tt_move) {
            score = 200000;
        } else if (board.piece_at(m.to_sq()) != NO_PIECE || m.is_en_passant() || m.is_promotion()) {
            Piece victim = board.piece_at(m.to_sq());
            Piece attacker = board.piece_at(m.from_sq());
            PieceType v_pt = (victim != NO_PIECE) ? piece_type(victim) : PAWN;
            PieceType a_pt = piece_type(attacker);

            Value see_val = SEE::evaluate(board, m);
            if (see_val >= 0) {
                score = 100000 + (SEE::PIECE_VALUES[v_pt] * 10) - SEE::PIECE_VALUES[a_pt] + see_val;
            } else {
                score = -50000 + see_val;
            }
        } else {
            if (ply < 128) {
                if (m == w.killer_moves[ply][0]) {
                    score = 80000;
                } else if (m == w.killer_moves[ply][1]) {
                    score = 70000;
                } else {
                    Piece p = board.piece_at(m.from_sq());
                    score = w.history[p][m.to_sq()];
                }
            }
        }
        scored_moves.emplace_back(score, m);
    }

    std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    for (size_t i = 0; i < moves.size(); ++i) {
        moves[i] = scored_moves[i].second;
    }
}

Value Search::qsearch(Board& board, SearchWorker& w, int ply, Value alpha, Value beta) {
    if (check_time()) return VALUE_ZERO;

    nodes_++;
    if (ply > w.seldepth) w.seldepth = ply;

    if (board.is_draw()) return VALUE_DRAW;

    bool in_check = board.in_check();
    Value stand_pat = -VALUE_INFINITE;

    if (!in_check) {
        // Query assigned GPU for this worker
        stand_pat = Neural::g_neural_gpu.evaluate(board, w.id);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    std::vector<Move> moves;
    if (in_check) {
        board.generate_legal_moves(moves);
        if (moves.empty()) return -VALUE_MATE + ply;
    } else {
        board.generate_captures(moves);
    }

    order_moves(board, w, moves, MOVE_NONE, ply);

    for (const auto& m : moves) {
        if (!in_check && !m.is_promotion() && SEE::evaluate(board, m) < 0) {
            continue;
        }

        if (!board.make_move(m)) continue;
        Value score = -qsearch(board, w, ply + 1, -beta, -alpha);
        board.unmake_move(m);

        if (stop_flag_.load()) return VALUE_ZERO;

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

Value Search::pvs(Board& board, SearchWorker& w, int depth, int ply, Value alpha, Value beta, bool cut_node) {
    if (check_time()) return VALUE_ZERO;

    w.pv_length[ply] = ply;

    if (ply >= 127) return Neural::g_neural_gpu.evaluate(board, w.id);

    if (ply > 0 && board.is_draw()) return VALUE_DRAW;

    alpha = std::max(alpha, static_cast<Value>(-VALUE_MATE + ply));
    beta  = std::min(beta, static_cast<Value>(VALUE_MATE - ply - 1));
    if (alpha >= beta) return alpha;

    bool in_check = board.in_check();
    if (in_check) depth++;

    if (depth <= 0) {
        return qsearch(board, w, ply, alpha, beta);
    }

    nodes_++;

    TTEntry tt_entry;
    Move tt_move = MOVE_NONE;
    if (g_tt.probe(board.key(), tt_entry)) {
        tt_move = Move(tt_entry.move_data);
        if (tt_entry.depth >= depth && ply > 0) {
            Value tt_score = g_tt.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return tt_score;
            if (tt_entry.flag == TT_LOWERBOUND && tt_score >= beta) return beta;
            if (tt_entry.flag == TT_UPPERBOUND && tt_score <= alpha) return alpha;
        }
    }

    if (!in_check && depth >= 3 && ply > 0 && (board.piece_count(W_QUEEN) + board.piece_count(B_QUEEN) + board.piece_count(W_ROOK) + board.piece_count(B_ROOK) > 0)) {
        board.make_null_move();
        int R = (depth >= 6) ? 3 : 2;
        Value null_score = -pvs(board, w, depth - 1 - R, ply + 1, -beta, -beta + 1, !cut_node);
        board.unmake_null_move();

        if (null_score >= beta && std::abs(null_score) < VALUE_MATE_THRESHOLD) {
            return beta;
        }
    }

    std::vector<Move> moves;
    board.generate_legal_moves(moves);

    if (moves.empty()) {
        if (in_check) return -VALUE_MATE + ply;
        return VALUE_DRAW;
    }

    order_moves(board, w, moves, tt_move, ply);

    Move best_move = MOVE_NONE;
    TTFlag flag = TT_UPPERBOUND;
    Value best_score = -VALUE_INFINITE;
    int moves_searched = 0;

    for (const auto& m : moves) {
        if (!board.make_move(m)) continue;
        moves_searched++;

        Value score;
        if (moves_searched == 1) {
            score = -pvs(board, w, depth - 1, ply + 1, -beta, -alpha, false);
        } else {
            int reduction = 0;
            if (depth >= 3 && moves_searched > 4 && !in_check && !m.is_promotion() && board.piece_at(m.to_sq()) == NO_PIECE) {
                reduction = 1;
                if (moves_searched > 8) reduction = 2;
            }

            score = -pvs(board, w, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, true);

            if (score > alpha && reduction > 0) {
                score = -pvs(board, w, depth - 1, ply + 1, -alpha - 1, -alpha, true);
            }
            if (score > alpha && score < beta) {
                score = -pvs(board, w, depth - 1, ply + 1, -beta, -alpha, false);
            }
        }

        board.unmake_move(m);

        if (stop_flag_.load()) return VALUE_ZERO;

        if (score > best_score) {
            best_score = score;
            best_move = m;

            if (score > alpha) {
                alpha = score;
                flag = TT_EXACT;

                w.pv_table[ply][ply] = m;
                for (int next_ply = ply + 1; next_ply < w.pv_length[ply + 1]; ++next_ply) {
                    w.pv_table[ply][next_ply] = w.pv_table[ply + 1][next_ply];
                }
                w.pv_length[ply] = w.pv_length[ply + 1];

                if (score >= beta) {
                    flag = TT_LOWERBOUND;
                    if (board.piece_at(m.to_sq()) == NO_PIECE && !m.is_promotion()) {
                        if (ply < 128) {
                            w.killer_moves[ply][1] = w.killer_moves[ply][0];
                            w.killer_moves[ply][0] = m;
                            Piece p = board.piece_at(m.from_sq());
                            w.history[p][m.to_sq()] += depth * depth;
                        }
                    }
                    break;
                }
            }
        }
    }

    if (!stop_flag_.load()) {
        g_tt.store(board.key(), best_move, best_score, depth, flag, ply);
    }

    return best_score;
}

void Search::worker_search(Board board, int worker_id, int max_depth) {
    SearchWorker w;
    w.id = worker_id;
    w.seldepth = 0;
    std::memset(w.killer_moves, 0, sizeof(w.killer_moves));
    std::memset(w.history, 0, sizeof(w.history));
    std::memset(w.pv_table, 0, sizeof(w.pv_table));
    std::memset(w.pv_length, 0, sizeof(w.pv_length));

    // Helper threads search odd / offset depths (Lazy SMP)
    for (int depth = 1 + (worker_id % 2); depth <= max_depth && !stop_flag_.load(); ++depth) {
        pvs(board, w, depth, 0, -VALUE_INFINITE, VALUE_INFINITE, false);
    }
}

void Search::start(Board& board, const SearchLimits& limits) {
    limits_ = limits;
    stop_flag_.store(false);
    searching_.store(true);
    nodes_.store(0);
    start_time_ = std::chrono::steady_clock::now();

    allocated_time_ms_ = -1;
    if (limits_.movetime > 0) {
        allocated_time_ms_ = limits_.movetime - 10;
    } else {
        int64_t my_time = (board.side_to_move() == WHITE) ? limits_.wtime : limits_.btime;
        int64_t my_inc  = (board.side_to_move() == WHITE) ? limits_.winc  : limits_.binc;
        if (my_time > 0) {
            int moves_left = std::max(10, limits_.movestogo);
            allocated_time_ms_ = (my_time / moves_left) + (my_inc * 3 / 4);
            allocated_time_ms_ = std::min(allocated_time_ms_, my_time - 50);
            if (allocated_time_ms_ < 10) allocated_time_ms_ = 10;
        }
    }

    g_tt.new_search();
    main_worker_.id = 0;
    main_worker_.seldepth = 0;
    std::memset(main_worker_.killer_moves, 0, sizeof(main_worker_.killer_moves));
    std::memset(main_worker_.history, 0, sizeof(main_worker_.history));

    int max_depth = std::min(limits_.depth, 64);

    // Launch helper threads across available GPUs!
    helper_threads_.clear();
    for (int i = 1; i < num_threads_; ++i) {
        helper_threads_.emplace_back(&Search::worker_search, this, board, i, max_depth);
    }

    search_root(board);

    // Stop and join helper threads
    stop_flag_.store(true);
    for (auto& t : helper_threads_) {
        if (t.joinable()) t.join();
    }
    helper_threads_.clear();

    searching_.store(false);
}

void Search::search_root(Board& board) {
    Move best_move = MOVE_NONE;
    Value best_score = -VALUE_INFINITE;

    std::vector<Move> root_moves;
    board.generate_legal_moves(root_moves);
    if (root_moves.empty()) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    best_move = root_moves[0];

    int max_depth = std::min(limits_.depth, 64);

    for (int depth = 1; depth <= max_depth; ++depth) {
        Value alpha = -VALUE_INFINITE;
        Value beta  = VALUE_INFINITE;

        if (depth >= 4) {
            alpha = std::max(-VALUE_INFINITE, best_score - 50);
            beta  = std::min(VALUE_INFINITE, best_score + 50);
        }

        Value score = pvs(board, main_worker_, depth, 0, alpha, beta, false);

        if (score <= alpha || score >= beta) {
            score = pvs(board, main_worker_, depth, 0, -VALUE_INFINITE, VALUE_INFINITE, false);
        }

        if (stop_flag_.load()) break;

        best_score = score;
        if (main_worker_.pv_length[0] > 0) {
            best_move = main_worker_.pv_table[0][0];
        }

        auto now = std::chrono::steady_clock::now();
        int64_t elapsed_ms = std::max(int64_t(1), std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
        uint64_t current_nodes = nodes_.load();
        uint64_t nps = (current_nodes * 1000) / elapsed_ms;

        std::cout << "info depth " << depth
                  << " seldepth " << main_worker_.seldepth
                  << " score ";
        if (std::abs(score) >= VALUE_MATE_THRESHOLD) {
            int mate_in = (VALUE_MATE - std::abs(score) + 1) / 2;
            if (score < 0) mate_in = -mate_in;
            std::cout << "mate " << mate_in;
        } else {
            std::cout << "cp " << score;
        }
        std::cout << " nodes " << current_nodes
                  << " nps " << nps
                  << " time " << elapsed_ms
                  << " hashfull " << g_tt.hashfull()
                  << " pv";

        for (int i = 0; i < main_worker_.pv_length[0]; ++i) {
            std::cout << " " << main_worker_.pv_table[0][i].to_uci();
        }
        std::cout << std::endl;

        if (limits_.nodes > 0 && current_nodes >= static_cast<uint64_t>(limits_.nodes)) break;
        if (allocated_time_ms_ > 0 && elapsed_ms * 2 >= allocated_time_ms_) break;
        if (std::abs(score) >= VALUE_MATE_THRESHOLD) break;
    }

    std::cout << "bestmove " << best_move.to_uci() << std::endl;
}
