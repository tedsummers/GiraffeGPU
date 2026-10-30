#include "search.h"
#include "see.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cmath>

Search g_search;

// Precomputed Late Move Reductions (LMR) table based on depth and move index
static int LMR_TABLE[64][64];

static void init_lmr_table() {
    for (int d = 1; d < 64; ++d) {
        for (int m = 1; m < 64; ++m) {
            // Logarithmic reduction formula: base + ln(depth) * ln(moves) / 2.0
            double r = 0.5 + std::log(d) * std::log(m) / 2.0;
            LMR_TABLE[d][m] = static_cast<int>(r);
        }
    }
}

Search::Search()
    : stop_flag_(false)
    , searching_(false)
    , pondering_(false)
    , nodes_(0)
    , ponder_enabled_(true)
    , num_threads_(2)
    , target_time_ms_(-1)
    , hard_limit_ms_(-1)
{
    init_lmr_table();
    main_worker_.id = 0;
    main_worker_.clear();
}

Search::~Search() {
    stop();
    for (auto& t : helper_threads_) {
        if (t.joinable()) t.join();
    }
    helper_threads_.clear();
}

void Search::set_threads(int num_threads) {
    num_threads_ = std::max(1, std::min(num_threads, 64));
}

void Search::stop() {
    stop_flag_.store(true, std::memory_order_relaxed);
    pondering_.store(false, std::memory_order_relaxed);
}

void Search::ponderhit() {
    pondering_.store(false);
    start_time_ = std::chrono::steady_clock::now();
}

bool Search::check_time() {
    if (stop_flag_.load(std::memory_order_relaxed)) return true;
    if (pondering_.load(std::memory_order_relaxed)) return false;

    if ((nodes_.load(std::memory_order_relaxed) & 1023) == 0 && hard_limit_ms_ > 0) {
        auto now = std::chrono::steady_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        if (elapsed >= hard_limit_ms_) {
            stop_flag_.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

// History gravity update: updates score towards +/- max limit without overflowing
static inline void update_history_score(int& score, int bonus) {
    constexpr int MAX_HISTORY = 16384;
    score += bonus - (score * std::abs(bonus)) / MAX_HISTORY;
}

void Search::update_quiet_history(SearchWorker& w, const Board& board, Move best_quiet,
                                  const std::vector<Move>& quiet_moves, int depth, int ply)
{
    int bonus = std::min(depth * depth * 32, 2000);

    // Reward best quiet move
    Piece best_p = board.piece_at(best_quiet.from_sq());
    update_history_score(w.history[best_p][best_quiet.to_sq()], bonus);

    // Countermove update
    if (ply > 0 && !w.move_stack[ply - 1].is_null()) {
        Move prev_m = w.move_stack[ply - 1];
        // Note: prev piece is retrieved at prev_m.to_sq()
        Piece prev_p = board.piece_at(prev_m.to_sq());
        if (prev_p != NO_PIECE) {
            w.countermove[prev_p][prev_m.to_sq()] = best_quiet;
            update_history_score(w.conthist[prev_p][prev_m.to_sq()][best_quiet.to_sq()], bonus);
        }
    }

    // Penalize other tried quiet moves with negative gravity
    int malus = -bonus / 2;
    for (const auto& m : quiet_moves) {
        if (m == best_quiet) continue;
        Piece p = board.piece_at(m.from_sq());
        update_history_score(w.history[p][m.to_sq()], malus);

        if (ply > 0 && !w.move_stack[ply - 1].is_null()) {
            Move prev_m = w.move_stack[ply - 1];
            Piece prev_p = board.piece_at(prev_m.to_sq());
            if (prev_p != NO_PIECE) {
                update_history_score(w.conthist[prev_p][prev_m.to_sq()][m.to_sq()], malus);
            }
        }
    }
}

void Search::order_moves(const Board& board, const SearchWorker& w, std::vector<Move>& moves, Move tt_move, int ply) {
    std::vector<std::pair<int, Move>> scored_moves;
    scored_moves.reserve(moves.size());

    Move prev_m = (ply > 0) ? w.move_stack[ply - 1] : MOVE_NONE;
    Piece prev_p = (!prev_m.is_null()) ? board.piece_at(prev_m.to_sq()) : NO_PIECE;
    Move cm = (prev_p != NO_PIECE) ? w.countermove[prev_p][prev_m.to_sq()] : MOVE_NONE;

    for (const auto& m : moves) {
        int score = 0;
        if (m == tt_move) {
            score = 300000;
        } else if (board.piece_at(m.to_sq()) != NO_PIECE || m.is_en_passant() || m.is_promotion()) {
            // Captures and Promotions: MVV-LVA + SEE
            Piece victim = board.piece_at(m.to_sq());
            Piece attacker = board.piece_at(m.from_sq());
            PieceType v_pt = (victim != NO_PIECE) ? piece_type(victim) : PAWN;
            PieceType a_pt = piece_type(attacker);

            if (m.is_promotion()) {
                score = 150000 + SEE::PIECE_VALUES[m.promotion_type()];
            } else {
                Value see_val = SEE::evaluate(board, m);
                if (see_val >= 0) {
                    score = 100000 + (SEE::PIECE_VALUES[v_pt] * 10) - SEE::PIECE_VALUES[a_pt] + see_val;
                } else {
                    score = -50000 + see_val;
                }
            }
        } else {
            // Quiet moves: Killers -> Countermove -> Continuation History -> Main History
            if (ply < 128 && m == w.killer_moves[ply][0]) {
                score = 90000;
            } else if (ply < 128 && m == w.killer_moves[ply][1]) {
                score = 80000;
            } else if (m == cm) {
                score = 75000;
            } else {
                Piece p = board.piece_at(m.from_sq());
                int h_score = w.history[p][m.to_sq()];
                int ch_score = (prev_p != NO_PIECE) ? w.conthist[prev_p][prev_m.to_sq()][m.to_sq()] : 0;
                score = h_score + ch_score * 2;
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
        stand_pat = Neural::g_neural_gpu.evaluate(board, w.id);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    std::vector<Move> moves;
    if (in_check) {
        board.generate_legal_moves(moves);
        if (moves.empty()) return -VALUE_MATE + ply;
    } else {
        // Generate captures and promotions
        board.generate_captures(moves);

        // Quiescence improvement: Include quiet checks when within 2 plies of root
        if (ply <= 2) {
            std::vector<Move> all_moves;
            board.generate_legal_moves(all_moves);
            for (const auto& m : all_moves) {
                if (board.piece_at(m.to_sq()) == NO_PIECE && !m.is_promotion()) {
                    if (board.gives_check(m)) {
                        moves.push_back(m);
                    }
                }
            }
        }
    }

    order_moves(board, w, moves, MOVE_NONE, ply);

    for (const auto& m : moves) {
        if (!in_check && !m.is_promotion() && SEE::evaluate(board, m) < 0) {
            continue;
        }

        w.move_stack[ply] = m;
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

    // Check extensions
    bool in_check = board.in_check();
    if (in_check) depth++;

    if (depth <= 0) {
        return qsearch(board, w, ply, alpha, beta);
    }

    nodes_++;

    // 1. Syzygy Tablebase Probe
    Value tb_score;
    if (ply > 0 && Syzygy::probe_wdl(board, tb_score, ply)) {
        if (tb_score >= beta) return beta;
        if (tb_score <= alpha) return alpha;
        return tb_score;
    }

    // 2. Transposition Table Probe
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

    // 3. Null Move Pruning
    if (!in_check && depth >= 3 && ply > 0 &&
        (board.piece_count(W_QUEEN) + board.piece_count(B_QUEEN) +
         board.piece_count(W_ROOK)  + board.piece_count(B_ROOK) > 0))
    {
        board.make_null_move();
        int R = 3 + depth / 4;
        Value null_score = -pvs(board, w, depth - 1 - R, ply + 1, -beta, -beta + 1, !cut_node);
        board.unmake_null_move();

        if (null_score >= beta && std::abs(null_score) < VALUE_MATE_THRESHOLD) {
            return beta;
        }
    }

    // 4. ProbCut (Pruning non-tactical branches if shallow capture search easily beats beta + threshold)
    if (!in_check && depth >= 5 && std::abs(beta) < VALUE_MATE_THRESHOLD) {
        Value probcut_beta = beta + 180;
        int probcut_depth = depth - 4;

        std::vector<Move> captures;
        board.generate_captures(captures);
        order_moves(board, w, captures, tt_move, ply);

        for (const auto& cm : captures) {
            if (SEE::evaluate(board, cm) < 0) continue;
            w.move_stack[ply] = cm;
            if (!board.make_move(cm)) continue;

            Value score = -qsearch(board, w, ply + 1, -probcut_beta, -probcut_beta + 1);
            if (score >= probcut_beta) {
                score = -pvs(board, w, probcut_depth, ply + 1, -probcut_beta, -probcut_beta + 1, !cut_node);
            }
            board.unmake_move(cm);

            if (score >= probcut_beta) {
                return beta;
            }
        }
    }

    // Generate legal moves
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
    std::vector<Move> quiet_moves_tried;

    for (const auto& m : moves) {
        bool is_capture = (board.piece_at(m.to_sq()) != NO_PIECE) || m.is_en_passant();
        bool is_quiet = !is_capture && !m.is_promotion();

        w.move_stack[ply] = m;
        if (!board.make_move(m)) continue;
        moves_searched++;

        if (is_quiet) quiet_moves_tried.push_back(m);

        Value score;
        if (moves_searched == 1) {
            score = -pvs(board, w, depth - 1, ply + 1, -beta, -alpha, false);
        } else {
            // Systematic Late Move Reductions (LMR)
            int reduction = 0;
            if (depth >= 3 && moves_searched > 3 && is_quiet && !in_check) {
                int d_idx = std::min(depth, 63);
                int m_idx = std::min(moves_searched, 63);
                reduction = LMR_TABLE[d_idx][m_idx];

                // Reduce less if cut node or in check
                if (cut_node) reduction += 1;
                if (board.gives_check(m)) reduction -= 1;
                reduction = std::max(0, std::min(depth - 2, reduction));
            }

            score = -pvs(board, w, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, true);

            // Re-search at full depth if reduced search exceeded alpha
            if (score > alpha && reduction > 0) {
                score = -pvs(board, w, depth - 1, ply + 1, -alpha - 1, -alpha, true);
            }
            // PVS full window re-search
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

                // Beta-cutoff (fail-high)
                if (score >= beta) {
                    flag = TT_LOWERBOUND;

                    if (is_quiet) {
                        if (ply < 128) {
                            w.killer_moves[ply][1] = w.killer_moves[ply][0];
                            w.killer_moves[ply][0] = m;
                        }
                        update_quiet_history(w, board, m, quiet_moves_tried, depth, ply);
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
    w.clear();

    for (int depth = 1 + (worker_id % 2); depth <= max_depth && !stop_flag_.load(); ++depth) {
        pvs(board, w, depth, 0, -VALUE_INFINITE, VALUE_INFINITE, false);
    }
}

void Search::start(Board& board, const SearchLimits& limits) {
    limits_ = limits;
    stop_flag_.store(false);
    searching_.store(true);
    pondering_.store(limits_.ponder);
    nodes_.store(0);
    start_time_ = std::chrono::steady_clock::now();

    target_time_ms_ = -1;
    hard_limit_ms_ = -1;

    if (limits_.infinite) {
        target_time_ms_ = -1;
        hard_limit_ms_ = -1;
    } else if (limits_.movetime > 0) {
        hard_limit_ms_ = std::max(int64_t(10), limits_.movetime - 15);
        target_time_ms_ = std::max(int64_t(10), limits_.movetime - 30);
    } else {
        int64_t my_time = (board.side_to_move() == WHITE) ? limits_.wtime : limits_.btime;
        int64_t my_inc  = (board.side_to_move() == WHITE) ? limits_.winc  : limits_.binc;

        if (my_time > 0) {
            int moves_left = (limits_.movestogo > 0) ? std::min(limits_.movestogo, 50) : 30;
            int64_t target = (my_time / moves_left) + (my_inc * 4 / 5);
            int64_t max_allowed = std::min(static_cast<int64_t>(my_time * 0.4), my_time - 50);
            int64_t hard = std::min(target * 3, max_allowed);

            target_time_ms_ = std::max(int64_t(10), target);
            hard_limit_ms_ = std::max(target_time_ms_, hard);
        }
    }

    g_tt.new_search();
    main_worker_.id = 0;
    main_worker_.clear();

    // Check Root Tablebase probe first for instantaneous endgame win/draw moves
    Move tb_root_move = MOVE_NONE;
    Value tb_root_score = VALUE_ZERO;
    if (Syzygy::probe_root(board, tb_root_move, tb_root_score)) {
        std::cout << "info depth 1 score cp " << tb_root_score << " nodes 1 time 1 pv " << tb_root_move.to_uci() << std::endl;
        std::cout << "bestmove " << tb_root_move.to_uci() << std::endl;
        searching_.store(false);
        return;
    }

    int max_depth = std::min(limits_.depth, 64);

    helper_threads_.clear();
    for (int i = 1; i < num_threads_; ++i) {
        helper_threads_.emplace_back(&Search::worker_search, this, board, i, max_depth);
    }

    search_root(board);

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

    bool single_reply = (root_moves.size() == 1) && !limits_.infinite && (limits_.depth == 64);
    int max_depth = std::min(limits_.depth, 64);
    int64_t prev_elapsed = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        Value alpha = -VALUE_INFINITE;
        Value beta  = VALUE_INFINITE;

        // Systematic Aspiration Window: expands progressively (+/- 25, 60, 150, inf)
        int delta = 25;
        if (depth >= 4) {
            alpha = std::max(-VALUE_INFINITE, best_score - delta);
            beta  = std::min(VALUE_INFINITE, best_score + delta);
        }

        Value score = pvs(board, main_worker_, depth, 0, alpha, beta, false);

        // Aspiration window fail-high or fail-low loop
        while (!stop_flag_.load()) {
            if (score <= alpha) {
                alpha = std::max(-VALUE_INFINITE, alpha - delta);
                delta *= 2;
                score = pvs(board, main_worker_, depth, 0, alpha, beta, false);
            } else if (score >= beta) {
                beta = std::min(VALUE_INFINITE, beta + delta);
                delta *= 2;
                score = pvs(board, main_worker_, depth, 0, alpha, beta, false);
            } else {
                break;
            }

            if (delta >= 1000) {
                score = pvs(board, main_worker_, depth, 0, -VALUE_INFINITE, VALUE_INFINITE, false);
                break;
            }
        }

        if (stop_flag_.load()) break;

        best_score = score;
        if (main_worker_.pv_length[0] > 0) {
            best_move = main_worker_.pv_table[0][0];
        }

        auto now = std::chrono::steady_clock::now();
        int64_t elapsed_ms = std::max(int64_t(1), std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count());
        int64_t depth_time = elapsed_ms - prev_elapsed;
        prev_elapsed = elapsed_ms;

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

        if (single_reply && depth >= 1 && !pondering_.load()) break;
        if (limits_.nodes > 0 && current_nodes >= static_cast<uint64_t>(limits_.nodes)) break;
        if (std::abs(score) >= VALUE_MATE_THRESHOLD && !pondering_.load()) break;

        if (!pondering_.load(std::memory_order_relaxed)) {
            if (target_time_ms_ > 0) {
                if (elapsed_ms >= target_time_ms_) break;
                if (hard_limit_ms_ > 0 && depth >= 5) {
                    if (elapsed_ms + (depth_time * 2.2) > hard_limit_ms_) {
                        break;
                    }
                }
            }
        }
    }

    while (pondering_.load(std::memory_order_relaxed) && !stop_flag_.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    Move ponder_move = MOVE_NONE;
    if (main_worker_.pv_length[0] > 1) {
        ponder_move = main_worker_.pv_table[0][1];
    }

    std::cout << "bestmove " << best_move.to_uci();
    if (ponder_enabled_ && !ponder_move.is_null()) {
        std::cout << " ponder " << ponder_move.to_uci();
    }
    std::cout << std::endl;
}
