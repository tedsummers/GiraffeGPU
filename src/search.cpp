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
    , tb_hits_(0)
    , start_time_ns_(0)
    , ponder_enabled_(true)
    , policy_net_enabled_(true)
    , num_threads_(4)
    , aspiration_window_(32)
    , null_move_reduction_(3)
    , lmr_moves_threshold_(3)
    , lmr_depth_threshold_(3)
    , probcut_margin_(180)
    , move_overhead_ms_(30)
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
    while (static_cast<int>(workers_.size()) < num_threads_ - 1) {
        auto w = std::make_unique<SearchWorker>();
        w->id = static_cast<int>(workers_.size()) + 1;
        w->clear();
        workers_.push_back(std::move(w));
    }
}

void Search::clear_histories() {
    main_worker_.clear();
    for (auto& w : workers_) {
        if (w) w->clear();
    }
}

void Search::stop() {
    stop_flag_.store(true, std::memory_order_relaxed);
    pondering_.store(false, std::memory_order_relaxed);
}

void Search::ponderhit() {
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    start_time_ns_.store(now_ns, std::memory_order_relaxed);
    pondering_.store(false, std::memory_order_relaxed);
}

int64_t Search::elapsed_ms() const {
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t start_ns = start_time_ns_.load(std::memory_order_relaxed);
    return std::max<int64_t>(0, (now_ns - start_ns) / 1000000);
}

bool Search::check_time(bool force_clock_check) {
    if (stop_flag_.load(std::memory_order_relaxed)) return true;
    if (limits_.nodes > 0 && nodes_.load(std::memory_order_relaxed) >= static_cast<uint64_t>(limits_.nodes)) {
        stop_flag_.store(true, std::memory_order_relaxed);
        return true;
    }
    if (pondering_.load(std::memory_order_relaxed)) return false;

    if (hard_limit_ms_ > 0 && (force_clock_check || (nodes_.load(std::memory_order_relaxed) & 7) == 0)) {
        if (elapsed_ms() >= hard_limit_ms_) {
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
    score = std::max(-MAX_HISTORY, std::min(MAX_HISTORY, score));
}

static inline void update_history_score(int16_t& score, int bonus) {
    constexpr int MAX_HISTORY = 16384;
    int value = static_cast<int>(score);
    value += bonus - (value * std::abs(bonus)) / MAX_HISTORY;
    score = static_cast<int16_t>(std::max(-MAX_HISTORY, std::min(MAX_HISTORY, value)));
}

void Search::update_quiet_history(SearchWorker& w, const Board& board, Move best_quiet,
                                  const MoveList& quiet_moves, int depth, int ply)
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
        Piece curr_p = board.piece_at(best_quiet.from_sq());
        if (prev_p != NO_PIECE && curr_p != NO_PIECE) {
            w.countermove[prev_p][prev_m.to_sq()] = best_quiet;
            update_history_score(w.conthist[prev_p][prev_m.to_sq()][curr_p][best_quiet.to_sq()], bonus);
        }
    }

    // Penalize other tried quiet moves with negative gravity
    int malus = -bonus / 2;
    for (size_t i = 0; i < quiet_moves.size(); ++i) {
        Move m = quiet_moves[i];
        if (m == best_quiet) continue;
        Piece p = board.piece_at(m.from_sq());
        update_history_score(w.history[p][m.to_sq()], malus);

        if (ply > 0 && !w.move_stack[ply - 1].is_null()) {
            Move prev_m = w.move_stack[ply - 1];
            Piece prev_p = board.piece_at(prev_m.to_sq());
            if (prev_p != NO_PIECE) {
                Piece curr_p = board.piece_at(m.from_sq());
                if (curr_p != NO_PIECE)
                    update_history_score(w.conthist[prev_p][prev_m.to_sq()][curr_p][m.to_sq()], malus);
            }
        }
    }
}

void Search::order_moves(const Board& board, const SearchWorker& w, MoveList& moves, Move tt_move, int ply, const float* policy_h2) {
    int scores[256];
    const size_t n = moves.size();

    Move prev_m = (ply > 0) ? w.move_stack[ply - 1] : MOVE_NONE;
    Piece prev_p = (!prev_m.is_null()) ? board.piece_at(prev_m.to_sq()) : NO_PIECE;
    Move cm = (prev_p != NO_PIECE) ? w.countermove[prev_p][prev_m.to_sq()] : MOVE_NONE;

    for (size_t i = 0; i < n; ++i) {
        const Move m = moves[i];
        int score = 0;
        int pol_pts = 0;
        if (policy_h2 != nullptr && m != tt_move) {
            int pol_idx = (static_cast<int>(m.from_sq()) << 6) | static_cast<int>(m.to_sq());
            float pol_logit = Neural::g_neural_gpu.score_move_policy_fast_cpu(pol_idx, policy_h2);
            if (m.is_promotion() && m.promotion_type() == QUEEN) {
                pol_logit += 0.5f;
            }
            pol_pts = static_cast<int>(pol_logit * 5000.0f);
        }

        if (m == tt_move) {
            score = 300000;
        } else if (board.piece_at(m.to_sq()) != NO_PIECE || m.is_en_passant() || m.is_promotion()) {
            // Captures and Promotions: MVV-LVA + SEE + Policy
            Piece victim = board.piece_at(m.to_sq());
            Piece attacker = board.piece_at(m.from_sq());
            PieceType v_pt = (victim != NO_PIECE) ? piece_type(victim) : PAWN;
            PieceType a_pt = piece_type(attacker);

            if (m.is_promotion()) {
                score = 150000 + SEE::PIECE_VALUES[m.promotion_type()] + pol_pts;
            } else {
                Value see_val = SEE::evaluate(board, m);
                if (see_val >= 0) {
                    score = 100000 + (SEE::PIECE_VALUES[v_pt] * 10) - SEE::PIECE_VALUES[a_pt] + see_val + (pol_pts / 2);
                } else {
                    score = -50000 + see_val;
                }
            }
        } else {
            // Quiet moves: Killers -> Countermove -> Continuation History -> Main History + Policy
            if (ply < 128 && m == w.killer_moves[ply][0]) {
                score = 90000 + pol_pts;
            } else if (ply < 128 && m == w.killer_moves[ply][1]) {
                score = 80000 + pol_pts;
            } else if (m == cm) {
                score = 75000 + pol_pts;
            } else {
                Piece p = board.piece_at(m.from_sq());
                int h_score = w.history[p][m.to_sq()];
                int ch_score = (prev_p != NO_PIECE && p != NO_PIECE)
                    ? w.conthist[prev_p][prev_m.to_sq()][p][m.to_sq()] : 0;
                score = h_score + ch_score * 2 + pol_pts;
            }
        }
        scores[i] = score;
    }

    // In-place insertion sort on stack arrays (zero heap allocations, fast for N <= 64)
    for (size_t i = 1; i < n; ++i) {
        int key_score = scores[i];
        Move key_move = moves[i];
        size_t j = i;
        while (j > 0 && scores[j - 1] < key_score) {
            scores[j] = scores[j - 1];
            moves[j] = moves[j - 1];
            --j;
        }
        scores[j] = key_score;
        moves[j] = key_move;
    }
}

Value Search::qsearch(Board& board, SearchWorker& w, int ply, Value alpha, Value beta, int qply) {
    if (check_time((++w.local_nodes & 7) == 0)) return VALUE_ZERO;

    nodes_++;
    if (ply > w.seldepth) w.seldepth = ply;
    if (ply >= 127) return Neural::g_neural_gpu.evaluate(board, w.id);

    if (board.is_draw(ply)) return VALUE_DRAW;

    Value tb_score;
    if (ply > 0 && (hard_limit_ms_ < 0 || hard_limit_ms_ >= 35) && Syzygy::probe_wdl(board, tb_score, ply)) {
        tb_hits_.fetch_add(1, std::memory_order_relaxed);
        if (root_piece_count_ > Syzygy::max_pieces() || tb_score == VALUE_DRAW) {
            return tb_score;
        }
    }

    TTEntry tt_entry;
    Move tt_move = MOVE_NONE;
    bool tt_hit = g_tt.probe(board.key(), tt_entry);
    if (tt_hit) {
        tt_move = Move(tt_entry.move_data);
        Value tt_score = g_tt.score_from_tt(tt_entry.score, ply);
        if (tt_entry.flag == TT_EXACT) return tt_score;
        if (tt_entry.flag == TT_LOWERBOUND && tt_score >= beta) return beta;
        if (tt_entry.flag == TT_UPPERBOUND && tt_score <= alpha) return alpha;
    }

    bool in_check = board.in_check();
    Value stand_pat = -VALUE_INFINITE;

    if (!in_check) {
        if (tt_hit && tt_entry.eval != static_cast<int16_t>(VALUE_NONE)) {
            stand_pat = tt_entry.eval;
        } else if (w.eval_stack[ply] != VALUE_NONE) {
            stand_pat = w.eval_stack[ply];
        } else {
            stand_pat = Neural::g_neural_gpu.evaluate(board, w.id);
        }
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
    }

    MoveList moves;
    if (in_check) {
        board.generate_pseudo_legal_moves(moves);
    } else if (qply < 1) {
        MoveList all_moves;
        board.generate_pseudo_legal_moves(all_moves);
        for (size_t i = 0; i < all_moves.size(); ++i) {
            const Move m = all_moves[i];
            if (board.piece_at(m.to_sq()) != NO_PIECE || m.is_en_passant() || m.is_promotion() || board.gives_check(m)) {
                moves.push_back(m);
            }
        }
    } else {
        board.generate_pseudo_captures(moves);
    }

    order_moves(board, w, moves, tt_move, ply);

    Value q_child_evals[256];
    for (size_t k = 0; k < moves.size(); ++k) q_child_evals[k] = VALUE_NONE;
    bool q_children_batched = false;

    int legal_moves = 0;
    for (size_t i = 0; i < moves.size(); ++i) {
        // After the first legal capture fails to cut off, batch-evaluate all remaining legal capture children on the GPU!
        if (legal_moves == 1 && !q_children_batched && i + 1 < moves.size()) {
            if (check_time(true)) return VALUE_ZERO;
            q_children_batched = true;
            Board batch_boards[64];
            size_t batch_move_idx[64];
            Value batch_scores[64];
            int batch_count = 0;

            for (size_t k = i; k < moves.size() && batch_count < 64; ++k) {
                const Move cm = moves[k];
                if (!in_check && !cm.is_promotion() && SEE::evaluate(board, cm) < 0) continue;
                if (!board.make_move(cm)) continue;
                if (!board.in_check()) {
                    batch_boards[batch_count] = board;
                    batch_move_idx[batch_count] = k;
                    batch_count++;
                }
                board.unmake_move(cm);
            }

            if (batch_count > 1) {
                Neural::g_neural_gpu.evaluate_batch_raw(batch_boards, batch_count, batch_scores, w.id);
                for (int b = 0; b < batch_count; ++b) {
                    q_child_evals[batch_move_idx[b]] = batch_scores[b];
                }
            }
        }

        const Move m = moves[i];
        if (!in_check && !m.is_promotion() && SEE::evaluate(board, m) < 0) {
            continue;
        }

        w.move_stack[ply] = m;
        if (!board.make_move(m)) continue;
        legal_moves++;

        if (ply + 1 < 128) {
            w.eval_stack[ply + 1] = q_child_evals[i];
        }

        Value score = -qsearch(board, w, ply + 1, -beta, -alpha, qply + 1);
        board.unmake_move(m);

        if (stop_flag_.load(std::memory_order_relaxed)) return VALUE_ZERO;

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    if (in_check && legal_moves == 0) {
        return -VALUE_MATE + ply;
    }

    return alpha;
}

Value Search::pvs(Board& board, SearchWorker& w, int depth, int ply, Value alpha, Value beta, bool cut_node) {
    if (check_time(true)) return VALUE_ZERO;

    w.pv_length[ply] = ply;

    if (ply >= 127) return Neural::g_neural_gpu.evaluate(board, w.id);
    if (ply > 0 && board.is_draw(ply)) return VALUE_DRAW;

    const bool pv_node = (beta - alpha > 1);

    alpha = std::max(alpha, static_cast<Value>(-VALUE_MATE + ply));
    beta  = std::min(beta, static_cast<Value>(VALUE_MATE - ply - 1));
    if (alpha >= beta) return alpha;

    // Check extensions
    bool in_check = board.in_check();
    if (in_check) depth++;

    if (depth <= 0) {
        return qsearch(board, w, ply, alpha, beta, 0);
    }

    nodes_++;
    ++w.local_nodes;

    // 1. Syzygy Tablebase Probe
    Value tb_score;
    if (ply > 0 && (hard_limit_ms_ < 0 || hard_limit_ms_ >= 35) && Syzygy::probe_wdl(board, tb_score, ply)) {
        tb_hits_.fetch_add(1, std::memory_order_relaxed);
        if (root_piece_count_ > Syzygy::max_pieces() || tb_score == VALUE_DRAW) {
            TTFlag tb_flag = (tb_score >= beta) ? TT_LOWERBOUND : (tb_score <= alpha) ? TT_UPPERBOUND : TT_EXACT;
            g_tt.store(board.key(), MOVE_NONE, tb_score, depth, tb_flag, ply, VALUE_NONE);
            if (tb_score >= beta) return beta;
            if (tb_score <= alpha) return alpha;
            return tb_score;
        }
    }

    // 2. Transposition Table Probe
    TTEntry tt_entry;
    Move tt_move = MOVE_NONE;
    bool tt_hit = g_tt.probe(board.key(), tt_entry);
    if (tt_hit) {
        tt_move = Move(tt_entry.move_data);
        if (tt_entry.depth >= depth && ply > 0 && !pv_node) {
            Value tt_score = g_tt.score_from_tt(tt_entry.score, ply);
            if (tt_entry.flag == TT_EXACT) return tt_score;
            if (tt_entry.flag == TT_LOWERBOUND && tt_score >= beta) return beta;
            if (tt_entry.flag == TT_UPPERBOUND && tt_score <= alpha) return alpha;
        }
    }

    // Compute or retrieve static evaluation and optional L2 hidden state on GPU
    Value static_eval = VALUE_NONE;
    alignas(32) float policy_h2[Neural::L2_DIM];
    const float* h2_ptr = nullptr;
    const bool use_policy = policy_net_enabled_ && (depth >= 4 || ply == 0);

    if (!in_check) {
        if (use_policy) {
            static_eval = Neural::g_neural_gpu.evaluate_with_h2(board, policy_h2, w.id);
            h2_ptr = policy_h2;
        } else if (tt_hit && tt_entry.eval != static_cast<int16_t>(VALUE_NONE)) {
            static_eval = tt_entry.eval;
        } else if (ply > 0 && w.eval_stack[ply] != VALUE_NONE) {
            static_eval = w.eval_stack[ply];
        } else {
            static_eval = Neural::g_neural_gpu.evaluate(board, w.id);
        }
        w.eval_stack[ply] = static_eval;
    } else {
        w.eval_stack[ply] = VALUE_NONE;
    }

    const bool improving = !in_check && ply >= 2 &&
                           w.eval_stack[ply - 2] != VALUE_NONE &&
                           static_eval > w.eval_stack[ply - 2];

    if (!pv_node && !in_check && ply > 0 && std::abs(beta) < VALUE_MATE_THRESHOLD) {
        // 3a. Reverse Futility Pruning (Static Null Move Pruning)
        if (depth <= 6) {
            int rfp_margin = (depth - (improving ? 1 : 0)) * 75;
            if (static_eval - rfp_margin >= beta) {
                return static_eval;
            }
        }

        // 3b. Razoring
        if (depth <= 2 && static_eval + 250 * depth <= alpha) {
            Value r_score = qsearch(board, w, ply, alpha, alpha + 1, 0);
            if (r_score <= alpha) return r_score;
        }

        // 3c. Null Move Pruning (checking side-to-move's non-pawn material + clearing move_stack)
        if (depth >= 3 && static_eval >= beta && board.has_non_pawn_material(board.side_to_move())) {
            w.move_stack[ply] = MOVE_NONE;
            if (ply + 1 < 128) w.eval_stack[ply + 1] = VALUE_NONE;
            board.make_null_move();
            int R = null_move_reduction_ + depth / 4 + std::min(3, static_cast<int>(static_eval - beta) / 180);
            Value null_score = -pvs(board, w, depth - 1 - R, ply + 1, -beta, -beta + 1, !cut_node);
            board.unmake_null_move();

            if (null_score >= beta && std::abs(null_score) < VALUE_MATE_THRESHOLD) {
                if (depth >= 10) {
                    Value verif = pvs(board, w, depth - R, ply, beta - 1, beta, false);
                    if (verif >= beta) return beta;
                } else {
                    return beta;
                }
            }
        }

        // 4. ProbCut (Pruning non-tactical branches if shallow capture search easily beats beta + threshold)
        if (depth >= 5) {
            Value probcut_beta = beta + probcut_margin_;
            int probcut_depth = depth - 4;

            MoveList captures;
            board.generate_pseudo_captures(captures);
            order_moves(board, w, captures, tt_move, ply);

            for (size_t i = 0; i < captures.size(); ++i) {
                const Move cm = captures[i];
                if (SEE::evaluate(board, cm) < 0) continue;
                w.move_stack[ply] = cm;
                if (ply + 1 < 128) w.eval_stack[ply + 1] = VALUE_NONE;
                if (!board.make_move(cm)) continue;

                Value score = -qsearch(board, w, ply + 1, -probcut_beta, -probcut_beta + 1, 0);
                if (score >= probcut_beta) {
                    score = -pvs(board, w, probcut_depth, ply + 1, -probcut_beta, -probcut_beta + 1, !cut_node);
                }
                board.unmake_move(cm);

                if (score >= probcut_beta) {
                    return beta;
                }
            }
        }
    }

    // Internal Iterative Reduction (IIR)
    if (depth >= 4 && tt_move.is_null() && (pv_node || cut_node)) {
        depth--;
    }

    // Generate pseudo-legal moves (or use Syzygy-filtered root moves at ply 0)
    MoveList moves;
    if (ply == 0 && !root_tb_moves_.empty()) {
        moves = root_tb_moves_;
    } else {
        board.generate_pseudo_legal_moves(moves);
    }
    order_moves(board, w, moves, tt_move, ply, h2_ptr);

    Move best_move = MOVE_NONE;
    TTFlag flag = TT_UPPERBOUND;
    Value best_score = -VALUE_INFINITE;
    int moves_searched = 0;
    MoveList quiet_moves_tried;
    bool history_updated_on_cutoff = false;

    Value child_gpu_evals[256];
    for (size_t k = 0; k < moves.size(); ++k) child_gpu_evals[k] = VALUE_NONE;
    bool children_batched = false;

    for (size_t i = 0; i < moves.size(); ++i) {
        // After searching the first move (e.g. TT move), if no beta cutoff occurred,
        // batch-evaluate all remaining legal child positions in a single multi-block GPU launch!
        if (moves_searched == 1 && !children_batched && i + 1 < moves.size()) {
            if (check_time(true)) return VALUE_ZERO;
            children_batched = true;
            Board batch_boards[64];
            size_t batch_move_idx[64];
            Value batch_scores[64];
            int batch_count = 0;
            int sim_moves_searched = moves_searched;

            for (size_t k = i; k < moves.size() && batch_count < 64; ++k) {
                Move cm = moves[k];
                bool cm_capture = (board.piece_at(cm.to_sq()) != NO_PIECE) || cm.is_en_passant();
                bool cm_quiet = !cm_capture && !cm.is_promotion();
                bool cm_check = board.gives_check(cm);

                if (ply > 0 && !pv_node && !in_check && cm_quiet && !cm_check && best_score > -VALUE_MATE_THRESHOLD) {
                    if (depth <= 4 && sim_moves_searched >= (3 + depth * depth) / (2 - (improving ? 1 : 0))) continue;
                    if (depth <= 3 && static_eval + 110 * depth + 80 <= alpha) continue;
                    if (depth <= 4 && SEE::evaluate(board, cm) < -60 * depth) continue;
                }

                if (!board.make_move(cm)) continue;
                sim_moves_searched++;
                if (!board.in_check()) {
                    batch_boards[batch_count] = board;
                    batch_move_idx[batch_count] = k;
                    batch_count++;
                }
                board.unmake_move(cm);
            }

            if (batch_count > 1) {
                Neural::g_neural_gpu.evaluate_batch_raw(batch_boards, batch_count, batch_scores, w.id);
                for (int b = 0; b < batch_count; ++b) {
                    child_gpu_evals[batch_move_idx[b]] = batch_scores[b];
                }
            }
        }

        const Move m = moves[i];
        bool is_capture = (board.piece_at(m.to_sq()) != NO_PIECE) || m.is_en_passant();
        bool is_quiet = !is_capture && !m.is_promotion();
        bool gives_check = board.gives_check(m);

        // Futility Pruning & Late Move Pruning (LMP) for quiet moves at shallow depths
        if (ply > 0 && !pv_node && !in_check && is_quiet && !gives_check &&
            moves_searched > 0 && best_score > -VALUE_MATE_THRESHOLD)
        {
            if (depth <= 4 && moves_searched >= (3 + depth * depth) / (2 - (improving ? 1 : 0))) {
                continue;
            }
            if (depth <= 3 && static_eval + 110 * depth + 80 <= alpha) {
                continue;
            }
            if (depth <= 4 && SEE::evaluate(board, m) < -60 * depth) {
                continue;
            }
        }

        w.move_stack[ply] = m;
        if (!board.make_move(m)) continue;
        moves_searched++;

        if (ply + 1 < 128) {
            w.eval_stack[ply + 1] = child_gpu_evals[i];
        }

        if (is_quiet) quiet_moves_tried.push_back(m);

        Value score;
        if (moves_searched == 1) {
            score = -pvs(board, w, depth - 1, ply + 1, -beta, -alpha, false);
        } else {
            // Systematic Late Move Reductions (LMR)
            int reduction = 0;
            if (depth >= lmr_depth_threshold_ && moves_searched > lmr_moves_threshold_ && (is_quiet || !pv_node) && !in_check) {
                int d_idx = std::min(depth, 63);
                int m_idx = std::min(moves_searched, 63);
                reduction = LMR_TABLE[d_idx][m_idx];

                if (!pv_node) reduction += 1;
                if (cut_node) reduction += 1;
                if (!improving) reduction += 1;
                if (gives_check) reduction -= 1;

                if (is_quiet) {
                    Piece p = board.piece_at(m.to_sq()); // piece is on to_sq after make_move
                    int h_score = w.history[p][m.to_sq()];
                    reduction -= h_score / 8192;
                }

                // Policy-guided reduction discount or penalty
                if (h2_ptr) {
                    int pol_idx = (static_cast<int>(m.from_sq()) << 6) | static_cast<int>(m.to_sq());
                    float pol = Neural::g_neural_gpu.score_move_policy_fast_cpu(pol_idx, h2_ptr);
                    if (pol > 1.2f) reduction -= 1;
                    else if (pol < -1.5f) reduction += 1;
                }

                reduction = std::max(0, std::min(depth - 2, reduction));
            }

            score = -pvs(board, w, depth - 1 - reduction, ply + 1, -alpha - 1, -alpha, true);

            // Re-search at full depth if reduced search exceeded alpha
            if (score > alpha && reduction > 0) {
                score = -pvs(board, w, depth - 1, ply + 1, -alpha - 1, -alpha, !cut_node);
            }
            // PVS full window re-search
            if (score > alpha && score < beta) {
                score = -pvs(board, w, depth - 1, ply + 1, -beta, -alpha, false);
            }
        }

        board.unmake_move(m);

        if (stop_flag_.load(std::memory_order_relaxed)) return VALUE_ZERO;

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
                        if (ply < 128 && w.killer_moves[ply][0] != m) {
                            w.killer_moves[ply][1] = w.killer_moves[ply][0];
                            w.killer_moves[ply][0] = m;
                        }
                        update_quiet_history(w, board, m, quiet_moves_tried, depth, ply);
                        history_updated_on_cutoff = true;
                    }
                    break;
                }
            }
        }
    }

    if (moves_searched == 0) {
        if (in_check) return -VALUE_MATE + ply;
        return VALUE_DRAW;
    }

    // Also learn from principal-variation nodes. Restricting updates to
    // cutoffs leaves history tables cold in quiet positions.
    if (!history_updated_on_cutoff && !best_move.is_null() && !best_move.is_promotion() &&
        board.piece_at(best_move.to_sq()) == NO_PIECE && !best_move.is_en_passant()) {
        update_quiet_history(w, board, best_move, quiet_moves_tried, depth, ply);
    }

    if (!stop_flag_.load(std::memory_order_relaxed)) {
        g_tt.store(board.key(), best_move, best_score, depth, flag, ply, static_eval);
    }

    return best_score;
}

void Search::worker_search(Board board, int worker_id, int max_depth) {
    SearchWorker& w = *workers_[worker_id - 1];
    w.id = worker_id;
    w.clear_search();

    for (int depth = 1 + (worker_id % 2); depth <= max_depth && !stop_flag_.load(std::memory_order_relaxed); ++depth) {
        pvs(board, w, depth, 0, -VALUE_INFINITE, VALUE_INFINITE, false);
    }
}

void Search::start(Board& board, const SearchLimits& limits) {
    limits_ = limits;
    stop_flag_.store(false);
    searching_.store(true);
    pondering_.store(limits_.ponder);
    nodes_.store(0);
    tb_hits_.store(0);
    auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        limits_.start_time.time_since_epoch()).count();
    start_time_ns_.store(start_ns, std::memory_order_relaxed);

    target_time_ms_ = -1;
    hard_limit_ms_ = -1;

    if (limits_.infinite) {
        target_time_ms_ = -1;
        hard_limit_ms_ = -1;
    } else if (limits_.movetime > 0) {
        int64_t safety = std::max(int64_t(move_overhead_ms_),
                                  std::max(int64_t(25), std::min(int64_t(250), limits_.movetime / 10)));
        hard_limit_ms_ = std::max(int64_t(2), limits_.movetime - safety);
        target_time_ms_ = std::max(int64_t(1), hard_limit_ms_ * 3 / 4);
    } else {
        int64_t my_time = (board.side_to_move() == WHITE) ? limits_.wtime : limits_.btime;
        int64_t my_inc  = (board.side_to_move() == WHITE) ? limits_.winc  : limits_.binc;

        if (my_time > 0) {
            int moves_left = (limits_.movestogo > 0) ? std::min(limits_.movestogo, 50) : 35;
            int64_t safety = std::max(int64_t(move_overhead_ms_),
                                      std::max(int64_t(40), std::min(int64_t(500), my_time / 15)));
            if (my_time <= safety + 5) {
                target_time_ms_ = 1;
                hard_limit_ms_ = std::max(int64_t(2), my_time / 4);
            } else {
                int64_t available = my_time - safety;
                int64_t allocation = (available / moves_left) + (my_inc * 3 / 4);
                target_time_ms_ = std::max(int64_t(1), std::min(available / 3, allocation));
                hard_limit_ms_ = std::max(target_time_ms_, std::min(available * 3 / 4, target_time_ms_ * 2 + 10));
            }
        }
    }

    g_tt.new_search();
    main_worker_.id = 0;
    main_worker_.clear_search();
    set_threads(num_threads_);

    root_piece_count_ = popcount(board.occupied_bb());
    root_tb_moves_.clear();
    // Avoid blocking on 6-piece DTZ disk I/O if clock is critically low (< 35ms)
    if (hard_limit_ms_ < 0 || hard_limit_ms_ >= 35 || root_piece_count_ <= 5) {
        MoveList legal_root;
        board.generate_legal_moves(legal_root);
        Value tb_root_score = VALUE_ZERO;
        if (!legal_root.empty() && Syzygy::filter_root_moves(board, legal_root, tb_root_score)) {
            root_tb_moves_ = legal_root;
            tb_hits_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    int max_depth = std::min(limits_.depth, 64);

    helper_threads_.clear();
    if (hard_limit_ms_ < 0 || hard_limit_ms_ >= 25) {
        for (int i = 1; i < num_threads_; ++i) {
            helper_threads_.emplace_back(&Search::worker_search, this, board, i, max_depth);
        }
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

    MoveList root_moves;
    if (!root_tb_moves_.empty()) {
        root_moves = root_tb_moves_;
    } else {
        board.generate_legal_moves(root_moves);
    }
    if (root_moves.empty()) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }

    alignas(32) float root_h2[Neural::L2_DIM];
    const float* h2_ptr = nullptr;
    if (policy_net_enabled_ && (hard_limit_ms_ < 0 || hard_limit_ms_ >= 15)) {
        Neural::g_neural_gpu.evaluate_with_h2(board, root_h2, 0);
        h2_ptr = root_h2;
    }
    order_moves(board, main_worker_, root_moves, MOVE_NONE, 0, h2_ptr);
    best_move = root_moves[0];

    bool single_reply = (root_moves.size() == 1) && !limits_.infinite && (limits_.depth == 64);
    int max_depth = std::min(limits_.depth, 64);
    int64_t prev_elapsed = 0;

    for (int depth = 1; depth <= max_depth; ++depth) {
        if (depth > 1 && !pondering_.load(std::memory_order_relaxed) && target_time_ms_ > 0) {
            if (elapsed_ms() >= target_time_ms_) break;
        }
        Value alpha = -VALUE_INFINITE;
        Value beta  = VALUE_INFINITE;

        // Use a narrow window once the score is stable, then widen the failing bound
        // geometrically after a fail.
        int delta = (depth < 8) ? aspiration_window_ : std::max(8, aspiration_window_ / 2);
        if (depth >= 4 && std::abs(best_score) < VALUE_MATE_THRESHOLD) {
            alpha = std::max(-VALUE_INFINITE, best_score - delta);
            beta  = std::min(VALUE_INFINITE, best_score + delta);
        }

        Value score = pvs(board, main_worker_, depth, 0, alpha, beta, false);

        // Aspiration window fail-high or fail-low loop
        while (!stop_flag_.load(std::memory_order_relaxed)) {
            if (score <= alpha) {
                beta = (alpha + beta) / 2;
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

        int64_t cur_elapsed = std::max(int64_t(1), elapsed_ms());
        int64_t depth_time = std::max(int64_t(0), cur_elapsed - prev_elapsed);
        prev_elapsed = cur_elapsed;

        uint64_t current_nodes = nodes_.load();
        uint64_t current_tbhits = tb_hits_.load(std::memory_order_relaxed);
        uint64_t nps = (current_nodes * 1000) / cur_elapsed;

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
                  << " tbhits " << current_tbhits
                  << " time " << cur_elapsed
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
                // Soft time limit: if we've used >= 60% of target time, don't start another iteration
                // that would likely blow past target_time_ms_ into hard_limit_ms_.
                if (cur_elapsed >= (target_time_ms_ * 6) / 10) break;
                if (hard_limit_ms_ > 0 && cur_elapsed + (depth_time * 2) >= hard_limit_ms_) {
                    break;
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
