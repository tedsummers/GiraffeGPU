#ifndef SEARCH_H
#define SEARCH_H

#include "types.h"
#include "board.h"
#include "tt.h"
#include "cuda/neural_gpu.h"
#include "syzygy_tb.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>
#include <memory>
#include <array>
#include <cstring>

struct SearchLimits {
    int depth = 64;
    int64_t nodes = -1;
    int64_t movetime = -1;
    int64_t wtime = -1;
    int64_t btime = -1;
    int64_t winc = 0;
    int64_t binc = 0;
    int movestogo = 0;
    bool infinite = false;
    bool ponder = false;
};

// Search tables for move ordering: history, countermove, and continuation history
struct SearchWorker {
    int id;
    int seldepth;
    Move killer_moves[128][2];
    
    // Main history: piece -> to_square
    int history[PIECE_NB][64];

    // Countermove table: prev_piece -> prev_to_square -> counter_move
    Move countermove[PIECE_NB][64];

    // Continuation history (1-ply prior move response):
    // prev_piece -> prev_to_square -> curr_piece -> curr_to_square
    int conthist[PIECE_NB][64][64];

    // Move stack for continuation history lookups
    Move move_stack[128];

    Move pv_table[128][128];
    int pv_length[128];

    void clear() {
        seldepth = 0;
        for (int i = 0; i < 128; i++) {
            killer_moves[i][0] = Move();
            killer_moves[i][1] = Move();
            move_stack[i] = Move();
            pv_length[i] = 0;
            for (int j = 0; j < 128; j++) pv_table[i][j] = Move();
        }
        std::memset(history, 0, sizeof(history));
        for (int p = 0; p < PIECE_NB; p++) {
            for (int s = 0; s < 64; s++) {
                countermove[p][s] = Move();
            }
        }
        std::memset(conthist, 0, sizeof(conthist));
    }
};

class Search {
public:
    Search();
    ~Search();

    void set_threads(int num_threads);
    int num_threads() const { return num_threads_; }

    void start(Board& board, const SearchLimits& limits);
    void stop();
    void ponderhit();

    void set_ponder_enabled(bool enabled) { ponder_enabled_ = enabled; }
    bool is_ponder_enabled() const { return ponder_enabled_; }
    bool is_pondering() const { return pondering_.load(); }

    bool is_searching() const { return searching_.load(); }
    uint64_t nodes_searched() const { return nodes_.load(); }

    void set_syzygy_path(const std::string& path) {
        Syzygy::init(path);
    }

private:
    void search_root(Board& board);
    void worker_search(Board board, int worker_id, int max_depth);

    Value pvs(Board& board, SearchWorker& w, int depth, int ply, Value alpha, Value beta, bool cut_node);
    Value qsearch(Board& board, SearchWorker& w, int ply, Value alpha, Value beta);

    void order_moves(const Board& board, const SearchWorker& w, std::vector<Move>& moves, Move tt_move, int ply);
    void update_quiet_history(SearchWorker& w, const Board& board, Move best_quiet, const std::vector<Move>& quiet_moves, int depth, int ply);

    bool check_time();

    std::atomic<bool> stop_flag_;
    std::atomic<bool> searching_;
    std::atomic<bool> pondering_;
    std::atomic<uint64_t> nodes_;

    bool ponder_enabled_;
    int num_threads_;
    SearchLimits limits_;
    std::chrono::steady_clock::time_point start_time_;
    int64_t target_time_ms_;
    int64_t hard_limit_ms_;

    std::vector<std::thread> helper_threads_;
    SearchWorker main_worker_;
};

extern Search g_search;

#endif // SEARCH_H
