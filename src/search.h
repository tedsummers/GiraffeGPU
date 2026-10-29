#ifndef SEARCH_H
#define SEARCH_H

#include "types.h"
#include "board.h"
#include "tt.h"
#include "cuda/neural_gpu.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>
#include <memory>

struct SearchLimits {
    int depth = 64;
    int64_t nodes = -1;
    int64_t movetime = -1;
    int64_t wtime = -1;
    int64_t btime = -1;
    int64_t winc = 0;
    int64_t binc = 0;
    int movestogo = 30;
    bool infinite = false;
};

struct SearchWorker {
    int id;
    int seldepth;
    Move killer_moves[128][2];
    int history[PIECE_NB][64];
    Move pv_table[128][128];
    int pv_length[128];
};

class Search {
public:
    Search();
    ~Search();

    void set_threads(int num_threads);
    int num_threads() const { return num_threads_; }

    void start(Board& board, const SearchLimits& limits);
    void stop();

    bool is_searching() const { return searching_.load(); }
    uint64_t nodes_searched() const { return nodes_.load(); }

private:
    void search_root(Board& board);
    void worker_search(Board board, int worker_id, int max_depth);

    Value pvs(Board& board, SearchWorker& w, int depth, int ply, Value alpha, Value beta, bool cut_node);
    Value qsearch(Board& board, SearchWorker& w, int ply, Value alpha, Value beta);

    void order_moves(const Board& board, const SearchWorker& w, std::vector<Move>& moves, Move tt_move, int ply);
    bool check_time();

    std::atomic<bool> stop_flag_;
    std::atomic<bool> searching_;
    std::atomic<uint64_t> nodes_;

    int num_threads_;
    SearchLimits limits_;
    std::chrono::steady_clock::time_point start_time_;
    int64_t allocated_time_ms_;

    std::vector<std::thread> helper_threads_;
    SearchWorker main_worker_;
};

extern Search g_search;

#endif // SEARCH_H
