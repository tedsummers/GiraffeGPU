#ifndef TT_H
#define TT_H

#include "types.h"
#include <vector>
#include <cstdint>

constexpr Value VALUE_NONE = 32001;

enum TTFlag : uint8_t {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_LOWERBOUND = 2,
    TT_UPPERBOUND = 3
};

// A compact transposition-table entry.  Keep the complete key XORed with payload
// for lockless multi-threaded safety, and store full 16-bit static evaluation.
struct TTEntry {
    uint64_t key;           // Full Zobrist key (XORed with data in table storage)
    union {
        uint64_t data;
        struct {
            uint16_t move_data;     // Best move encoded
            int16_t score;          // Centipawn / mate score
            int16_t eval;           // Static evaluation cache
            int8_t depth;           // Search depth
            uint8_t flag : 2;       // Exact, Lowerbound, Upperbound
            uint8_t generation : 6; // Aging generation (0..63)
        };
    };
};

// Four entries per cluster gives replacement policy room without increasing
// the number of hash-index calculations.
struct TTCluster {
    TTEntry entries[4];
};

class TranspositionTable {
public:
    TranspositionTable();
    ~TranspositionTable();

    void resize(size_t mb);
    void clear();
    void new_search();

    bool probe(uint64_t key, TTEntry& entry) const;
    void store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply, Value static_eval = VALUE_NONE);

    Value score_to_tt(Value score, int ply) const;
    Value score_from_tt(Value score, int ply) const;

    size_t size_mb() const { return size_mb_; }
    int hashfull() const;

private:
    std::vector<TTCluster> table_;
    size_t size_mb_;
    size_t num_clusters_;
    uint8_t generation_;
};

extern TranspositionTable g_tt;

#endif // TT_H
