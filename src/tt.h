#ifndef TT_H
#define TT_H

#include "types.h"
#include <vector>
#include <cstdint>

enum TTFlag : uint8_t {
    TT_NONE = 0,
    TT_EXACT = 1,
    TT_LOWERBOUND = 2,
    TT_UPPERBOUND = 3
};

// A compact transposition-table entry.  Keep the complete key: a 32-bit
// verification key is too collision-prone for long searches.
struct TTEntry {
    uint64_t key;           // Full Zobrist key
    uint16_t move_data;     // Best move encoded
    int16_t score;          // Centipawn / mate score
    int8_t depth;           // Search depth
    uint8_t flag;           // Exact, Lowerbound, Upperbound
    uint8_t generation;     // Aging generation
    int8_t eval;            // Static evaluation cache (optional)
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
    void store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply);

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
