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

// 16-byte packed Transposition Table entry with 32-bit key verification
struct TTEntry {
    uint32_t key32;         // Upper 32 bits of 64-bit Zobrist key
    uint16_t move_data;     // Best move encoded
    int16_t score;          // Centipawn / mate score
    int8_t depth;           // Search depth
    uint8_t flag;           // Exact, Lowerbound, Upperbound
    uint8_t generation;     // Aging generation
    int8_t eval;            // Static evaluation cache (optional)
    uint32_t padding;       // Alignment to 16 bytes
};

// 2-slot cluster: [0] Depth-preferred, [1] Always-replace / Age-preferred
struct TTCluster {
    TTEntry entries[2];
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
