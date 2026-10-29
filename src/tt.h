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

struct TTEntry {
    uint64_t key;
    uint16_t move_data;
    int16_t score;
    int8_t depth;
    uint8_t flag;
    uint8_t generation;
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
    std::vector<TTEntry> table_;
    size_t size_mb_;
    size_t num_entries_;
    uint8_t generation_;
};

extern TranspositionTable g_tt;

#endif // TT_H
