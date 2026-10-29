#include "tt.h"
#include <cstring>
#include <algorithm>

TranspositionTable g_tt;

TranspositionTable::TranspositionTable()
    : size_mb_(0)
    , num_entries_(0)
    , generation_(0)
{
    resize(64); // Default 64 MB
}

TranspositionTable::~TranspositionTable() = default;

void TranspositionTable::resize(size_t mb) {
    size_mb_ = std::max(size_t(1), std::min(mb, size_t(8192)));
    size_t bytes = size_mb_ * 1024 * 1024;
    num_entries_ = bytes / sizeof(TTEntry);
    table_.assign(num_entries_, TTEntry{0, 0, 0, 0, TT_NONE, 0});
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTEntry{0, 0, 0, 0, TT_NONE, 0});
    generation_ = 0;
}

void TranspositionTable::new_search() {
    generation_++;
}

bool TranspositionTable::probe(uint64_t key, TTEntry& entry) const {
    if (num_entries_ == 0) return false;
    size_t idx = key % num_entries_;
    const TTEntry& e = table_[idx];
    if (e.key == key) {
        entry = e;
        return true;
    }
    return false;
}

void TranspositionTable::store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply) {
    if (num_entries_ == 0) return;
    size_t idx = key % num_entries_;
    TTEntry& e = table_[idx];

    // Replacement scheme:
    // Replace if empty, different key, deeper depth, or older generation
    if (e.key != key || e.flag == TT_NONE || depth >= e.depth || e.generation != generation_) {
        e.key = key;
        if (!move.is_null() || e.key != key) {
            e.move_data = move.data;
        }
        e.score = static_cast<int16_t>(score_to_tt(score, ply));
        e.depth = static_cast<int8_t>(depth);
        e.flag = flag;
        e.generation = generation_;
    } else if (e.key == key && !move.is_null()) {
        e.move_data = move.data;
    }
}

Value TranspositionTable::score_to_tt(Value score, int ply) const {
    if (score >= VALUE_MATE_THRESHOLD) return score + ply;
    if (score <= -VALUE_MATE_THRESHOLD) return score - ply;
    return score;
}

Value TranspositionTable::score_from_tt(Value score, int ply) const {
    if (score >= VALUE_MATE_THRESHOLD) return score - ply;
    if (score <= -VALUE_MATE_THRESHOLD) return score + ply;
    return score;
}

int TranspositionTable::hashfull() const {
    if (num_entries_ == 0) return 0;
    int sampled = 0;
    int filled = 0;
    for (size_t i = 0; i < 1000 && i < num_entries_; ++i) {
        sampled++;
        if (table_[i].flag != TT_NONE && table_[i].generation == generation_) {
            filled++;
        }
    }
    return (filled * 1000) / sampled;
}
