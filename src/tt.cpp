#include "tt.h"
#include <cstring>
#include <algorithm>

TranspositionTable g_tt;

TranspositionTable::TranspositionTable()
    : size_mb_(0)
    , num_clusters_(0)
    , generation_(0)
{
    resize(64); // Default 64 MB
}

TranspositionTable::~TranspositionTable() = default;

void TranspositionTable::resize(size_t mb) {
    size_mb_ = std::max(size_t(1), std::min(mb, size_t(16384)));
    size_t bytes = size_mb_ * 1024 * 1024;
    num_clusters_ = bytes / sizeof(TTCluster);
    table_.assign(num_clusters_, TTCluster{});
    clear();
}

void TranspositionTable::clear() {
    std::fill(table_.begin(), table_.end(), TTCluster{});
    generation_ = 0;
}

void TranspositionTable::new_search() {
    generation_++;
}

bool TranspositionTable::probe(uint64_t key, TTEntry& entry) const {
    if (num_clusters_ == 0) return false;
    size_t idx = key % num_clusters_;
    const TTCluster& cluster = table_[idx];

    for (const TTEntry& candidate : cluster.entries) {
        if (candidate.flag != TT_NONE && candidate.key == key) {
            entry = candidate;
            return true;
        }
    }
    return false;
}

void TranspositionTable::store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply) {
    if (num_clusters_ == 0) return;
    size_t idx = key % num_clusters_;
    TTCluster& cluster = table_[idx];

    int target_slot = -1;

    // 1. If key already exists in one of the slots, update that slot
    for (int i = 0; i < 4; ++i) {
        if (cluster.entries[i].key == key && cluster.entries[i].flag != TT_NONE) {
            target_slot = i;
            break;
        }
    }

    if (target_slot != -1) {
        TTEntry& e = cluster.entries[target_slot];
        if (depth >= e.depth - 2 || e.generation != generation_ || flag == TT_EXACT) {
            e.key = key;
            if (!move.is_null()) e.move_data = move.data;
            e.score = static_cast<int16_t>(score_to_tt(score, ply));
            e.depth = static_cast<int8_t>(depth);
            e.flag = flag;
            e.generation = generation_;
        } else if (!move.is_null() && e.move_data == 0) {
            e.move_data = move.data;
        }
        return;
    }

    // Replace the lowest-quality entry.  Old generations are deliberately
    // discounted, so a shallow result from the current search cannot evict a
    // useful deep result while stale entries do not live forever.
    int worst_quality = 1 << 30;
    for (int i = 0; i < 4; ++i) {
        const TTEntry& e = cluster.entries[i];
        if (e.flag == TT_NONE) {
            target_slot = i;
            break;
        }
        const int age = static_cast<uint8_t>(generation_ - e.generation);
        const int quality = static_cast<int>(e.depth) - age * 8 + (e.flag == TT_EXACT ? 4 : 0);
        if (quality < worst_quality) {
            worst_quality = quality;
            target_slot = i;
        }
    }

    TTEntry& e = cluster.entries[target_slot];
    e.key = key;
    e.move_data = move.data;
    e.score = static_cast<int16_t>(score_to_tt(score, ply));
    e.depth = static_cast<int8_t>(depth);
    e.flag = flag;
    e.generation = generation_;
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
    if (num_clusters_ == 0) return 0;
    int sampled = 0;
    int filled = 0;
    for (size_t i = 0; i < 1000 && i < num_clusters_; ++i) {
        for (int s = 0; s < 4; ++s) {
            sampled++;
            if (table_[i].entries[s].flag != TT_NONE && table_[i].entries[s].generation == generation_) {
                filled++;
            }
        }
    }
    return (filled * 1000) / sampled;
}
