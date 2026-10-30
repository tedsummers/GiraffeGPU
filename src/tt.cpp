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
    uint32_t key32 = static_cast<uint32_t>(key >> 32);

    const TTCluster& cluster = table_[idx];

    // Check slot 0 (depth-preferred) then slot 1 (always-replace)
    for (int i = 0; i < 2; ++i) {
        if (cluster.entries[i].flag != TT_NONE && cluster.entries[i].key32 == key32) {
            entry = cluster.entries[i];
            return true;
        }
    }
    return false;
}

void TranspositionTable::store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply) {
    if (num_clusters_ == 0) return;
    size_t idx = key % num_clusters_;
    uint32_t key32 = static_cast<uint32_t>(key >> 32);

    TTCluster& cluster = table_[idx];

    int target_slot = -1;

    // 1. If key already exists in one of the slots, update that slot
    for (int i = 0; i < 2; ++i) {
        if (cluster.entries[i].key32 == key32 && cluster.entries[i].flag != TT_NONE) {
            target_slot = i;
            break;
        }
    }

    if (target_slot != -1) {
        TTEntry& e = cluster.entries[target_slot];
        if (depth >= e.depth - 2 || e.generation != generation_) {
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

    // 2. Otherwise pick replacement:
    // Slot 0 is reserved for deeper entries or stale entries from past moves.
    // Slot 1 is always replaced (or replaced if older).
    TTEntry& s0 = cluster.entries[0];
    TTEntry& s1 = cluster.entries[1];

    if (s0.flag == TT_NONE) {
        target_slot = 0;
    } else if (s1.flag == TT_NONE) {
        target_slot = 1;
    } else if (s0.generation != generation_) {
        target_slot = 0; // Age-out slot 0
    } else if (depth > s0.depth) {
        // Demote previous slot 0 to slot 1 if deeper than current slot 1
        if (s0.depth > s1.depth || s1.generation != generation_) {
            s1 = s0;
        }
        target_slot = 0;
    } else {
        target_slot = 1;
    }

    TTEntry& e = cluster.entries[target_slot];
    e.key32 = key32;
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
        for (int s = 0; s < 2; ++s) {
            sampled++;
            if (table_[i].entries[s].flag != TT_NONE && table_[i].entries[s].generation == generation_) {
                filled++;
            }
        }
    }
    return (filled * 1000) / sampled;
}
