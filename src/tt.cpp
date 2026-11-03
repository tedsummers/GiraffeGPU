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
    generation_ = (generation_ + 1) & 0x3F;
}

static inline size_t fast_cluster_index(uint64_t key, size_t num_clusters) {
#if defined(__SIZEOF_INT128__)
    return static_cast<size_t>((static_cast<__uint128_t>(key) * static_cast<__uint128_t>(num_clusters)) >> 64);
#else
    return key % num_clusters;
#endif
}

bool TranspositionTable::probe(uint64_t key, TTEntry& entry) const {
    if (num_clusters_ == 0) return false;
    size_t idx = fast_cluster_index(key, num_clusters_);
    const TTCluster& cluster = table_[idx];

    for (const TTEntry& candidate : cluster.entries) {
        uint64_t data = candidate.data;
        uint64_t k = candidate.key ^ data;
        if (k == key) {
            TTEntry decoded;
            decoded.key = key;
            decoded.data = data;
            if (decoded.flag != TT_NONE) {
                entry = decoded;
                return true;
            }
        }
    }
    return false;
}

void TranspositionTable::store(uint64_t key, Move move, Value score, int depth, TTFlag flag, int ply, Value static_eval) {
    if (num_clusters_ == 0) return;
    size_t idx = fast_cluster_index(key, num_clusters_);
    TTCluster& cluster = table_[idx];

    int target_slot = -1;
    TTEntry existing{};
    existing.eval = static_cast<int16_t>(VALUE_NONE);

    // 1. If key already exists in one of the slots, update that slot
    for (int i = 0; i < 4; ++i) {
        uint64_t data = cluster.entries[i].data;
        uint64_t k = cluster.entries[i].key ^ data;
        if (k == key) {
            TTEntry decoded;
            decoded.key = key;
            decoded.data = data;
            if (decoded.flag != TT_NONE) {
                target_slot = i;
                existing = decoded;
                break;
            }
        }
    }

    if (target_slot != -1) {
        if (depth >= existing.depth - 2 || existing.generation != generation_ || flag == TT_EXACT) {
            TTEntry updated{};
            updated.move_data = !move.is_null() ? move.data : existing.move_data;
            updated.score = static_cast<int16_t>(score_to_tt(score, ply));
            updated.eval = (static_eval != VALUE_NONE) ? static_cast<int16_t>(static_eval) : existing.eval;
            updated.depth = static_cast<int8_t>(depth);
            updated.flag = flag;
            updated.generation = generation_;
            cluster.entries[target_slot].key = key ^ updated.data;
            cluster.entries[target_slot].data = updated.data;
        } else if ((!move.is_null() && existing.move_data == 0) ||
                   (static_eval != VALUE_NONE && existing.eval == static_cast<int16_t>(VALUE_NONE))) {
            if (!move.is_null() && existing.move_data == 0) existing.move_data = move.data;
            if (static_eval != VALUE_NONE && existing.eval == static_cast<int16_t>(VALUE_NONE)) {
                existing.eval = static_cast<int16_t>(static_eval);
            }
            cluster.entries[target_slot].key = key ^ existing.data;
            cluster.entries[target_slot].data = existing.data;
        }
        return;
    }

    // Replace the lowest-quality entry. Old generations are deliberately
    // discounted, so a shallow result from the current search cannot evict a
    // useful deep result while stale entries do not live forever.
    int worst_quality = 1 << 30;
    target_slot = 0;
    for (int i = 0; i < 4; ++i) {
        TTEntry e;
        e.data = cluster.entries[i].data;
        if (e.flag == TT_NONE) {
            target_slot = i;
            break;
        }
        const int age = (generation_ - e.generation) & 0x3F;
        const int quality = static_cast<int>(e.depth) - age * 8 + (e.flag == TT_EXACT ? 4 : 0);
        if (quality < worst_quality) {
            worst_quality = quality;
            target_slot = i;
        }
    }

    TTEntry updated{};
    updated.move_data = move.data;
    updated.score = static_cast<int16_t>(score_to_tt(score, ply));
    updated.eval = static_cast<int16_t>(static_eval);
    updated.depth = static_cast<int8_t>(depth);
    updated.flag = flag;
    updated.generation = generation_;
    cluster.entries[target_slot].key = key ^ updated.data;
    cluster.entries[target_slot].data = updated.data;
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
