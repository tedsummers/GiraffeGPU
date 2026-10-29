#ifndef GIRAFFE_FEATURES_H
#define GIRAFFE_FEATURES_H

#include "types.h"
#include "board.h"
#include <vector>

namespace Giraffe {

constexpr int NUM_FEATURES = 363;

struct AttackMaps {
    PieceType whiteLVA[64];
    PieceType blackLVA[64];
    uint8_t whiteNumAtk[64];
    uint8_t blackNumAtk[64];
    float whiteCtrl[64];
    float blackCtrl[64];

    bool is_safe(PieceType pt, Square sq) const;
};

void compute_attack_maps(const Board& board, AttackMaps& maps);

// Extract the 363 Giraffe features for a position into an output array
void extract_features(const Board& board, float* out_features);

} // namespace Giraffe

#endif // GIRAFFE_FEATURES_H
