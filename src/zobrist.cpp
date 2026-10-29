#include "zobrist.h"

namespace Zobrist {

uint64_t PIECES[PIECE_NB][64];
uint64_t CASTLING[16];
uint64_t EP[65];
uint64_t SIDE;

// Deterministic 64-bit PRNG (splitmix64)
static uint64_t splitmix64(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void init() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    uint64_t seed = 0x123456789ABCDEF0ULL;

    for (int p = 0; p < PIECE_NB; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            PIECES[p][sq] = splitmix64(seed);
        }
    }

    for (int c = 0; c < 16; ++c) {
        CASTLING[c] = splitmix64(seed);
    }

    for (int sq = 0; sq < 65; ++sq) {
        EP[sq] = splitmix64(seed);
    }

    SIDE = splitmix64(seed);
}

} // namespace Zobrist
