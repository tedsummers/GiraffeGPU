#!/usr/bin/env python3
"""
GiraffeGPU Neural Weight Trainer
Fine-tunes the 3-layer MLP weights (363 -> 512 -> 64 -> 1) using self-play game datasets
and writes binary format 'GIRAFGPU' network files (eval.net).
"""

import sys
import os
import struct
import math
import json
import random
import argparse
import random

try:
    import numpy as np
    import chess
except ImportError:
    np = None
    chess = None

IN_DIM = 363
L1_DIM = 512
L2_DIM = 64
OUT_DIM = 1
POLICY_DIM = 4096

# The engine format reserves exactly eight bytes for the magic.  The historical
# human-readable name is GIRAFDUAL, whose first eight bytes are written/read.
MAGIC_GPU = b"GIRAFGPU"
MAGIC_DUAL = b"GIRAFDUA"

MAT = [0, 100, 325, 330, 500, 900, 20000]

def _coord(sq, exists):
    return [1.0, (sq & 7) * 0.1429, (sq >> 3) * 0.1429] if exists else [0.0, 0.0, 0.0]

def extract_features(fen):
    """Python mirror of Giraffe::extract_features for feature-aware training."""
    if chess is None:
        raise RuntimeError("python-chess is required for feature-aware training")
    b = chess.Board(fen)
    out = []
    def count(pt, color, maximum):
        return len(b.pieces(pt, color)) / maximum
    out += [count(chess.QUEEN, chess.WHITE, 1), count(chess.ROOK, chess.WHITE, 2),
            count(chess.BISHOP, chess.WHITE, 2), count(chess.KNIGHT, chess.WHITE, 2),
            count(chess.PAWN, chess.WHITE, 8)]
    out += [count(chess.QUEEN, chess.BLACK, 1), count(chess.ROOK, chess.BLACK, 2),
            count(chess.BISHOP, chess.BLACK, 2), count(chess.KNIGHT, chess.BLACK, 2),
            count(chess.PAWN, chess.BLACK, 8)]
    out.append(1.0 if b.turn == chess.WHITE else 0.0)
    for color in (chess.WHITE, chess.BLACK):
        king = next(iter(b.pieces(chess.KING, color)), None)
        out += [(king & 7) * 0.1429 if king is not None else 0.0,
                (king >> 3) * 0.1429 if king is not None else 0.0]
        out += [1.0 if b.has_kingside_castling_rights(color) else 0.0,
                1.0 if b.has_queenside_castling_rights(color) else 0.0]

    attackers = {}
    for sq in chess.SQUARES:
        attackers[(chess.WHITE, sq)] = list(b.attackers(chess.WHITE, sq))
        attackers[(chess.BLACK, sq)] = list(b.attackers(chess.BLACK, sq))
    lva = {}
    num = {}
    ctrl = {}
    for color in (chess.WHITE, chess.BLACK):
        for sq in chess.SQUARES:
            ats = attackers[(color, sq)]
            num[(color, sq)] = len(ats)
            lva[(color, sq)] = min((b.piece_at(x).piece_type for x in ats), default=0,
                                   key=lambda pt: MAT[pt])
            ctrl[(color, sq)] = ((MAT[6] + MAT[6] // 2 - MAT[lva[(color, sq)]]) /
                                 (MAT[6] * 2.0)) if ats else 0.0

    def safe(pt, color, sq):
        opp = lva[(not color, sq)]
        friend = lva[(color, sq)]
        if not opp or MAT[opp] >= MAT[pt]:
            return 1 if not opp or (friend and num[(color, sq)] > 1) else 0
        return 0

    def threat(color, sq, exists):
        return [ctrl[(chess.WHITE, sq)], ctrl[(chess.BLACK, sq)]] if exists else [0.0, 0.0]

    def sliding(sq, pt, color, exists):
        dirs = ([(1, 0), (-1, 0), (0, 1), (0, -1)] if pt == chess.ROOK else
                [(1, 1), (-1, 1), (1, -1), (-1, -1)] if pt == chess.BISHOP else
                [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)] if pt == chess.QUEEN else [])
        safe_moves = 0
        f, r = sq & 7, sq >> 3
        for df, dr in dirs:
            n = 0; nf, nr = f + df, r + dr
            while exists and 0 <= nf < 8 and 0 <= nr < 8:
                n += 1; ns = chess.square(nf, nr)
                safe_moves += safe(pt, color, ns)
                if b.piece_at(ns) is not None: break
                nf += df; nr += dr
            out.append(n / 7.0)
        if pt == chess.KNIGHT:
            for df, dr in ((2,1),(-2,1),(1,2),(-1,2),(-2,-1),(2,-1),(-1,-2),(1,-2)):
                nf, nr = f + df, r + dr
                if exists and 0 <= nf < 8 and 0 <= nr < 8:
                    safe_moves += safe(pt, color, chess.square(nf, nr))
        out.append(safe_moves / 16.0)

    def piece_pair(pt, color):
        pieces = sorted(b.pieces(pt, color))
        slots = [None, None]
        if len(pieces) == 1:
            slots[0 if (pieces[0] & 7) < 4 else 1] = pieces[0]
        elif len(pieces) >= 2:
            slots = pieces[:2]
            if (slots[0] & 7) > (slots[1] & 7): slots.reverse()
        for sq in slots:
            exists = sq is not None
            val = sq if exists else 0
            out.extend(_coord(val, exists)); sliding(val, pt, color, exists); out.extend(threat(color, val, exists))

    for color in (chess.WHITE, chess.BLACK):
        pieces = sorted(b.pieces(chess.PAWN, color)); slots = [None] * 8; rest = []
        for sq in pieces:
            file = sq & 7
            if slots[file] is None: slots[file] = sq
            else: rest.append(sq)
        for sq in rest:
            file = sq & 7
            choices = [i for i, x in enumerate(slots) if x is None]
            slot = min(choices, key=lambda i: abs(file - i)); slots[slot] = sq
        for sq in slots:
            exists = sq is not None; val = sq if exists else 0
            out.extend(_coord(val, exists)); out.extend(threat(color, val, exists))

    for color in (chess.WHITE, chess.BLACK):
        qs = sorted(b.pieces(chess.QUEEN, color)); exists = bool(qs); val = qs[0] if exists else 0
        out.extend(_coord(val, exists)); sliding(val, chess.QUEEN, color, exists); out.extend(threat(color, val, exists))
    for color in (chess.WHITE, chess.BLACK):
        piece_pair(chess.ROOK, color)
        out += [1.0 if b.has_kingside_castling_rights(color) else 0.0,
                1.0 if b.has_queenside_castling_rights(color) else 0.0]
    for color in (chess.WHITE, chess.BLACK): piece_pair(chess.BISHOP, color)
    for color in (chess.WHITE, chess.BLACK): piece_pair(chess.KNIGHT, color)
    for sq in chess.SQUARES: out += [ctrl[(chess.WHITE, sq)], ctrl[(chess.BLACK, sq)]]
    if len(out) != IN_DIM: raise ValueError(f"feature count {len(out)} != {IN_DIM}")
    return out

def uci_to_policy_idx(uci_str):
    if len(uci_str) < 4:
        return None
    f_col = ord(uci_str[0]) - ord('a')
    f_row = int(uci_str[1]) - 1
    t_col = ord(uci_str[2]) - ord('a')
    t_row = int(uci_str[3]) - 1
    from_sq = f_row * 8 + f_col
    to_sq = t_row * 8 + t_col
    if 0 <= from_sq < 64 and 0 <= to_sq < 64:
        return (from_sq << 6) | to_sq
    return None

def init_default_weights():
    w1 = [0.0] * (IN_DIM * L1_DIM)
    b1 = [0.0] * L1_DIM
    w2 = [0.0] * (L1_DIM * L2_DIM)
    b2 = [0.0] * L2_DIM
    w3 = [0.0] * (L2_DIM * OUT_DIM)
    b3 = [0.0] * OUT_DIM
    w_pol = [0.0] * (POLICY_DIM * L2_DIM)
    b_pol = [0.0] * POLICY_DIM

    # Material
    w1[0 * IN_DIM + 0] = 9.0; w1[0 * IN_DIM + 1] = 10.0; w1[0 * IN_DIM + 2] = 6.6; w1[0 * IN_DIM + 3] = 6.4; w1[0 * IN_DIM + 4] = 8.0
    w1[1 * IN_DIM + 5] = 9.0; w1[1 * IN_DIM + 6] = 10.0; w1[1 * IN_DIM + 7] = 6.6; w1[1 * IN_DIM + 8] = 6.4; w1[1 * IN_DIM + 9] = 8.0

    # Layer 2 & 3
    w2[0 * L1_DIM + 0] = 1.0; w2[0 * L1_DIM + 1] = -1.0
    w2[1 * L1_DIM + 0] = -1.0; w2[1 * L1_DIM + 1] = 1.0
    w3[0] = 0.10; w3[1] = -0.10

    # Policy priors for central destination squares
    central_sqs = [27, 28, 35, 36, 26, 34, 29, 37]
    for from_sq in range(64):
        for to_sq in central_sqs:
            b_pol[(from_sq << 6) | to_sq] += 0.40

    def set_prior(from_sq, to_sq, value):
        b_pol[(from_sq << 6) | to_sq] += value

    # Keep the Python trainer's legacy-weight policy exactly aligned with the
    # C++ built-in policy used when loading GIRAFGPU files.
    for from_sq, to_sq, value in [
        (12, 28, 2.5), (11, 27, 2.5), (6, 21, 2.0), (1, 18, 1.8),
        (10, 26, 1.8), (4, 6, 2.0), (4, 2, 1.5),
        (52, 36, 2.5), (51, 35, 2.5), (62, 45, 2.0), (57, 42, 1.8),
        (50, 34, 1.8), (60, 62, 2.0), (60, 58, 1.5),
    ]:
        set_prior(from_sq, to_sq, value)
    for from_sq in range(64):
        for to_sq in range(64):
            delta_rank = (to_sq // 8) - (from_sq // 8)
            idx = (from_sq << 6 | to_sq) * L2_DIM
            if delta_rank > 0:
                w_pol[idx + 0] = 0.04 * delta_rank
            elif delta_rank < 0:
                w_pol[idx + 1] = -0.04 * delta_rank

    return w1, b1, w2, b2, w3, b3, w_pol, b_pol

def load_weights(path):
    if not os.path.exists(path):
        return init_default_weights()
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic == MAGIC_DUAL:
            w1 = list(struct.unpack(f"{IN_DIM * L1_DIM}f", f.read(IN_DIM * L1_DIM * 4)))
            b1 = list(struct.unpack(f"{L1_DIM}f", f.read(L1_DIM * 4)))
            w2 = list(struct.unpack(f"{L1_DIM * L2_DIM}f", f.read(L1_DIM * L2_DIM * 4)))
            b2 = list(struct.unpack(f"{L2_DIM}f", f.read(L2_DIM * 4)))
            w3 = list(struct.unpack(f"{L2_DIM * OUT_DIM}f", f.read(L2_DIM * OUT_DIM * 4)))
            b3 = list(struct.unpack(f"{OUT_DIM}f", f.read(OUT_DIM * 4)))
            w_pol = list(struct.unpack(f"{POLICY_DIM * L2_DIM}f", f.read(POLICY_DIM * L2_DIM * 4)))
            b_pol = list(struct.unpack(f"{POLICY_DIM}f", f.read(POLICY_DIM * 4)))
            return w1, b1, w2, b2, w3, b3, w_pol, b_pol
        elif magic == MAGIC_GPU:
            w1 = list(struct.unpack(f"{IN_DIM * L1_DIM}f", f.read(IN_DIM * L1_DIM * 4)))
            b1 = list(struct.unpack(f"{L1_DIM}f", f.read(L1_DIM * 4)))
            w2 = list(struct.unpack(f"{L1_DIM * L2_DIM}f", f.read(L1_DIM * L2_DIM * 4)))
            b2 = list(struct.unpack(f"{L2_DIM}f", f.read(L2_DIM * 4)))
            w3 = list(struct.unpack(f"{L2_DIM * OUT_DIM}f", f.read(L2_DIM * OUT_DIM * 4)))
            b3 = list(struct.unpack(f"{OUT_DIM}f", f.read(OUT_DIM * 4)))
            _, _, _, _, _, _, w_pol, b_pol = init_default_weights()
            return w1, b1, w2, b2, w3, b3, w_pol, b_pol
        else:
            return init_default_weights()

def save_weights(path, w1, b1, w2, b2, w3, b3, w_pol, b_pol):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC_DUAL)
        f.write(struct.pack(f"{len(w1)}f", *w1))
        f.write(struct.pack(f"{len(b1)}f", *b1))
        f.write(struct.pack(f"{len(w2)}f", *w2))
        f.write(struct.pack(f"{len(b2)}f", *b2))
        f.write(struct.pack(f"{len(w3)}f", *w3))
        f.write(struct.pack(f"{len(b3)}f", *b3))
        f.write(struct.pack(f"{len(w_pol)}f", *w_pol))
        f.write(struct.pack(f"{len(b_pol)}f", *b_pol))

def train_on_dataset(data_path, in_weights_path, out_weights_path, epochs=3, lr=0.00025,
                     policy_lr=0.0, max_output_delta=0.02, seed=1337,
                     outcome_weight=0.7, eval_scale=400.0, holdout=0.2):
    w1, b1, w2, b2, w3, b3, w_pol, b_pol = load_weights(in_weights_path)
    base_policy_bias = list(b_pol)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        save_weights(out_weights_path, w1, b1, w2, b2, w3, b3, w_pol, b_pol)
        return

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                target = (item.get("target_outcome", 0.5) - 0.5) * 2.0
                best_move = item.get("bestmove") or item.get("move")
                pol_idx = uci_to_policy_idx(best_move) if best_move else None
                fen = item.get("fen")
                if fen:
                    # The engine's network output is white-perspective. UCI
                    # search scores and self-play outcomes are stored from the
                    # side-to-move perspective, so Black positions must be
                    # sign-flipped before training the raw network output.
                    white_sign = 1.0 if fen.split()[1] == "w" else -1.0
                    white_outcome = target * white_sign
                    eval_cp = float(item.get("eval_cp", 0.0))
                    white_eval_cp = eval_cp * white_sign
                    eval_target = math.tanh(white_eval_cp / max(1.0, eval_scale))
                    # Search score is a useful low-variance teacher, while the
                    # terminal result keeps the target grounded in game truth.
                    blended_target = (outcome_weight * white_outcome +
                                      (1.0 - outcome_weight) * eval_target)
                    samples.append((fen, blended_target, pol_idx))
            except: pass

    if not samples:
        print("No valid samples found in dataset.")
        save_weights(out_weights_path, w1, b1, w2, b2, w3, b3, w_pol, b_pol)
        return

    print(f"Loaded {len(samples)} training samples from {data_path}")
    if np is None or chess is None:
        raise RuntimeError("numpy and python-chess are required for feature-aware training")
    print(f"Training feature-aware value and bounded policy heads for {epochs} epochs with learning rate {lr}...")

    # Keep training bounded and memory-safe. The full dataset can be millions
    # of positions; feature batches are materialised one batch at a time.
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(samples)
    max_samples = min(len(samples), 200000)
    samples = samples[:max_samples]
    features = np.asarray([extract_features(item[0]) for item in samples], dtype=np.float32)
    targets = np.asarray([item[1] for item in samples], dtype=np.float32)
    policy_indices = [item[2] for item in samples]
    w1n = np.asarray(w1, dtype=np.float32).reshape((L1_DIM, IN_DIM))
    b1n = np.asarray(b1, dtype=np.float32)
    w2n = np.asarray(w2, dtype=np.float32).reshape((L2_DIM, L1_DIM))
    b2n = np.asarray(b2, dtype=np.float32)
    w3n = np.asarray(w3, dtype=np.float32)
    b3n = np.asarray(b3, dtype=np.float32)
    base_w3n = w3n.copy()
    base_b3n = b3n.copy()
    base_policy_bias = np.asarray(b_pol, dtype=np.float32).copy()
    batch_size = 1024
    split = max(1, min(len(features) - 1, int(len(features) * (1.0 - holdout))))
    split_order = np.random.permutation(len(features))
    train_idx = split_order[:split]
    val_idx = split_order[split:]

    def predict(inputs):
        h1 = np.maximum(0.0, inputs @ w1n.T + b1n)
        h2 = np.maximum(0.0, h1 @ w2n.T + b2n)
        return np.tanh(h2 @ w3n + b3n[0])

    initial_val_mse = float(np.mean((predict(features[val_idx]) - targets[val_idx]) ** 2))
    print(f"Training split: {len(train_idx)} train / {len(val_idx)} validation samples")
    print(f"Initial validation MSE: {initial_val_mse:.6f}")

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        policy_updates = 0
        order = np.random.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = features[idx]
            h1 = np.maximum(0.0, x @ w1n.T + b1n)
            h2 = np.maximum(0.0, h1 @ w2n.T + b2n)
            pred = np.tanh(h2 @ w3n + b3n[0])
            err = pred - targets[idx]
            total_loss += float(np.sum(err * err))

            grad = ((1.0 - pred * pred) * err).astype(np.float32)
            # Train the output layer with real position-dependent activations.
            w3n -= (lr * 0.1) * (h2.T @ grad) / len(idx)
            b3n[0] -= (lr * 0.1) * float(np.mean(grad))

            # A self-play batch is noisy. Keep each candidate close to the
            # current master network so one cycle cannot cause a large Elo drop.
            w3n[:] = np.clip(w3n, base_w3n - max_output_delta, base_w3n + max_output_delta)
            b3n[:] = np.clip(b3n, base_b3n - max_output_delta, base_b3n + max_output_delta)

        # Policy bias updates remain deliberately small and bounded.
        for pol_idx in policy_indices:
            if policy_lr > 0.0 and pol_idx is not None and 0 <= pol_idx < POLICY_DIM:
                b_pol[pol_idx] = max(base_policy_bias[pol_idx] - 0.25,
                                     min(base_policy_bias[pol_idx] + 0.25,
                                         b_pol[pol_idx] + policy_lr))
                policy_updates += 1

        mse = total_loss / len(train_idx)
        val_mse = float(np.mean((predict(features[val_idx]) - targets[val_idx]) ** 2))
        print(f"  Epoch {ep}/{epochs}: Train MSE = {mse:.6f} | Validation MSE = {val_mse:.6f} | Policy updates = {policy_updates}")

    save_weights(out_weights_path, w1n.ravel(), b1n, w2n.ravel(), b2n,
                 w3n, b3n, w_pol, b_pol)
    print(f"Dual-Head weights successfully saved to: {out_weights_path}")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Neural Network Weight Trainer")
    parser.add_argument("--data", default="testing/results/selfplay_dataset.jsonl", help="Training dataset (.jsonl)")
    parser.add_argument("--in-weights", default="eval.net", help="Input weights file")
    parser.add_argument("--out-weights", default="eval.net", help="Output weights file")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.00025,
                        help="Conservative value-head learning rate")
    parser.add_argument("--max-output-delta", type=float, default=0.02,
                        help="Maximum per-cycle change to value output weights")
    parser.add_argument("--seed", type=int, default=1337, help="Training shuffle seed")
    parser.add_argument("--outcome-weight", type=float, default=0.7,
                        help="Weight of final game result in blended target")
    parser.add_argument("--eval-scale", type=float, default=400.0,
                        help="Centipawn scale for normalized search targets")
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="Fraction reserved for validation")
    parser.add_argument("--policy-lr", type=float, default=0.0,
                        help="Global policy-bias learning rate; 0 disables unsafe context-free updates")

    args = parser.parse_args()

    train_on_dataset(args.data, args.in_weights, args.out_weights,
                     epochs=args.epochs, lr=args.lr, policy_lr=args.policy_lr,
                     max_output_delta=args.max_output_delta, seed=args.seed,
                     outcome_weight=max(0.0, min(1.0, args.outcome_weight)),
                     eval_scale=max(1.0, args.eval_scale),
                     holdout=max(0.05, min(0.5, args.holdout)))

if __name__ == "__main__":
    main()
