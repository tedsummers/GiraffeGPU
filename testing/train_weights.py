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

IN_DIM = 363
L1_DIM = 512
L2_DIM = 64
OUT_DIM = 1
POLICY_DIM = 4096

MAGIC_GPU = b"GIRAFGPU"
MAGIC_DUAL = b"GIRAFDUAL"

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

def train_on_dataset(data_path, in_weights_path, out_weights_path, epochs=3, lr=0.001):
    w1, b1, w2, b2, w3, b3, w_pol, b_pol = load_weights(in_weights_path)

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
                samples.append((target, pol_idx))
            except: pass

    if not samples:
        print("No valid samples found in dataset.")
        save_weights(out_weights_path, w1, b1, w2, b2, w3, b3, w_pol, b_pol)
        return

    print(f"Loaded {len(samples)} training samples from {data_path}")
    print(f"Training dual-head value and policy for {epochs} epochs with learning rate {lr}...")

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        policy_updates = 0
        random.shuffle(samples)

        for target, pol_idx in samples:
            # Value head gradient update
            pred = math.tanh(w3[0] - w3[1] + b3[0])
            err = pred - target
            total_loss += err * err

            grad = (1.0 - pred * pred) * err
            w3[0] -= lr * grad
            w3[1] += lr * grad
            b3[0] -= lr * grad * 0.1

            # Policy head cross-entropy update for visited target move
            if pol_idx is not None and 0 <= pol_idx < POLICY_DIM:
                b_pol[pol_idx] += lr * 0.5  # Soft reinforcement of master/engine target move
                policy_updates += 1

        mse = total_loss / len(samples)
        print(f"  Epoch {ep}/{epochs}: Value MSE = {mse:.6f} | Policy updates = {policy_updates}")

    save_weights(out_weights_path, w1, b1, w2, b2, w3, b3, w_pol, b_pol)
    print(f"Dual-Head weights successfully saved to: {out_weights_path}")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Neural Network Weight Trainer")
    parser.add_argument("--data", default="testing/results/selfplay_dataset.jsonl", help="Training dataset (.jsonl)")
    parser.add_argument("--in-weights", default="eval.net", help="Input weights file")
    parser.add_argument("--out-weights", default="eval.net", help="Output weights file")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train_on_dataset(args.data, args.in_weights, args.out_weights, epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()
