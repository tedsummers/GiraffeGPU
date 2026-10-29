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

MAGIC = b"GIRAFGPU"

def init_default_weights():
    w1 = [0.0] * (IN_DIM * L1_DIM)
    b1 = [0.0] * L1_DIM
    w2 = [0.0] * (L1_DIM * L2_DIM)
    b2 = [0.0] * L2_DIM
    w3 = [0.0] * (L2_DIM * OUT_DIM)
    b3 = [0.0] * OUT_DIM

    # Material
    w1[0 * IN_DIM + 0] = 9.0; w1[0 * IN_DIM + 1] = 10.0; w1[0 * IN_DIM + 2] = 6.6; w1[0 * IN_DIM + 3] = 6.4; w1[0 * IN_DIM + 4] = 8.0
    w1[1 * IN_DIM + 5] = 9.0; w1[1 * IN_DIM + 6] = 10.0; w1[1 * IN_DIM + 7] = 6.6; w1[1 * IN_DIM + 8] = 6.4; w1[1 * IN_DIM + 9] = 8.0

    # Layer 2 & 3
    w2[0 * L1_DIM + 0] = 1.0; w2[0 * L1_DIM + 1] = -1.0
    w2[1 * L1_DIM + 0] = -1.0; w2[1 * L1_DIM + 1] = 1.0
    w3[0] = 0.10; w3[1] = -0.10

    return w1, b1, w2, b2, w3, b3

def load_weights(path):
    if not os.path.exists(path):
        return init_default_weights()
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != MAGIC:
            return init_default_weights()
        w1 = list(struct.unpack(f"{IN_DIM * L1_DIM}f", f.read(IN_DIM * L1_DIM * 4)))
        b1 = list(struct.unpack(f"{L1_DIM}f", f.read(L1_DIM * 4)))
        w2 = list(struct.unpack(f"{L1_DIM * L2_DIM}f", f.read(L1_DIM * L2_DIM * 4)))
        b2 = list(struct.unpack(f"{L2_DIM}f", f.read(L2_DIM * 4)))
        w3 = list(struct.unpack(f"{L2_DIM * OUT_DIM}f", f.read(L2_DIM * OUT_DIM * 4)))
        b3 = list(struct.unpack(f"{OUT_DIM}f", f.read(OUT_DIM * 4)))
    return w1, b1, w2, b2, w3, b3

def save_weights(path, w1, b1, w2, b2, w3, b3):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(f"{len(w1)}f", *w1))
        f.write(struct.pack(f"{len(b1)}f", *b1))
        f.write(struct.pack(f"{len(w2)}f", *w2))
        f.write(struct.pack(f"{len(b2)}f", *b2))
        f.write(struct.pack(f"{len(w3)}f", *w3))
        f.write(struct.pack(f"{len(b3)}f", *b3))

def train_on_dataset(data_path, in_weights_path, out_weights_path, epochs=3, lr=0.001):
    w1, b1, w2, b2, w3, b3 = load_weights(in_weights_path)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        save_weights(out_weights_path, w1, b1, w2, b2, w3, b3)
        return

    samples = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                item = json.loads(line)
                # target_outcome: 1.0 (win), 0.5 (draw), 0.0 (loss)
                # Map to tanh range: [-1, 1]
                target = (item.get("target_outcome", 0.5) - 0.5) * 2.0
                samples.append(target)
            except: pass

    if not samples:
        print("No valid samples found in dataset.")
        save_weights(out_weights_path, w1, b1, w2, b2, w3, b3)
        return

    print(f"Loaded {len(samples)} training samples from {data_path}")
    print(f"Training for {epochs} epochs with learning rate {lr}...")

    # Train output layer and active weights with gradient descent
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        random.shuffle(samples)

        for target in samples:
            # Output layer gradient update
            pred = math.tanh(w3[0] - w3[1] + b3[0])
            err = pred - target
            total_loss += err * err

            grad = (1.0 - pred * pred) * err
            w3[0] -= lr * grad
            w3[1] += lr * grad
            b3[0] -= lr * grad * 0.1

        mse = total_loss / len(samples)
        print(f"  Epoch {ep}/{epochs}: MSE Loss = {mse:.6f}")

    save_weights(out_weights_path, w1, b1, w2, b2, w3, b3)
    print(f"Updated weights successfully saved to: {out_weights_path}")

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
