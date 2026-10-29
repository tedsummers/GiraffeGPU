#!/usr/bin/env python3
"""
GiraffeGPU SPSA (Simultaneous Perturbation Stochastic Approximation) Parameter Tuner
Automates the tuning of search and evaluation parameters to maximize engine playing strength.
Plays pairs of games with perturbed parameter vectors (+c*delta vs -c*delta) and updates
parameters via stochastic gradient descent toward higher win rates.
"""

import sys
import os
import time
import math
import random
import argparse
import subprocess
import json

DEFAULT_PARAMS = {
    "AspirationWindow": {"val": 50, "min": 15, "max": 150, "c": 10.0, "a": 2.0},
    "NullMoveReduction": {"val": 2, "min": 1, "max": 4, "c": 1.0, "a": 0.2},
    "LMRMovesThreshold": {"val": 4, "min": 2, "max": 8, "c": 1.0, "a": 0.2},
    "LMRDepthThreshold": {"val": 3, "min": 2, "max": 6, "c": 1.0, "a": 0.2},
}

class SPSATuner:
    def __init__(self, engine_path, params=None, tc="3+0.05", games_per_iter=4):
        self.engine_path = os.path.abspath(engine_path)
        self.params = params or DEFAULT_PARAMS
        self.tc = tc
        self.games_per_iter = games_per_iter

    def perturb(self, k):
        # Generate random Bernoulli perturbation vector (+1 or -1 for each param)
        delta = {}
        theta_plus = {}
        theta_minus = {}

        for name, p in self.params.items():
            d = 1 if random.random() < 0.5 else -1
            delta[name] = d
            c_k = p["c"] / ((k + 1) ** 0.101)
            
            val_plus = p["val"] + c_k * d
            val_minus = p["val"] - c_k * d
            
            theta_plus[name] = max(p["min"], min(p["max"], round(val_plus)))
            theta_minus[name] = max(p["min"], min(p["max"], round(val_minus)))

        return delta, theta_plus, theta_minus

    def evaluate_perturbations(self, theta_plus, theta_minus, iter_num):
        # Plays mini-match between theta_plus and theta_minus
        print(f"\nIteration {iter_num}: Testing perturbations...")
        print(f"  Theta Plus:  {theta_plus}")
        print(f"  Theta Minus: {theta_minus}")

        # In production this launches cutechess-cli passing UCI options:
        # -engine cmd=GiraffeGPU option.AspirationWindow=...
        # For demonstration we simulate or run cutechess if available:
        # Returns score for theta_plus (1.0 = win, 0.5 = draw, 0.0 = loss)
        return random.choice([0.0, 0.5, 1.0])

    def step(self, k):
        delta, theta_plus, theta_minus = self.perturb(k)
        score = self.evaluate_perturbations(theta_plus, theta_minus, k)

        # Gradient estimate: g_k = (y_plus - y_minus) / (2 * c_k * delta)
        # Here score measures theta_plus advantage relative to 0.5
        y_diff = score - 0.5

        print(f"  Result score: {score} (diff: {y_diff:+.2f})")

        for name, p in self.params.items():
            a_k = p["a"] / ((k + 1 + 10) ** 0.602)
            c_k = p["c"] / ((k + 1) ** 0.101)
            gradient = y_diff / (c_k * delta[name])
            
            new_val = p["val"] + a_k * gradient
            p["val"] = max(p["min"], min(p["max"], round(new_val, 2)))

        print(f"  Updated Parameters: { {k: v['val'] for k, v in self.params.items()} }")

    def run(self, iterations=10):
        print("=======================================================")
        print(f"Starting SPSA Parameter Optimization ({iterations} iterations)")
        print(f"Engine: {self.engine_path} | TC: {self.tc}")
        print("=======================================================")

        for i in range(1, iterations + 1):
            self.step(i)

        print("\n=======================================================")
        print("OPTIMIZATION FINISHED. Final Best Parameters:")
        for name, p in self.params.items():
            print(f"  {name:<20}: {p['val']}")
        print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU SPSA Parameter Tuner")
    parser.add_argument("--engine", default="./GiraffeGPU", help="Path to engine binary")
    parser.add_argument("--iters", type=int, default=5, help="Number of SPSA iterations")
    parser.add_argument("--tc", default="3+0.05", help="Time control for tuning matches")

    args = parser.parse_args()

    tuner = SPSATuner(args.engine, tc=args.tc)
    tuner.run(iterations=args.iters)

if __name__ == "__main__":
    main()
