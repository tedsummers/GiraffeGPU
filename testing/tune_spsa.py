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

from match_sprt import run_match

DEFAULT_PARAMS = {
    "AspirationWindow": {"val": 50, "min": 15, "max": 150, "c": 10.0, "a": 2.0},
    "NullMoveReduction": {"val": 2, "min": 1, "max": 4, "c": 1.0, "a": 0.2},
    "LMRMovesThreshold": {"val": 4, "min": 2, "max": 8, "c": 1.0, "a": 0.2},
    "LMRDepthThreshold": {"val": 3, "min": 2, "max": 6, "c": 1.0, "a": 0.2},
    "ProbCutMargin": {"val": 180, "min": 80, "max": 400, "c": 20.0, "a": 2.0},
}

class SPSATuner:
    def __init__(self, engine_path, params=None, tc="3+0.05", games_per_iter=4):
        self.engine_path = os.path.abspath(engine_path)
        self.params = params or DEFAULT_PARAMS
        self.tc = tc
        self.games_per_iter = games_per_iter
        self.repo_dir = os.path.dirname(self.engine_path)
        self.results_dir = os.path.join(self.repo_dir, "testing", "results")
        self.best_params = None
        self.best_score = -1.0
        self.best_iteration = 0

    @staticmethod
    def option_string(theta):
        return ",".join(f"{name}={int(value)}" for name, value in theta.items())

    def save_best(self):
        if self.best_params is None:
            return
        os.makedirs(self.results_dir, exist_ok=True)
        payload = {
            "iteration": self.best_iteration,
            "score": self.best_score,
            "parameters": self.best_params,
            "uci_options": self.option_string(self.best_params),
        }
        with open(os.path.join(self.results_dir, "best_search_params.json"), "w") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(self.results_dir, "best_search_options.txt"), "w") as f:
            f.write(payload["uci_options"] + "\n")

    def train_network(self, selfplay_games, epochs, movetime):
        data_path = os.path.join(self.results_dir, "selfplay_dataset.jsonl")
        has_fen = False
        if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
            with open(data_path, "r") as f:
                for _ in range(200):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        if json.loads(line).get("fen"):
                            has_fen = True
                            break
                    except json.JSONDecodeError:
                        continue
        if not has_fen:
            # Preserve legacy datasets; feature-aware training must never mix
            # silently with records that lack a position representation.
            data_path = os.path.join(self.results_dir, "feature_selfplay_dataset.jsonl")

        if not os.path.exists(data_path) or os.path.getsize(data_path) == 0:
            print(f"\nNo self-play dataset found; generating {selfplay_games} games...")
            subprocess.run([
                sys.executable, os.path.join(self.repo_dir, "testing", "self_play.py"),
                "--engine", self.engine_path,
                "--games", str(selfplay_games),
                "--movetime", str(movetime),
                "--engine-options", self.option_string(self.best_params),
                "--output", data_path,
            ], cwd=self.repo_dir, check=True)

        output_path = os.path.join(self.results_dir, "candidate.net")
        print(f"\nTraining neural weights from {data_path}...")
        subprocess.run([
            sys.executable, os.path.join(self.repo_dir, "testing", "train_weights.py"),
            "--data", data_path,
            "--in-weights", os.path.join(self.repo_dir, "eval.net"),
            "--out-weights", output_path,
            "--epochs", str(epochs),
        ], cwd=self.repo_dir, check=True)
        print(f"Candidate weights saved to: {output_path}")
        return output_path

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
        print(f"\nIteration {iter_num}: Testing perturbations...")
        print(f"  Theta Plus:  {theta_plus}")
        print(f"  Theta Minus: {theta_minus}")

        result = run_match(
            engine1_path=self.engine_path,
            engine2_path=self.engine_path,
            games=self.games_per_iter,
            tc=self.tc,
            book_path=os.path.join(os.path.dirname(os.path.dirname(self.engine_path)),
                                   "testing", "books", "openings.epd"),
            concurrency=1,
            engine1_options=self.option_string(theta_plus),
            engine2_options=self.option_string(theta_minus),
        )
        if not result or result.get("total", 0) == 0:
            raise RuntimeError("SPSA match produced no games; is cutechess-cli installed?")

        # Score from the plus perturbation's perspective, in [0, 1].
        score = result["stats"]["score_pct"] / 100.0
        # The better side of this A/B match is the observed candidate for this
        # iteration. Keep the best observed vector, not merely the final SPSA
        # center, since the final update can be noisy.
        candidate = theta_plus if score >= 0.5 else theta_minus
        candidate_score = score if score >= 0.5 else 1.0 - score
        if candidate_score > self.best_score:
            self.best_score = candidate_score
            self.best_params = dict(candidate)
            self.best_iteration = iter_num
            self.save_best()

        with open(os.path.join(self.results_dir, "spsa_history.jsonl"), "a") as f:
            f.write(json.dumps({"iteration": iter_num, "plus": theta_plus,
                                "minus": theta_minus, "result": result}) + "\n")
        return score

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

    def run(self, iterations=10, train_weights=True, selfplay_games=25,
            epochs=5, selfplay_movetime=80):
        print("=======================================================")
        print(f"Starting SPSA Parameter Optimization ({iterations} iterations)")
        print(f"Engine: {self.engine_path} | TC: {self.tc}")
        print("=======================================================")

        for i in range(1, iterations + 1):
            self.step(i)

        self.save_best()

        print("\n=======================================================")
        print("OPTIMIZATION FINISHED. Final Best Parameters:")
        for name, p in self.params.items():
            print(f"  {name:<20}: {p['val']}")
        if self.best_params:
            print(f"Best observed iteration: {self.best_iteration} (score {self.best_score:.3f})")
            print(f"Best UCI options: {self.option_string(self.best_params)}")
            print(f"Saved search parameters to: {os.path.join(self.results_dir, 'best_search_params.json')}")
        if train_weights:
            self.train_network(selfplay_games, epochs, selfplay_movetime)
        print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU SPSA Parameter Tuner")
    parser.add_argument("--engine", default="./GiraffeGPU", help="Path to engine binary")
    parser.add_argument("--iters", type=int, default=5, help="Number of SPSA iterations")
    parser.add_argument("--tc", default="3+0.05", help="Time control for tuning matches")
    parser.add_argument("--games", type=int, default=8, help="Games per SPSA perturbation match")
    parser.add_argument("--selfplay-games", type=int, default=25, help="Games to generate if no dataset exists")
    parser.add_argument("--selfplay-movetime", type=int, default=80, help="Self-play movetime in milliseconds")
    parser.add_argument("--epochs", type=int, default=5, help="Neural-weight training epochs")
    parser.add_argument("--no-train-weights", action="store_true", help="Only tune/search-save; skip neural training")

    args = parser.parse_args()

    tuner = SPSATuner(args.engine, tc=args.tc, games_per_iter=max(2, args.games))
    tuner.run(
        iterations=args.iters,
        train_weights=not args.no_train_weights,
        selfplay_games=args.selfplay_games,
        epochs=args.epochs,
        selfplay_movetime=args.selfplay_movetime,
    )

if __name__ == "__main__":
    main()
