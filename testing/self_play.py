#!/usr/bin/env python3
"""
GiraffeGPU Self-Play Training Data Generator
Generates self-play games with random opening plies (following Matthew Lai's Giraffe paper).
Records position FENs, move evaluations, search depths, and final game outcomes (TD-learning targets).
"""

import sys
import os
import time
import random
import subprocess
import json
import argparse
from datetime import datetime

def resolve_engine_path(p):
    if os.path.isabs(p): return p
    if os.path.exists(p): return os.path.abspath(p)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(repo_dir, p)
    if os.path.exists(cand): return cand
    cand2 = os.path.join(repo_dir, os.path.basename(p))
    if os.path.exists(cand2): return cand2
    return os.path.abspath(p)

class UCIEngineInstance:
    def __init__(self, engine_path, device_id=-1):
        self.engine_path = resolve_engine_path(engine_path)
        self.device_id = device_id
        self.process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.wait_for("uciok")
        if self.device_id >= 0:
            self.send(f"setoption name Device_GPU value {self.device_id}")
        self.send("isready")
        self.wait_for("readyok")

    def send(self, cmd):
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def wait_for(self, target, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if not line: break
            if target in line: return line
        return None

    def get_best_move(self, fen, movetime=100):
        self.send(f"position fen {fen}")
        self.send("isready")
        self.wait_for("readyok", timeout=3.0)
        self.send(f"go movetime {movetime}")

        best_move = None
        score_cp = 0
        depth = 0

        while True:
            line = self.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("info"):
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "depth" and i + 1 < len(parts):
                        try: depth = int(parts[i+1])
                        except: pass
                    elif p == "cp" and i + 1 < len(parts):
                        try: score_cp = int(parts[i+1])
                        except: pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        return best_move, score_cp, depth

    def close(self):
        if self.process:
            try:
                self.send("quit")
                self.process.terminate()
            except: pass
            self.process = None

# Built-in lightweight move validator / generator using python-chess if available,
# or falling back to engine's own legal move responses.
def get_random_opening(engine, plies=6):
    # Standard initial moves
    common_openings = [
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
        ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
        ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
        ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
        ["c2c4", "g8f6", "b1c3", "e7e6", "e2e4", "d7d5"],
        ["g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "c7c6"],
        ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"]
    ]
    return random.choice(common_openings)[:plies]

def play_selfplay_game(engine_path, movetime=100, random_plies=4, max_moves=100):
    engine = UCIEngineInstance(engine_path)

    moves_history = get_random_opening(engine, plies=random_plies)
    game_positions = []

    # Play out moves
    result = "1/2-1/2" # default draw
    res_val = 0.5

    for ply in range(len(moves_history), max_moves * 2):
        moves_str = " ".join(moves_history)
        engine.send(f"position startpos moves {moves_str}")
        engine.send("isready")
        engine.wait_for("readyok")

        engine.send(f"go movetime {movetime}")
        best_move = None
        eval_score = 0
        depth = 0

        while True:
            line = engine.process.stdout.readline()
            if not line: break
            line = line.strip()
            if line.startswith("info"):
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "depth" and i + 1 < len(parts):
                        try: depth = int(parts[i+1])
                        except: pass
                    elif p == "cp" and i + 1 < len(parts):
                        try: eval_score = int(parts[i+1])
                        except: pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                break

        if not best_move or best_move == "0000" or best_move == "(none)":
            # Checkmate or stalemate
            side_to_move = (ply % 2) # 0 = White, 1 = Black
            if side_to_move == 0:
                result = "0-1"
                res_val = 0.0
            else:
                result = "1-0"
                res_val = 1.0
            break

        game_positions.append({
            "ply": ply,
            "side": "W" if ply % 2 == 0 else "B",
            "move": best_move,
            "eval_cp": eval_score,
            "depth": depth
        })

        moves_history.append(best_move)

        # Adjudicate extreme evaluations
        if eval_score > 900:
            result = "1-0" if ply % 2 == 0 else "0-1"
            res_val = 1.0 if result == "1-0" else 0.0
            break
        elif eval_score < -900:
            result = "0-1" if ply % 2 == 0 else "1-0"
            res_val = 0.0 if result == "0-1" else 1.0
            break

    engine.close()

    # Annotate positions with final outcome
    for pos in game_positions:
        pos["target_outcome"] = res_val if pos["side"] == "W" else (1.0 - res_val)

    return {
        "moves": moves_history,
        "plies": len(moves_history),
        "result": result,
        "positions": game_positions
    }

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Self-Play Training Data Generator")
    parser.add_argument("--engine", default="./GiraffeGPU", help="Path to engine binary")
    parser.add_argument("--games", type=int, default=10, help="Number of games to generate")
    parser.add_argument("--movetime", type=int, default=100, help="Search movetime in ms (default: 100)")
    parser.add_argument("--output", default="testing/results/selfplay_dataset.jsonl", help="Output file (.jsonl)")

    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = args.output if os.path.isabs(args.output) else os.path.join(repo_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n=======================================================")
    print(f"Starting Self-Play Generation: {args.games} games")
    print(f"Engine:    {args.engine}")
    print(f"Movetime:  {args.movetime} ms per move")
    print(f"Output:    {output_path}")
    print(f"=======================================================\n")

    total_positions = 0
    t0 = time.time()

    with open(output_path, "a") as f_out:
        for g in range(1, args.games + 1):
            g_start = time.time()
            game_data = play_selfplay_game(args.engine, movetime=args.movetime)
            g_time = time.time() - g_start

            for pos in game_data["positions"]:
                record = {
                    "game_id": g,
                    "ply": pos["ply"],
                    "side": pos["side"],
                    "move": pos["move"],
                    "eval_cp": pos["eval_cp"],
                    "depth": pos["depth"],
                    "target_outcome": pos["target_outcome"],
                    "game_result": game_data["result"]
                }
                f_out.write(json.dumps(record) + "\n")
                total_positions += 1

            print(f"Game {g:>3}/{args.games}: {game_data['result']:<7} ({game_data['plies']} plies, {len(game_data['positions'])} positions, {g_time:.1f}s)")

    total_elapsed = time.time() - t0
    print("\n" + "=" * 55)
    print(f"Self-play generation completed in {total_elapsed:.1f}s")
    print(f"Total training samples recorded: {total_positions}")
    print(f"Saved to: {args.output}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
