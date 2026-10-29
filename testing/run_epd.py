#!/usr/bin/env python3
"""
GiraffeGPU EPD Test Suite Runner
Evaluates chess engine tactical and positional solving strength on standardized EPD benchmarks
(e.g., Bratko-Kopec Test, Win At Chess, custom tactical suites).
Supports single engine evaluation or head-to-head comparison between two engine builds.
"""

import sys
import os
import time
import argparse
import subprocess
import json
import re
from datetime import datetime

class UCIEngine:
    def __init__(self, engine_path, name="Engine"):
        self.engine_path = os.path.abspath(engine_path)
        self.name = name
        self.process = None

    def start(self):
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine binary not found: {self.engine_path}")
        self.process = subprocess.Popen(
            [self.engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1
        )
        self.send("uci")
        self.wait_for("uciok", timeout=5.0)
        self.send("isready")
        self.wait_for("readyok", timeout=10.0)

    def send(self, cmd):
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def wait_for(self, target, timeout=5.0):
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if not line:
                break
            if target in line:
                return line
        return None

    def solve(self, fen, movetime=None, depth=None):
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send("isready")
        self.wait_for("readyok", timeout=5.0)

        cmd = "go"
        if movetime is not None:
            cmd += f" movetime {movetime}"
        elif depth is not None:
            cmd += f" depth {depth}"
        else:
            cmd += " movetime 1000"

        start_t = time.time()
        self.send(cmd)

        best_move = None
        ponder_move = None
        nodes = 0
        nps = 0
        depth_reached = 0
        score_cp = 0

        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("info"):
                # Parse info fields
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "depth" and i + 1 < len(parts):
                        try: depth_reached = int(parts[i+1])
                        except: pass
                    elif p == "nodes" and i + 1 < len(parts):
                        try: nodes = int(parts[i+1])
                        except: pass
                    elif p == "nps" and i + 1 < len(parts):
                        try: nps = int(parts[i+1])
                        except: pass
                    elif p == "cp" and i + 1 < len(parts):
                        try: score_cp = int(parts[i+1])
                        except: pass
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                if len(parts) >= 4 and parts[2] == "ponder":
                    ponder_move = parts[3]
                break

        elapsed = time.time() - start_t
        return {
            "bestmove": best_move,
            "ponder": ponder_move,
            "depth": depth_reached,
            "nodes": nodes,
            "nps": nps,
            "time": elapsed,
            "score_cp": score_cp
        }

    def close(self):
        if self.process:
            try:
                self.send("quit")
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except:
                pass
            self.process = None

def parse_epd_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    # EPD format: <4 or 6 FEN tokens> [bm <move(s)>;] [am <move(s)>;] [id "<id>";]
    parts = line.split(";")
    fen_part = parts[0].strip()

    # Split fen and opcode in first token
    tokens = fen_part.split()
    if len(tokens) < 4:
        return None

    fen_tokens = tokens[:4]
    fen = " ".join(fen_tokens) + " 0 1"

    # Best moves
    best_moves = []
    avoid_moves = []
    test_id = ""

    # Look through remaining parts
    full_str = line
    bm_match = re.search(r'\bbm\s+([^;]+);', full_str)
    if bm_match:
        best_moves = [m.strip() for m in bm_match.group(1).split()]

    am_match = re.search(r'\bam\s+([^;]+);', full_str)
    if am_match:
        avoid_moves = [m.strip() for m in am_match.group(1).split()]

    id_match = re.search(r'\bid\s+"([^"]+)";', full_str)
    if id_match:
        test_id = id_match.group(1)
    else:
        id_match = re.search(r'\bid\s+([^;]+);', full_str)
        if id_match:
            test_id = id_match.group(1).strip()

    return {
        "fen": fen,
        "best_moves": best_moves,
        "avoid_moves": avoid_moves,
        "id": test_id or fen_part[:20]
    }

def normalize_move(move_str):
    if not move_str:
        return ""
    # Strip check/mate symbols
    m = move_str.replace("+", "").replace("#", "").replace("x", "").replace("=", "").strip().lower()
    return m

def run_suite(epd_path, engine_path, movetime=None, depth=None, max_positions=None):
    if not os.path.exists(epd_path):
        print(f"Error: EPD file not found: {epd_path}")
        return None

    engine = UCIEngine(engine_path, name=os.path.basename(engine_path))
    engine.start()

    positions = []
    with open(epd_path, "r") as f:
        for line in f:
            item = parse_epd_line(line)
            if item:
                positions.append(item)
                if max_positions and len(positions) >= max_positions:
                    break

    print(f"\n=======================================================")
    print(f"Running EPD Test Suite: {os.path.basename(epd_path)}")
    print(f"Engine:     {engine.name} ({engine_path})")
    print(f"Positions:  {len(positions)}")
    if movetime:
        print(f"Time limit: {movetime} ms per position")
    elif depth:
        print(f"Depth limit: {depth} ply")
    print(f"=======================================================\n")

    results = []
    solved_count = 0
    total_nodes = 0
    total_time = 0.0

    print(f"{'#':<4} {'ID':<18} {'Target':<10} {'Engine Move':<12} {'Depth':<6} {'Nodes':<10} {'Time':<8} {'Status'}")
    print("-" * 80)

    for idx, pos in enumerate(positions, 1):
        res = engine.solve(pos["fen"], movetime=movetime, depth=depth)
        best_norm = normalize_move(res["bestmove"])
        target_norms = [normalize_move(bm) for bm in pos["best_moves"]]

        # Check solution
        # Some targets are SAN like 'Bxf7+', engine outputs UCI like 'c4f7'
        # We check both exact or file/rank match
        is_solved = False
        for tm in target_norms:
            if best_norm == tm:
                is_solved = True
                break
            # Check if UCI move matches SAN destination square
            if len(best_norm) >= 4 and len(tm) >= 2:
                if best_norm[2:4] == tm[-2:]:
                    is_solved = True
                    break

        if is_solved:
            solved_count += 1
            status_str = "\033[92m[SOLVED]\033[0m"
        else:
            status_str = "\033[91m[FAILED]\033[0m"

        total_nodes += res["nodes"]
        total_time += res["time"]

        tgt_display = pos["best_moves"][0] if pos["best_moves"] else "none"
        print(f"{idx:<4} {pos['id']:<18} {tgt_display:<10} {res['bestmove'] or 'None':<12} {res['depth']:<6} {res['nodes']:<10} {res['time']:<7.2f}s {status_str}")

        results.append({
            "idx": idx,
            "id": pos["id"],
            "fen": pos["fen"],
            "targets": pos["best_moves"],
            "bestmove": res["bestmove"],
            "solved": is_solved,
            "depth": res["depth"],
            "nodes": res["nodes"],
            "time": res["time"],
            "score_cp": res["score_cp"]
        })

    engine.close()

    pct = (solved_count / len(positions)) * 100.0 if positions else 0.0
    avg_nps = int(total_nodes / total_time) if total_time > 0 else 0

    print("\n" + "=" * 80)
    print("EPD Test Suite Summary:")
    print(f"  Solved:       {solved_count} / {len(positions)} ({pct:.1f}%)")
    print(f"  Total time:   {total_time:.2f} s")
    print(f"  Total nodes:  {total_nodes}")
    print(f"  Average NPS:  {avg_nps:,} nps")
    print("=" * 80 + "\n")

    summary = {
        "suite": os.path.basename(epd_path),
        "engine": engine_path,
        "total": len(positions),
        "solved": solved_count,
        "solve_pct": pct,
        "total_time": total_time,
        "total_nodes": total_nodes,
        "avg_nps": avg_nps,
        "timestamp": datetime.now().isoformat(),
        "positions": results
    }

    # Save results to testing/results/
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"epd_{os.path.splitext(os.path.basename(epd_path))[0]}_{int(time.time())}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed results saved to: {out_file}\n")

    return summary

def compare_engines(epd_path, engine1_path, engine2_path, movetime=1000):
    print("\n--- Comparing Engine 1 (Candidate) vs Engine 2 (Baseline) ---")
    res1 = run_suite(epd_path, engine1_path, movetime=movetime)
    res2 = run_suite(epd_path, engine2_path, movetime=movetime)

    if not res1 or not res2:
        return

    print("=======================================================")
    print("COMPARISON RESULTS:")
    print(f"Candidate: {engine1_path} -> {res1['solved']}/{res1['total']} ({res1['solve_pct']:.1f}%)")
    print(f"Baseline:  {engine2_path} -> {res2['solved']}/{res2['total']} ({res2['solve_pct']:.1f}%)")
    diff = res1['solved'] - res2['solved']
    if diff > 0:
        print(f"\033[92mImprovement: +{diff} solved positions!\033[0m")
    elif diff < 0:
        print(f"\033[91mRegression: {diff} solved positions!\033[0m")
    else:
        print("Neutral: Same number of solved positions.")
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU EPD Benchmark Runner")
    parser.add_argument("--epd", default="testing/epd/bk.epd", help="Path to EPD file")
    parser.add_argument("--engine", default="./GiraffeGPU", help="Path to engine binary")
    parser.add_argument("--compare", default=None, help="Path to baseline engine to compare against")
    parser.add_argument("--movetime", type=int, default=1000, help="Search time per position in ms (default: 1000)")
    parser.add_argument("--depth", type=int, default=None, help="Fixed depth per position")
    parser.add_argument("--max", type=int, default=None, help="Max positions to run")

    args = parser.parse_args()

    # Resolve relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    epd_path = args.epd if os.path.isabs(args.epd) else os.path.join(base_dir, args.epd)
    engine_path = args.engine if os.path.isabs(args.engine) else os.path.join(base_dir, args.engine)

    if args.compare:
        compare_path = args.compare if os.path.isabs(args.compare) else os.path.join(base_dir, args.compare)
        compare_engines(epd_path, engine_path, compare_path, movetime=args.movetime)
    else:
        run_suite(epd_path, engine_path, movetime=args.movetime, depth=args.depth, max_positions=args.max)

if __name__ == "__main__":
    main()
