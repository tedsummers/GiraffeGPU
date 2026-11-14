#!/usr/bin/env python3
"""
Live Tactical Benchmark Monitor for GiraffeGPU
Watches and displays live results from Day 5 EPD benchmark evaluations (Bratko-Kopec + Tactics).
Usage:
    python3 testing/watch_benchmarks.py          # View latest benchmark results
    python3 testing/watch_benchmarks.py --watch  # Live auto-refreshing monitor
"""

import sys
import os
import glob
import json
import time
import argparse

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "testing", "results")

def get_latest_file(pattern):
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def display_suite(filepath, label):
    if not filepath or not os.path.exists(filepath):
        print(f"No {label} benchmark results found yet in {RESULTS_DIR}.")
        return

    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    suite_name = data.get("suite", label)
    solved = data.get("solved", 0)
    total = data.get("total", 0)
    pct = data.get("solve_pct", 0.0)
    total_time = data.get("total_time", 0.0)
    avg_nps = data.get("avg_nps", 0)
    ts = data.get("timestamp", "unknown")

    pct_color = "\033[92m" if pct >= 40.0 else ("\033[93m" if pct >= 25.0 else "\033[91m")
    reset = "\033[0m"

    print(f"\n================================================================================")
    print(f"  {label.upper()} TEST SUITE: {suite_name}")
    print(f"  Timestamp:   {ts}")
    print(f"  Score:       {pct_color}{solved} / {total} ({pct:.1f}%){reset} solved")
    print(f"  Search Time: {total_time:.2f}s | Avg NPS: {avg_nps:,} nps")
    print(f"================================================================================")
    print(f"{'#':<4} {'ID':<18} {'Target':<10} {'Played':<10} {'Depth':<6} {'Time':<8} {'Status'}")
    print("-" * 80)

    for p in data.get("positions", []):
        is_solved = p.get("solved", False)
        status_str = "\033[92m[SOLVED]\033[0m" if is_solved else "\033[91m[FAILED]\033[0m"
        tgt = p.get("targets", [""])[0] if p.get("targets") else "none"
        played = p.get("bestmove") or "none"
        depth = p.get("depth", 0)
        pos_time = p.get("time", 0.0)

        print(f"{p.get('idx', 0):<4} {p.get('id', ''):<18} {tgt:<10} {played:<10} {depth:<6} {pos_time:<7.2f}s {status_str}")

def render_all():
    latest_bk = get_latest_file("epd_bk_*.json")
    latest_tac = get_latest_file("epd_tactics_*.json")

    print("\033[1m================================================================================")
    print("           GiraffeGPU Live Tactical Benchmark Evaluation Monitor")
    print("================================================================================\033[0m")
    
    display_suite(latest_bk, "Bratko-Kopec (BK)")
    display_suite(latest_tac, "Tactics Suite")
    print("\n" + "=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Live Tactical Benchmark Monitor")
    parser.add_argument("--watch", action="store_true", help="Live auto-refresh mode (refreshes on every new test)")
    parser.add_argument("--interval", type=int, default=3, help="Check interval in seconds (default: 3)")
    args = parser.parse_args()

    if not args.watch:
        render_all()
        return

    print("Starting live benchmark monitor. Press Ctrl+C to exit.")
    last_mtime = 0
    while True:
        try:
            latest_bk = get_latest_file("epd_bk_*.json")
            latest_tac = get_latest_file("epd_tactics_*.json")
            mtimes = [os.path.getmtime(f) for f in (latest_bk, latest_tac) if f]
            cur_mtime = max(mtimes) if mtimes else 0

            if cur_mtime != last_mtime:
                last_mtime = cur_mtime
                # Clear terminal screen
                print("\033[2J\033[H", end="")
                render_all()
                print(f"Waiting for next evaluation cycle (auto-refreshes every ~2 mins)... [Ctrl+C to quit]")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nExiting monitor.")
            break

if __name__ == "__main__":
    main()
