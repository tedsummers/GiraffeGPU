#!/usr/bin/env python3
"""
GiraffeGPU Unified Testing & Strength Improvement Suite
Main CLI front-end for running EPD tactical benchmarks, SPRT matches, self-play data generation,
and parameter optimization.
"""

import sys
import os
import argparse
import subprocess

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTING_DIR = os.path.join(REPO_DIR, "testing")

def run_cmd(script_name, args_list):
    script_path = os.path.join(TESTING_DIR, script_name)
    cmd = [sys.executable, script_path] + args_list
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(
        description="GiraffeGPU Testing Framework for Measuring and Improving Playing Strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Bratko-Kopec tactical test suite (24 benchmark positions):
  python3 testing/run_tests.py --suite bk --movetime 1000

  # Run tactics test suite:
  python3 testing/run_tests.py --suite tactics --movetime 500

  # Compare candidate engine against a baseline on tactical benchmark:
  python3 testing/run_tests.py --suite bk --compare /path/to/baseline_engine

  # Run head-to-head match (20 games) against another engine:
  python3 testing/run_tests.py --match --engine2 /path/to/opponent --games 20

  # Run SPRT hypothesis test (statistical validation of Elo gain):
  python3 testing/run_tests.py --sprt --engine2 /path/to/baseline_engine --elo0 0.0 --elo1 10.0

  # Generate self-play games for training / TD-learning:
  python3 testing/run_tests.py --selfplay --games 25 --movetime 100

  # Run SPSA parameter tuning:
  python3 testing/run_tests.py --spsa --iters 10
"""
    )

    parser.add_argument("--suite", choices=["bk", "tactics"], default=None, help="Run EPD test suite ('bk' or 'tactics')")
    parser.add_argument("--epd", default=None, help="Custom EPD file path")
    parser.add_argument("--engine", default="./GiraffeGPU", help="Target engine binary (default: ./GiraffeGPU)")
    parser.add_argument("--compare", default=None, help="Baseline engine to compare against on EPD suite")
    parser.add_argument("--movetime", type=int, default=1000, help="Movetime per position/move in ms (default: 1000)")
    parser.add_argument("--depth", type=int, default=None, help="Fixed search depth")

    parser.add_argument("--match", action="store_true", help="Run head-to-head match against engine2")
    parser.add_argument("--sprt", action="store_true", help="Run SPRT test against engine2")
    parser.add_argument("--compare-policy", action="store_true", help="Run A/B match comparing PolicyNet enabled vs disabled")
    parser.add_argument("--engine2", default=None, help="Opponent / baseline engine binary")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument("--tc", default="5+0.1", help="Time control for match (default: 5+0.1)")
    parser.add_argument("--elo0", type=float, default=0.0, help="SPRT H0 Elo bound")
    parser.add_argument("--elo1", type=float, default=10.0, help="SPRT H1 Elo bound")

    parser.add_argument("--selfplay", action="store_true", help="Run self-play data generation")
    parser.add_argument("--train", action="store_true", help="Train dual-head neural network weights on selfplay data")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--spsa", action="store_true", help="Run SPSA search parameter tuning")
    parser.add_argument("--iters", type=int, default=10, help="Iterations for SPSA tuning")

    args = parser.parse_args()

    # Determine action
    if args.compare_policy:
        sub_args = [
            "--engine1", args.engine,
            "--compare-policy",
            "--games", str(args.games),
            "--tc", args.tc
        ]
        if args.sprt:
            sub_args.extend(["--sprt", "--elo0", str(args.elo0), "--elo1", str(args.elo1)])
        sys.exit(run_cmd("match_sprt.py", sub_args))

    elif args.match or args.sprt:
        if not args.engine2:
            print("Error: --engine2 <path> is required for --match and --sprt.")
            sys.exit(1)
        sub_args = [
            "--engine1", args.engine,
            "--engine2", args.engine2,
            "--games", str(args.games),
            "--tc", args.tc
        ]
        if args.sprt:
            sub_args.extend(["--sprt", "--elo0", str(args.elo0), "--elo1", str(args.elo1)])
        sys.exit(run_cmd("match_sprt.py", sub_args))

    elif args.train:
        sub_args = [
            "--epochs", str(args.epochs)
        ]
        sys.exit(run_cmd("train_weights.py", sub_args))

    elif args.selfplay:
        sub_args = [
            "--engine", args.engine,
            "--games", str(args.games),
            "--movetime", str(args.movetime)
        ]
        sys.exit(run_cmd("self_play.py", sub_args))

    elif args.spsa:
        sub_args = [
            "--engine", args.engine,
            "--iters", str(args.iters),
            "--tc", args.tc
        ]
        sys.exit(run_cmd("tune_spsa.py", sub_args))

    else:
        # Default: EPD suite
        suite_file = "bk.epd"
        if args.suite == "tactics":
            suite_file = "tactics.epd"
        elif args.epd:
            suite_file = args.epd
        
        epd_path = os.path.join(TESTING_DIR, "epd", suite_file) if not os.path.isabs(suite_file) else suite_file
        sub_args = ["--epd", epd_path, "--engine", args.engine, "--movetime", str(args.movetime)]
        if args.compare:
            sub_args.extend(["--compare", args.compare])
        if args.depth:
            sub_args.extend(["--depth", str(args.depth)])
        sys.exit(run_cmd("run_epd.py", sub_args))

if __name__ == "__main__":
    main()
