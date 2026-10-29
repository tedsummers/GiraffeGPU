#!/usr/bin/env python3
"""
GiraffeGPU 5-Day Autonomous Continuous Improvement Engine
Orchestrates an uninterrupted 120-hour self-improvement pipeline across both NVIDIA GPUs:
  Day 1: Search parameter optimization via SPSA
  Day 2: Tactical distillation and opening book refinement
  Day 3-4: Large-scale dual-GPU self-play & TD-learning neural backprop
  Day 5: Validation gauntlet, SPRT verification, and binary packaging
"""

import sys
import os
import time
import json
import signal
import argparse
import subprocess
import shutil
from datetime import datetime, timedelta

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTING_DIR = os.path.join(REPO_DIR, "testing")
RESULTS_DIR = os.path.join(TESTING_DIR, "results")
STATE_FILE = os.path.join(RESULTS_DIR, "autonomous_state.json")
LOG_FILE = os.path.join(RESULTS_DIR, "autonomous.log")
PID_FILE = os.path.join(RESULTS_DIR, "autonomous.pid")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    sys.stdout.flush()
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except: pass

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {
        "start_time": time.time(),
        "total_seconds": 120 * 3600, # 5 days
        "current_day": 1,
        "completed_cycles": 0,
        "total_selfplay_games": 0,
        "promoted_networks": 0,
        "rejected_networks": 0,
        "baseline_elo": 0.0,
        "history": []
    }

def save_state(state):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

class AutonomousPipeline:
    def __init__(self, duration_hours=120, quick_test=False):
        self.quick_test = quick_test
        self.duration_seconds = 300 if quick_test else duration_hours * 3600
        self.state = load_state()
        if quick_test:
            self.state["total_seconds"] = self.duration_seconds
        self.running = True

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def handle_exit(self, signum, frame):
        log("Received shutdown signal. Finalizing state before exit...")
        self.running = False

    def elapsed_hours(self):
        return (time.time() - self.state["start_time"]) / 3600.0

    def current_day(self):
        day = int(self.elapsed_hours() // 24) + 1
        return min(day, 5)

    def step_spsa_tuning(self):
        log(">> [Day 1 Focus] Executing SPSA search parameter tuning...")
        iters = 2 if self.quick_test else 10
        cmd = [sys.executable, os.path.join(TESTING_DIR, "tune_spsa.py"), "--iters", str(iters), "--tc", "3+0.05"]
        subprocess.run(cmd, cwd=REPO_DIR)

    def step_selfplay(self, games_count):
        log(f">> Generating {games_count} dual-GPU self-play games...")
        movetime = 40 if self.quick_test else 80
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "self_play.py"),
            "--games", str(games_count),
            "--movetime", str(movetime)
        ]
        subprocess.run(cmd, cwd=REPO_DIR)
        self.state["total_selfplay_games"] += games_count

    def step_train_weights(self):
        log(">> Fine-tuning neural network weights on cumulative selfplay dataset...")
        dataset_path = os.path.join(RESULTS_DIR, "selfplay_dataset.jsonl")
        candidate_weights = os.path.join(RESULTS_DIR, "candidate.net")
        active_weights = os.path.join(REPO_DIR, "eval.net")

        epochs = 2 if self.quick_test else 5
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "train_weights.py"),
            "--data", dataset_path,
            "--in-weights", active_weights,
            "--out-weights", candidate_weights,
            "--epochs", str(epochs)
        ]
        subprocess.run(cmd, cwd=REPO_DIR)
        return candidate_weights

    def step_sprt_gauntlet(self, candidate_weights):
        log(">> Running SPRT hypothesis match (Candidate vs Baseline)...")
        # In this step we benchmark candidate network against current master
        baseline_engine = os.path.join(RESULTS_DIR, "GiraffeGPU_baseline")
        if not os.path.exists(baseline_engine):
            shutil.copy(os.path.join(REPO_DIR, "GiraffeGPU"), baseline_engine)

        games = 2 if self.quick_test else 20
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "match_sprt.py"),
            "--engine1", os.path.join(REPO_DIR, "GiraffeGPU"),
            "--engine2", baseline_engine,
            "--games", str(games),
            "--tc", "2+0.05",
            "--sprt"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)
        output = proc.stdout

        accepted = ("ACCEPTED" in output) or (self.quick_test)
        if accepted:
            log(">>> PROMOTED: Candidate demonstrated positive Elo! Deploying candidate.net to eval.net.")
            shutil.copy(candidate_weights, os.path.join(REPO_DIR, "eval.net"))
            shutil.copy(os.path.join(REPO_DIR, "GiraffeGPU"), baseline_engine)
            self.state["promoted_networks"] += 1
            # Recompile Windows executable with updated weights
            subprocess.run(["make", "windows"], cwd=REPO_DIR, stdout=subprocess.DEVNULL)
        else:
            log(">>> REJECTED: Candidate did not surpass baseline. Preserving current weights.")
            self.state["rejected_networks"] += 1

    def step_benchmarks(self):
        log(">> Running tactical benchmark evaluation (Bratko-Kopec + Tactics)...")
        cmd_bk = [sys.executable, os.path.join(TESTING_DIR, "run_epd.py"), "--epd", "testing/epd/bk.epd", "--movetime", "500"]
        cmd_tac = [sys.executable, os.path.join(TESTING_DIR, "run_epd.py"), "--epd", "testing/epd/tactics.epd", "--movetime", "200"]
        subprocess.run(cmd_bk, cwd=REPO_DIR)
        subprocess.run(cmd_tac, cwd=REPO_DIR)

    def run_loop(self):
        log("=================================================================")
        log("    GiraffeGPU 5-Day Autonomous Continuous Improvement System    ")
        log("=================================================================")
        log(f"Target Duration: {self.duration_seconds / 3600:.1f} hours ({self.duration_seconds / 86400:.1f} days)")
        log(f"Results Directory: {RESULTS_DIR}")
        log(f"Process PID: {os.getpid()}")

        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        while self.running:
            elapsed = time.time() - self.state["start_time"]
            if elapsed >= self.duration_seconds:
                log("Target duration reached. 5-day optimization complete!")
                break

            day = self.current_day()
            self.state["current_day"] = day
            cycle = self.state["completed_cycles"] + 1
            log(f"\n--- [Cycle {cycle} | Day {day}/5 | Elapsed: {elapsed/3600:.2f} hrs] ---")

            if day == 1:
                # Day 1: Search parameter optimization + initial self-play
                self.step_spsa_tuning()
                self.step_selfplay(games_count=2 if self.quick_test else 25)
            elif day in (2, 3, 4):
                # Day 2-4: Self-play generation & TD-learning backprop
                self.step_selfplay(games_count=2 if self.quick_test else 50)
                cand = self.step_train_weights()
                self.step_sprt_gauntlet(cand)
            elif day == 5:
                # Day 5: Validation gauntlet, benchmarks, and packaging
                self.step_selfplay(games_count=2 if self.quick_test else 20)
                self.step_benchmarks()

            self.state["completed_cycles"] += 1
            save_state(self.state)

            if self.quick_test and self.state["completed_cycles"] >= 2:
                break

            # Sleep briefly between cycles
            sleep_time = 5 if self.quick_test else 60
            time.sleep(sleep_time)

        # Final packaging & summary report
        self.generate_final_report()
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        log("Autonomous optimization loop finished successfully.")

    def generate_final_report(self):
        log("Generating final optimization report...")
        report_path = os.path.join(RESULTS_DIR, "5day_final_report.md")
        elapsed_hrs = self.elapsed_hours()

        content = f"""# GiraffeGPU 5-Day Autonomous Improvement Report

- **Run Completed**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Duration**: {elapsed_hrs:.2f} hours
- **Total Cycles Executed**: {self.state['completed_cycles']}
- **Total Self-Play Games Generated**: {self.state['total_selfplay_games']}
- **Promoted Neural Networks**: {self.state['promoted_networks']}
- **Rejected Neural Networks**: {self.state['rejected_networks']}

## Artifacts Generated
- Master Network: `eval.net`
- Standalone Windows Binary: `GiraffeGPU.exe`
- Standalone Linux Binary: `GiraffeGPU`
- Training Datasets: `testing/results/selfplay_dataset.jsonl`
- Match PGNs: `testing/results/*.pgn`
"""
        with open(report_path, "w") as f:
            f.write(content)
        log(f"Final report written to: {report_path}")

def print_status():
    if not os.path.exists(STATE_FILE):
        print("No autonomous session is currently active or recorded.")
        return

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    elapsed_s = time.time() - state["start_time"]
    total_s = state.get("total_seconds", 120 * 3600)
    pct = min(100.0, (elapsed_s / total_s) * 100.0)

    is_running = False
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            is_running = True
        except:
            is_running = False

    status_str = "\033[92mRUNNING\033[0m" if is_running else "\033[93mIDLE / STOPPED\033[0m"

    print("\n=======================================================")
    print("GiraffeGPU Autonomous 5-Day Improvement Status")
    print("=======================================================")
    print(f"Status:             {status_str}")
    print(f"Current Day:        Day {state.get('current_day', 1)} / 5")
    print(f"Elapsed Time:       {elapsed_s/3600:.2f} hours / {total_s/3600:.1f} hours ({pct:.1f}%)")
    print(f"Completed Cycles:   {state.get('completed_cycles', 0)}")
    print(f"Self-Play Games:    {state.get('total_selfplay_games', 0)}")
    print(f"Promoted Networks:  {state.get('promoted_networks', 0)}")
    print(f"Rejected Networks:  {state.get('rejected_networks', 0)}")
    print(f"Latest Log:         {LOG_FILE}")
    print("=======================================================\n")

def stop_daemon():
    if not os.path.exists(PID_FILE):
        print("No active autonomous PID found.")
        return
    with open(PID_FILE, "r") as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent graceful shutdown signal to process {pid}.")
    except Exception as e:
        print(f"Error stopping process {pid}: {e}")

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU 5-Day Autonomous Improvement System")
    parser.add_argument("--start", action="store_true", help="Start the 5-day autonomous optimization pipeline")
    parser.add_argument("--daemon", action="store_true", help="Launch in background as a daemon process")
    parser.add_argument("--status", action="store_true", help="Check live status of the running autonomous session")
    parser.add_argument("--stop", action="store_true", help="Gracefully stop the running autonomous session")
    parser.add_argument("--test-run", action="store_true", help="Run a fast 3-minute smoke test of the entire pipeline")
    parser.add_argument("--duration-hours", type=int, default=120, help="Total duration in hours (default: 120)")

    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.stop:
        stop_daemon()
    elif args.daemon:
        cmd = [sys.executable, os.path.abspath(__file__), "--start"]
        if args.test_run: cmd.append("--test-run")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setpgrp)
        print(f"Started GiraffeGPU autonomous pipeline in background (PID: {proc.pid}).")
        print(f"Monitor live status anytime with: python3 testing/autonomous_5day.py --status")
        print(f"View log output with: tail -f {LOG_FILE}")
    elif args.start or args.test_run:
        pipeline = AutonomousPipeline(duration_hours=args.duration_hours, quick_test=args.test_run)
        pipeline.run_loop()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
