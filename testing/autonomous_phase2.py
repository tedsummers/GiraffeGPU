#!/usr/bin/env python3
"""
GiraffeGPU Phase 2: 5-Day Master-Level Refinement Engine
Orchestrates a second 120-hour (5-day) self-improvement pipeline across both NVIDIA GPUs,
building directly upon the Phase 1 promoted neural network (eval.net) and tuned search parameters:
  Day 1: Fine-grained Micro-SPSA search tuning (--step-scale 0.5 --resume-best) + Depth-14 opening-book self-play
  Day 2-4: Master opening-book Dual-GPU self-play (Depth-14 Stockfish teacher, 120ms movetime),
           15-epoch full-network CUDA AdamW + Policy Head sharpening, and 40-game SPRT/LOS promotion gauntlets
  Day 5: Final validation gauntlet, deeper EPD tactical benchmarks (1000ms BK / 500ms Tactics), and packaging
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
STATE_FILE = os.path.join(RESULTS_DIR, "autonomous_phase2_state.json")
LOG_FILE = os.path.join(RESULTS_DIR, "autonomous_phase2.log")
PID_FILE = os.path.join(RESULTS_DIR, "autonomous_phase2.pid")
PHASE1_PID_FILE = os.path.join(RESULTS_DIR, "autonomous.pid")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] [PHASE-2] {msg}"
    print(formatted)
    sys.stdout.flush()
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "phase": 2,
        "start_time": time.time(),
        "total_seconds": 120 * 3600,  # 5 days
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


def is_pid_running(pid_file):
    if not os.path.exists(pid_file):
        return False, None
    try:
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True, pid
    except Exception:
        return False, None


class Phase2AutonomousPipeline:
    def __init__(self, duration_hours=120, quick_test=False):
        self.quick_test = quick_test
        self.duration_seconds = 900 if quick_test else duration_hours * 3600
        self.state = load_state()
        if quick_test:
            self.state["total_seconds"] = self.duration_seconds
        self.running = True

        signal.signal(signal.SIGINT, self.handle_exit)
        signal.signal(signal.SIGTERM, self.handle_exit)

    def handle_exit(self, signum, frame):
        log("Received shutdown signal. Finalizing Phase 2 state before exit...")
        self.running = False

    def elapsed_hours(self):
        return (time.time() - self.state["start_time"]) / 3600.0

    def current_day(self):
        day = int(self.elapsed_hours() // 24) + 1
        return min(day, 5)

    def active_search_options(self):
        search_options = ""
        search_options_path = os.path.join(RESULTS_DIR, "best_search_options.txt")
        if os.path.exists(search_options_path):
            try:
                with open(search_options_path, "r") as f:
                    search_options = f.read().strip()
            except Exception:
                pass
        common_options = "PolicyNet=true"
        if search_options:
            common_options += "," + search_options
        return common_options

    def step_micro_spsa_tuning(self):
        log(">> [Day 1 Focus] Executing Phase 2 Micro-SPSA search parameter refinement (--step-scale 0.5)...")
        iters = 1 if self.quick_test else 5
        games = 2 if self.quick_test else 6
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "tune_spsa.py"),
            "--iters", str(iters),
            "--games", str(games),
            "--tc", "60+0.5",
            "--step-scale", "0.5",
            "--resume-best",
            "--no-train-weights",
        ]
        subprocess.run(cmd, cwd=REPO_DIR)

    def step_selfplay(self, games_count):
        movetime = 50 if self.quick_test else 120
        teacher_depth = 10 if self.quick_test else 14
        book_path = os.path.join(TESTING_DIR, "books", "openings.epd")
        engine_opts = self.active_search_options()
        log(f">> Generating {games_count} dual-GPU opening-book self-play games (movetime={movetime}ms, teacher_depth={teacher_depth})...")
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "self_play.py"),
            "--games", str(games_count),
            "--movetime", str(movetime),
            "--random-plies", "2",
            "--book", book_path,
            "--teacher-depth", str(teacher_depth),
            "--engine-options", engine_opts,
        ]
        subprocess.run(cmd, cwd=REPO_DIR)
        self.state["total_selfplay_games"] += games_count

    def step_train_weights(self):
        log(">> Fine-tuning full neural network + policy head (15 epochs, lr=0.0003, policy-lr=0.0025)...")
        dataset_path = os.path.join(RESULTS_DIR, "selfplay_dataset.jsonl")
        candidate_weights = os.path.join(RESULTS_DIR, "candidate_phase2.net")
        active_weights = os.path.join(REPO_DIR, "eval.net")

        epochs = 3 if self.quick_test else 15
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "train_weights.py"),
            "--data", dataset_path,
            "--in-weights", active_weights,
            "--out-weights", candidate_weights,
            "--epochs", str(epochs),
            "--lr", "0.0003",
            "--policy-lr", "0.0025",
            "--max-output-delta", "0.035",
            "--seed", str(2026 + self.state["completed_cycles"]),
        ]
        subprocess.run(cmd, cwd=REPO_DIR)
        return candidate_weights

    def step_sprt_gauntlet(self, candidate_weights):
        log(">> Running Phase 2 Dual-GPU SPRT/LOS promotion gauntlet (Candidate vs Master Baseline)...")
        baseline_engine = os.path.join(RESULTS_DIR, "GiraffeGPU_baseline")
        if not os.path.exists(baseline_engine):
            shutil.copy(os.path.join(REPO_DIR, "GiraffeGPU"), baseline_engine)

        common_options = self.active_search_options()
        candidate_options = f"WeightsFile={candidate_weights},{common_options}"

        games = 2 if self.quick_test else 40
        cmd = [
            sys.executable, os.path.join(TESTING_DIR, "match_sprt.py"),
            "--engine1", os.path.join(REPO_DIR, "GiraffeGPU"),
            "--engine2", baseline_engine,
            "--engine1-options", candidate_options,
            "--engine2-options", f"WeightsFile={os.path.join(REPO_DIR, 'eval.net')},{common_options}",
            "--games", str(games),
            "--tc", "60+0.5",
            "--concurrency", "2",
            "--sprt"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR)
        output = proc.stdout

        accepted = (("SPRT Early Stop: ACCEPTED" in output) or ("SPRT Decision: ACCEPTED" in output)) and not self.quick_test
        if accepted:
            log(">>> PROMOTED (Phase 2): Candidate surpassed Master baseline! Deploying candidate_phase2.net to eval.net.")
            shutil.copy(candidate_weights, os.path.join(REPO_DIR, "eval.net"))
            shutil.copy(os.path.join(REPO_DIR, "GiraffeGPU"), baseline_engine)
            self.state["promoted_networks"] += 1
            subprocess.run(["make", "windows"], cwd=REPO_DIR, stdout=subprocess.DEVNULL)
        else:
            log(">>> REJECTED (Phase 2): Candidate did not surpass baseline. Preserving current eval.net.")
            self.state["rejected_networks"] += 1

    def step_benchmarks(self):
        log(">> Running Phase 2 Master tactical benchmark evaluation (Bratko-Kopec @ 1000ms + Tactics @ 500ms)...")
        cmd_bk = [sys.executable, os.path.join(TESTING_DIR, "run_epd.py"), "--epd", "testing/epd/bk.epd", "--movetime", "1000"]
        cmd_tac = [sys.executable, os.path.join(TESTING_DIR, "run_epd.py"), "--epd", "testing/epd/tactics.epd", "--movetime", "500"]

        subprocess.run(cmd_bk, cwd=REPO_DIR)
        subprocess.run(cmd_tac, cwd=REPO_DIR)

        bk_score, tac_score = 0.0, 0.0
        try:
            with open(os.path.join(RESULTS_DIR, "bk_results.json")) as f:
                bk_score = json.load(f).get("score_pct", 0.0)
            with open(os.path.join(RESULTS_DIR, "tactics_results.json")) as f:
                tac_score = json.load(f).get("score_pct", 0.0)
        except Exception:
            pass

        return bk_score, tac_score

    def run(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        log("=================================================================")
        log("GIRAFFEGPU PHASE 2: 5-DAY MASTER-LEVEL REFINEMENT ENGINE STARTED")
        log(f"Target Duration: {self.duration_seconds / 3600:.2f} hours | PID: {os.getpid()}")
        log("=================================================================")

        baseline_engine = os.path.join(RESULTS_DIR, "GiraffeGPU_baseline")
        shutil.copy(os.path.join(REPO_DIR, "GiraffeGPU"), baseline_engine)

        while self.running:
            elapsed = time.time() - self.state["start_time"]
            if elapsed >= self.duration_seconds:
                log("Target 5-day (120-hour) Phase 2 duration reached! Concluding training loop.")
                break

            day = self.current_day()
            self.state["current_day"] = day
            cycle = self.state["completed_cycles"] + 1
            remaining_hrs = max(0.0, (self.duration_seconds - elapsed) / 3600.0)

            log(f"--- Starting Phase 2 Cycle #{cycle} | Day {day}/5 | Remaining: {remaining_hrs:.1f} hrs ---")

            if day == 1 or self.quick_test:
                self.step_micro_spsa_tuning()
                self.step_selfplay(4 if self.quick_test else 30)
                if not self.quick_test:
                    cand = self.step_train_weights()
                    self.step_sprt_gauntlet(cand)

            if day in (2, 3, 4) or self.quick_test:
                games_to_play = 4 if self.quick_test else 50
                self.step_selfplay(games_to_play)
                cand = self.step_train_weights()
                self.step_sprt_gauntlet(cand)

            if day == 5 or self.quick_test:
                self.step_selfplay(2 if self.quick_test else 25)
                cand = self.step_train_weights()
                self.step_sprt_gauntlet(cand)

            bk_pct, tac_pct = self.step_benchmarks()

            self.state["completed_cycles"] = cycle
            self.state["history"].append({
                "cycle": cycle,
                "day": day,
                "timestamp": datetime.now().isoformat(),
                "selfplay_games": self.state["total_selfplay_games"],
                "promoted": self.state["promoted_networks"],
                "bk_score_pct": bk_pct,
                "tactics_score_pct": tac_pct
            })
            save_state(self.state)

            log(f"--- Completed Phase 2 Cycle #{cycle} | BK: {bk_pct:.1f}% | Tactics: {tac_pct:.1f}% | Promoted Nets: {self.state['promoted_networks']} ---")

            if self.quick_test:
                log("Quick test cycle completed successfully.")
                break

        self.generate_final_report()
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)

    def generate_final_report(self):
        report_path = os.path.join(RESULTS_DIR, "phase2_5day_final_report.md")
        elapsed_h = (time.time() - self.state["start_time"]) / 3600.0
        bk_last = self.state["history"][-1]["bk_score_pct"] if self.state["history"] else 0.0
        tac_last = self.state["history"][-1]["tactics_score_pct"] if self.state["history"] else 0.0

        content = f"""# GiraffeGPU Phase 2 (Master-Level Refinement) Final Report

- **Completion Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Runtime:** {elapsed_h:.2f} hours
- **Completed Optimization Cycles:** {self.state['completed_cycles']}
- **Total Opening-Book Self-Play Games Generated (Depth-14 Teacher):** {self.state['total_selfplay_games']}
- **Promoted Neural Network Revisions (`eval.net`):** {self.state['promoted_networks']}
- **Rejected Candidate Revisions:** {self.state['rejected_networks']}
- **Final Bratko-Kopec Tactical Score (1000ms):** {bk_last:.1f}%
- **Final Tactical Suite Score (500ms):** {tac_last:.1f}%

## Deployed Artifacts
- **Linux CUDA Binary:** `./GiraffeGPU`
- **Windows 64-bit Standalone Executable:** `./GiraffeGPU.exe`
- **Master Neural Network Weights:** `./eval.net`
- **Refined Search Parameters:** `testing/results/best_search_params.json`
"""
        with open(report_path, "w") as f:
            f.write(content)
        log(f"Final Phase 2 5-day summary report written to: {report_path}")


def print_status():
    p1_running, p1_pid = is_pid_running(PHASE1_PID_FILE)
    if p1_running:
        print(f"[INFO] Phase 1 daemon is currently ACTIVE (PID: {p1_pid}). Wait for Phase 1 to finish before starting Phase 2.")

    if not os.path.exists(STATE_FILE):
        print("No active or previous Phase 2 autonomous run state found.")
        return

    with open(STATE_FILE, "r") as f:
        st = json.load(f)

    running, pid = is_pid_running(PID_FILE)
    elapsed = time.time() - st["start_time"]
    rem = max(0, st.get("total_seconds", 432000) - elapsed)

    print("=======================================================")
    print("GIRAFFEGPU PHASE 2 (MASTER REFINEMENT) STATUS")
    print("=======================================================")
    print(f"Process State:        {'RUNNING (PID ' + str(pid) + ')' if running else 'STOPPED / COMPLETED'}")
    print(f"Current Day:          Day {st['current_day']} of 5")
    print(f"Elapsed Time:         {timedelta(seconds=int(elapsed))}")
    print(f"Remaining Time:       {timedelta(seconds=int(rem))}")
    print(f"Completed Cycles:     {st['completed_cycles']}")
    print(f"Self-Play Games:      {st['total_selfplay_games']}")
    print(f"Promoted Networks:    {st['promoted_networks']} (Rejected: {st['rejected_networks']})")
    prog_file = os.path.join(RESULTS_DIR, "elo_progression.json")
    if os.path.exists(prog_file):
        try:
            with open(prog_file, "r") as pf:
                prog = json.load(pf)
            print(f"Estimated Elo:        ~{int(prog.get('current_elo', 1850))} Elo")
            if prog.get("promotions"):
                latest_ckpt = prog["promotions"][-1].get("checkpoint_file", "")
                print(f"Latest Checkpoint:    {latest_ckpt} ({len(prog['promotions'])} saved)")
        except Exception:
            pass
    if st.get("history"):
        last = st["history"][-1]
        print(f"Latest Benchmarks:    BK={last['bk_score_pct']:.1f}% | Tactics={last['tactics_score_pct']:.1f}%")
    print("=======================================================")


def stop_daemon():
    if not os.path.exists(PID_FILE):
        print("No running Phase 2 PID file found.")
        return
    try:
        with open(PID_FILE, "r") as f:
            pid = int(f.read().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to Phase 2 autonomous daemon (PID {pid}).")
    except Exception as e:
        print(f"Could not stop Phase 2 process: {e}")


def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Phase 2 5-Day Master-Level Refinement Engine")
    parser.add_argument("--start", action="store_true", help="Start the 5-day Phase 2 pipeline in foreground")
    parser.add_argument("--daemon", action="store_true", help="Launch the 5-day Phase 2 pipeline in background")
    parser.add_argument("--status", action="store_true", help="Check current progress of Phase 2 training run")
    parser.add_argument("--stop", action="store_true", help="Gracefully stop a running Phase 2 background daemon")
    parser.add_argument("--test-run", action="store_true", help="Execute 1 rapid verification cycle (~15 mins)")
    parser.add_argument("--fresh", action="store_true", help="Reset Phase 2 state before starting")
    parser.add_argument("--force", action="store_true", help="Allow starting even if Phase 1 PID file is active")
    parser.add_argument("--duration-hours", type=float, default=120.0, help="Override total duration in hours (default: 120)")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.stop:
        stop_daemon()
        return

    if args.start or args.daemon or args.test_run:
        p1_running, p1_pid = is_pid_running(PHASE1_PID_FILE)
        if p1_running and not args.force:
            print(f"ERROR: Phase 1 training is still running (PID {p1_pid}).")
            print("Please wait for Phase 1 to complete (or stop it with `python3 testing/autonomous_5day.py --stop`)")
            print("before launching Phase 2 so both runs do not compete for the GPUs.")
            sys.exit(1)

    if args.fresh and os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

    if args.daemon:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        cmd = [sys.executable, os.path.abspath(__file__), "--start", "--duration-hours", str(args.duration_hours)]
        if args.force:
            cmd.append("--force")
        with open(LOG_FILE, "a") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=out, start_new_session=True, cwd=REPO_DIR)
        print(f"Phase 2 5-Day Autonomous Engine launched in background! PID: {proc.pid}")
        print(f"Monitor progress with: python3 testing/autonomous_phase2.py --status")
        print(f"Live log file:         tail -f {LOG_FILE}")
        return

    if args.test_run:
        pipeline = Phase2AutonomousPipeline(duration_hours=0.1, quick_test=True)
        pipeline.run()
        return

    if args.start:
        pipeline = Phase2AutonomousPipeline(duration_hours=args.duration_hours, quick_test=False)
        pipeline.run()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
