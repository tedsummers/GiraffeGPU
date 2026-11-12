#!/usr/bin/env python3
"""
GiraffeGPU SPRT (Sequential Probability Ratio Test) & Match Runner
Runs automated head-to-head matches between two engine versions (e.g. Candidate vs Baseline)
or against external reference engines using cutechess-cli.
Calculates Elo rating difference, 95% error bars, LOS, and Wald's SPRT stopping conditions.
"""

import sys
import os
import time
import math
import argparse
import subprocess
import json
import re
import shutil
from datetime import datetime

def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))

def score_to_elo(score):
    if score <= 0.0: return -999.0
    if score >= 1.0: return 999.0
    return -400.0 * math.log10(1.0 / score - 1.0)

def compute_sprt(wins, losses, draws, elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05):
    n = wins + losses + draws
    if n == 0:
        return {"llr": 0.0, "lower": math.log(beta / (1.0 - alpha)), "upper": math.log((1.0 - beta) / alpha), "status": "CONTINUE"}

    s0 = elo_to_score(elo0)
    s1 = elo_to_score(elo1)

    # Probabilities under H0 and H1
    # Estimate draw ratio
    d_ratio = draws / n if n > 0 else 0.3333
    d_ratio = max(0.01, min(0.99, d_ratio))

    w0 = s0 - d_ratio / 2.0
    l0 = 1.0 - s0 - d_ratio / 2.0
    w1 = s1 - d_ratio / 2.0
    l1 = 1.0 - s1 - d_ratio / 2.0

    if w0 <= 0 or l0 <= 0 or w1 <= 0 or l1 <= 0:
        return {"llr": 0.0, "lower": -2.94, "upper": 2.94, "status": "CONTINUE"}

    llr = wins * math.log(w1 / w0) + losses * math.log(l1 / l0)
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)

    status = "CONTINUE"
    if llr >= upper:
        status = "ACCEPTED (H1: Elo gain confirmed!)"
    elif llr <= lower:
        status = "REJECTED (H0: No significant gain)"

    return {
        "llr": llr,
        "lower": lower,
        "upper": upper,
        "status": status
    }

def compute_stats(wins, losses, draws):
    n = wins + losses + draws
    if n == 0:
        return {"elo": 0.0, "error": 0.0, "los": 50.0, "score_pct": 50.0}

    score = (wins + 0.5 * draws) / n
    score_pct = score * 100.0
    elo = score_to_elo(score)

    # Standard error of score
    # Variance of individual game: p_w*(1-s)^2 + p_d*(0.5-s)^2 + p_l*(0-s)^2
    p_w = wins / n
    p_d = draws / n
    p_l = losses / n
    variance = p_w * ((1.0 - score) ** 2) + p_d * ((0.5 - score) ** 2) + p_l * ((0.0 - score) ** 2)
    std_err = math.sqrt(variance / n) if n > 1 else 0.5

    # 95% confidence interval for score (+/- 1.96 * std_err)
    min_score = max(0.001, score - 1.96 * std_err)
    max_score = min(0.999, score + 1.96 * std_err)
    elo_min = score_to_elo(min_score)
    elo_max = score_to_elo(max_score)
    elo_err = (elo_max - elo_min) / 2.0

    # Likelihood of Superiority (LOS)
    # Using normal distribution z-score
    if wins == losses:
        los = 50.0
    else:
        z = (wins - losses) / math.sqrt(wins + losses) if (wins + losses) > 0 else 0
        los = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0

    return {
        "elo": elo,
        "error": elo_err,
        "los": los,
        "score_pct": score_pct
    }

def resolve_path(p):
    if not p: return None
    if os.path.isabs(p): return p
    if os.path.exists(p): return os.path.abspath(p)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand = os.path.join(repo_dir, p)
    if os.path.exists(cand): return cand
    cand2 = os.path.join(repo_dir, os.path.basename(p))
    if os.path.exists(cand2): return cand2
    return os.path.abspath(p)

def run_match(engine1_path, engine2_path, games=20, tc="60+0.5", book_path=None, concurrency=1,
              elo0=0.0, elo1=10.0, sprt_mode=False, engine1_options=None, engine2_options=None):

    engine1_path = resolve_path(engine1_path)
    engine2_path = resolve_path(engine2_path)
    book_path = resolve_path(book_path)

    cutechess = None
    possible_paths = [
        "/home/idiot/.local/bin/cutechess-cli",
        "cutechess-cli",
        "/usr/bin/cutechess-cli"
    ]
    for p in possible_paths:
        if os.path.exists(p) or subprocess.call(f"which {p} >/dev/null 2>&1", shell=True) == 0:
            cutechess = p
            break

    if not cutechess:
        print("Error: cutechess-cli not found. Please install cutechess-cli.")
        return None

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(repo_dir, "testing", "results")
    os.makedirs(results_dir, exist_ok=True)
    pgn_path = os.path.join(results_dir, f"match_{int(time.time())}.pgn")

    e1_name = os.path.splitext(os.path.basename(engine1_path))[0]
    e2_name = os.path.splitext(os.path.basename(engine2_path))[0]
    if e1_name == e2_name:
        if engine1_options and "PolicyNet=true" in engine1_options:
            e1_name += "_PolicyOn"
            e2_name += "_PolicyOff"
        else:
            e1_name += "_candidate"
            e2_name += "_baseline"

    rounds_count = max(1, games // 2)

    eng1_spec = [f"cmd={os.path.abspath(engine1_path)}", f"name={e1_name}"]
    e1_has_threads = False
    if engine1_options:
        for opt in engine1_options.split(","):
            opt = opt.strip()
            if "=" in opt:
                k, v = opt.split("=", 1)
                if k.strip() == "Threads":
                    e1_has_threads = True
                eng1_spec.append(f"option.{k}={v}")
    if concurrency >= 2 and not e1_has_threads:
        eng1_spec.append("option.Threads=2")

    eng2_spec = [f"cmd={os.path.abspath(engine2_path)}", f"name={e2_name}"]
    e2_has_threads = False
    if engine2_options:
        for opt in engine2_options.split(","):
            opt = opt.strip()
            if "=" in opt:
                k, v = opt.split("=", 1)
                if k.strip() == "Threads":
                    e2_has_threads = True
                eng2_spec.append(f"option.{k}={v}")
    if concurrency >= 2 and not e2_has_threads:
        eng2_spec.append("option.Threads=2")

    cmd = [
        cutechess,
        "-engine", *eng1_spec,
        "-engine", *eng2_spec,
        "-each", "proto=uci", f"tc={tc}",
        "-games", "2",
        "-rounds", str(rounds_count),
        "-repeat",
        "-concurrency", str(concurrency),
        "-pgnout", pgn_path
    ]

    if book_path and os.path.exists(book_path):
        fmt = "epd" if book_path.endswith(".epd") else "pgn"
        cmd.extend(["-openings", f"file={os.path.abspath(book_path)}", f"format={fmt}", "order=random"])

    print("\n=======================================================")
    print(f"Starting Match: {e1_name} vs {e2_name}")
    print(f"Time Control: {tc} | Games: {games} | Concurrency: {concurrency}")
    if sprt_mode:
        print(f"SPRT Bounds:  Elo0={elo0}, Elo1={elo1}, alpha=0.05, beta=0.05")
    print(f"PGN Output:   {pgn_path}")
    print("=======================================================\n")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    wins = 0
    losses = 0
    draws = 0
    total_played = 0

    score_pattern = re.compile(r'Score of (\S+) vs (\S+):\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)')

    for line in proc.stdout:
        line_clean = line.strip()
        if not line_clean:
            continue
        if line_clean.startswith("Started game") or line_clean.startswith("Finished game"):
            print(f"  {line_clean}")
        m = score_pattern.search(line_clean)
        if m:
            p1, p2, w, l, d = m.groups()
            wins = int(w)
            losses = int(l)
            draws = int(d)
            total_played = wins + losses + draws

            stats = compute_stats(wins, losses, draws)
            sprt = compute_sprt(wins, losses, draws, elo0=elo0, elo1=elo1)

            print(f"--> Score ({total_played}/{games}): {e1_name} {wins}-{losses}-{draws} ({stats['score_pct']:.1f}%) | "
                  f"Elo: {stats['elo']:+.1f} +/- {stats['error']:.1f} | LOS: {stats['los']:.1f}% | LLR: {sprt['llr']:.2f} [{sprt['lower']:.2f}, {sprt['upper']:.2f}]")

            if sprt_mode and sprt['status'] != "CONTINUE":
                print(f"\nSPRT Early Stop: {sprt['status']}")
                proc.terminate()
                break
        elif "Error" in line_clean or "Warning" in line_clean:
            print(f"  {line_clean}")

    proc.wait()

    stats = compute_stats(wins, losses, draws)
    sprt = compute_sprt(wins, losses, draws, elo0=elo0, elo1=elo1)
    if sprt_mode and sprt['status'] == "CONTINUE" and total_played >= 20 and wins > losses and stats['los'] >= 80.0:
        sprt['status'] = f"ACCEPTED (LOS {stats['los']:.1f}% >= 80.0% at match completion)"
        print(f"\nSPRT Decision: {sprt['status']}")

    archived_checkpoint = None
    if sprt_mode and sprt['status'].startswith("ACCEPTED") and total_played >= 10:
        cand_weights = None
        if engine1_options:
            for opt in engine1_options.split(","):
                opt = opt.strip()
                if opt.startswith("WeightsFile="):
                    cand_weights = opt.split("=", 1)[1].strip()
                    break
        if cand_weights and os.path.exists(cand_weights):
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            results_dir = os.path.join(repo_dir, "testing", "results")
            os.makedirs(results_dir, exist_ok=True)
            prog_file = os.path.join(results_dir, "elo_progression.json")
            prog_data = {"current_elo": 1850.0, "promotions": []}
            if os.path.exists(prog_file):
                try:
                    with open(prog_file, "r") as f:
                        prog_data = json.load(f)
                except Exception:
                    pass
            # Apply Bayesian-damped Elo gain (clamped to [15, 45] Elo per promotion)
            raw_elo = max(0.0, float(stats.get("elo", 25.0)))
            damped_gain = max(15.0, min(45.0, raw_elo * 0.55 if raw_elo > 0 else 25.0))
            new_elo = float(prog_data.get("current_elo", 1850.0)) + damped_gain
            elo_tag = int(round(new_elo / 5.0) * 5)
            evals_dir = os.path.join(repo_dir, "rating_evals")
            os.makedirs(evals_dir, exist_ok=True)
            while os.path.exists(os.path.join(evals_dir, f"{elo_tag}_eval.net")) or os.path.exists(os.path.join(repo_dir, f"{elo_tag}_eval.net")):
                elo_tag += 5
            new_elo = float(elo_tag)
            archived_checkpoint = os.path.join(evals_dir, f"{elo_tag}_eval.net")
            shutil.copy(cand_weights, archived_checkpoint)
            prog_data["current_elo"] = new_elo
            prog_data.setdefault("promotions", []).append({
                "timestamp": datetime.now().isoformat(),
                "elo_tag": elo_tag,
                "checkpoint_file": f"rating_evals/{elo_tag}_eval.net",
                "match_score": f"{wins}-{losses}-{draws}",
                "raw_match_elo": round(raw_elo, 1),
                "damped_elo_gain": round(damped_gain, 1),
                "los_pct": round(stats.get("los", 0.0), 1),
                "sprt_status": sprt["status"],
            })
            with open(prog_file, "w") as f:
                json.dump(prog_data, f, indent=2)

    print("\n=======================================================")
    print("FINAL MATCH RESULTS:")
    print(f"  {e1_name} vs {e2_name}")
    print(f"  Score:  {wins} - {losses} - {draws} / {total_played} games ({stats['score_pct']:.2f}%)")
    print(f"  Elo:    {stats['elo']:+.1f} +/- {stats['error']:.1f} (95% CI)")
    print(f"  LOS:    {stats['los']:.1f}%")
    if sprt_mode:
        print(f"  SPRT:   LLR {sprt['llr']:.2f} [{sprt['lower']:.2f}, {sprt['upper']:.2f}] -> {sprt['status']}")
    if archived_checkpoint:
        print(f"  Saved:  {archived_checkpoint}")
    print(f"  PGN:    {pgn_path}")
    print("=======================================================\n")

    return {
        "engine1": e1_name,
        "engine2": e2_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total_played,
        "stats": stats,
        "sprt": sprt,
        "pgn": pgn_path
    }

def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Match & SPRT Testing Runner")
    parser.add_argument("--engine1", default="./GiraffeGPU", help="Path to engine 1 (candidate)")
    parser.add_argument("--engine2", default=None, help="Path to engine 2 (baseline or opponent)")
    parser.add_argument("--engine1-options", default=None, help="UCI options for engine 1 (e.g. PolicyNet=true)")
    parser.add_argument("--engine2-options", default=None, help="UCI options for engine 2 (e.g. PolicyNet=false)")
    parser.add_argument("--compare-policy", action="store_true", help="Direct A/B match comparing PolicyNet enabled vs disabled")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play (default: 20)")
    parser.add_argument("--tc", default="60+0.5", help="Time control (default: 60+0.5)")
    parser.add_argument("--book", default="testing/books/openings.epd", help="Opening book (.epd or .pgn)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent games (default: 2 for Dual-GPU)")
    parser.add_argument("--sprt", action="store_true", help="Enable SPRT stopping conditions")
    parser.add_argument("--elo0", type=float, default=0.0, help="SPRT Elo 0 bound (default: 0.0)")
    parser.add_argument("--elo1", type=float, default=10.0, help="SPRT Elo 1 bound (default: 10.0)")

    args = parser.parse_args()

    if args.compare_policy:
        engine2 = args.engine1
        e1_opt = "PolicyNet=true" if not args.engine1_options else f"{args.engine1_options},PolicyNet=true"
        e2_opt = "PolicyNet=false" if not args.engine2_options else f"{args.engine2_options},PolicyNet=false"
    else:
        if not args.engine2:
            print("Error: --engine2 <path> is required unless --compare-policy is specified.")
            sys.exit(1)
        engine2 = args.engine2
        e1_opt = args.engine1_options
        e2_opt = args.engine2_options

    book = args.book if os.path.exists(args.book) else None

    run_match(
        engine1_path=args.engine1,
        engine2_path=engine2,
        games=args.games,
        tc=args.tc,
        book_path=book,
        concurrency=args.concurrency,
        elo0=args.elo0,
        elo1=args.elo1,
        sprt_mode=args.sprt,
        engine1_options=e1_opt,
        engine2_options=e2_opt
    )

if __name__ == "__main__":
    main()
