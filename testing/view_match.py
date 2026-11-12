#!/usr/bin/env python3
"""
GiraffeGPU Live Match & Gauntlet Terminal Viewer
Monitors the currently active (or most recent) SPRT promotion match in real time,
displaying live scores, Elo difference, LOS %, recent game outcomes, and ASCII board state.
"""

import sys
import os
import time
import glob
import argparse

try:
    import chess
    import chess.pgn
except ImportError:
    print("Error: python-chess is required. Install with: pip install chess")
    sys.exit(1)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTING_DIR = os.path.join(REPO_DIR, "testing")
RESULTS_DIR = os.path.join(TESTING_DIR, "results")
sys.path.insert(0, TESTING_DIR)

try:
    from match_sprt import compute_stats, compute_sprt
except Exception:
    def compute_stats(w, l, d):
        tot = w + l + d
        pct = (w + 0.5 * d) / tot * 100 if tot else 50.0
        return {"score_pct": pct, "elo": 0.0, "error": 0.0, "los": 50.0}
    def compute_sprt(w, l, d):
        return {"status": "CONTINUE", "llr": 0.0, "lower": -2.94, "upper": 2.94}


def find_latest_match_file():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "match_*.pgn")), key=os.path.getmtime)
    return files[-1] if files else None


def format_board_ascii(board):
    unicode_map = {
        'P': '♟', 'N': '♞', 'B': '♝', 'R': '♜', 'Q': '♛', 'K': '♚',
        'p': '♙', 'n': '♘', 'b': '♗', 'r': '♖', 'q': '♕', 'k': '♔',
        '.': '·'
    }
    lines = []
    lines.append("    +---+---+---+---+---+---+---+---+")
    for rank in range(7, -1, -1):
        row = [f" {board.piece_at(chess.square(file, rank)).symbol() if board.piece_at(chess.square(file, rank)) else '.'} " for file in range(8)]
        row_str = "|".join(row)
        lines.append(f"  {rank+1} |{row_str}|")
        lines.append("    +---+---+---+---+---+---+---+---+")
    lines.append("      a   b   c   d   e   f   g   h  ")
    return "\n".join(lines)


def render_match(pgn_path, total_games=40):
    if not pgn_path or not os.path.exists(pgn_path):
        print(f"No match PGN found in {RESULTS_DIR}")
        return

    games = []
    wins, losses, draws = 0, 0, 0

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            try:
                g = chess.pgn.read_game(f)
                if not g:
                    break
                games.append(g)
            except Exception:
                break

    for g in games:
        res = g.headers.get("Result", "*")
        black = g.headers.get("Black", "").lower()
        if "baseline" in black:  # Candidate is White
            if res == "1-0": wins += 1
            elif res == "0-1": losses += 1
            elif res == "1/2-1/2": draws += 1
        else:  # Candidate is Black
            if res == "0-1": wins += 1
            elif res == "1-0": losses += 1
            elif res == "1/2-1/2": draws += 1

    total_played = wins + losses + draws
    stats = compute_stats(wins, losses, draws)
    sprt = compute_sprt(wins, losses, draws, elo0=0.0, elo1=10.0)

    # Header
    print("\033[1;36m=======================================================================\033[0m")
    print("\033[1;32m                 GIRAFFEGPU LIVE MATCH / GAUNTLET MONITOR             \033[0m")
    print("\033[1;36m=======================================================================\033[0m")
    print(f"  \033[1mMatch PGN File:\033[0m    {os.path.basename(pgn_path)}")
    pct_complete = (total_played / total_games) * 100 if total_games else 0.0
    print(f"  \033[1mGames Played:\033[0m      {total_played} / {total_games} ({pct_complete:.1f}%)")

    # Scoreboard
    cand_color = "\033[1;32m" if wins > losses else ("\033[1;31m" if losses > wins else "\033[1;33m")
    score_line = f"  \033[1mScore (Cand vs Base):\033[0m {cand_color}{wins} Wins - {losses} Losses - {draws} Draws\033[0m ({stats['score_pct']:.1f}%)"
    print(score_line)

    elo_str = f"{stats['elo']:+.1f} +/- {stats['error']:.1f}" if stats.get('error') else f"{stats['elo']:+.1f}"
    los_val = stats.get('los', 50.0)
    los_color = "\033[1;32m" if los_val >= 80.0 else "\033[1;33m"
    print(f"  \033[1mPerformance Elo:\033[0m   {stats['elo']:+.1f} (95% CI: {elo_str})")
    print(f"  \033[1mConfidence (LOS):\033[0m  {los_color}{los_val:.1f}%\033[0m (Promotion Target: >= 80.0%)")

    # Status assessment
    if total_played >= 20 and wins > losses and los_val >= 80.0:
        status_banner = "\033[1;42;30m [PASSING / PROMOTION ON TRACK] \033[0m"
    elif wins > losses:
        status_banner = "\033[1;43;30m [LEADING (+)] \033[0m"
    elif wins < losses:
        status_banner = "\033[1;41;37m [TRAILING (-)] \033[0m"
    else:
        status_banner = "\033[1;44;37m [TIED (=)] \033[0m"
    print(f"  \033[1mMatch Trajectory:\033[0m  {status_banner}")
    print("\033[1;36m-----------------------------------------------------------------------\033[0m")

    # Recent Games
    if games:
        print("\033[1;33mRecent Completed Games (Latest 6):\033[0m")
        print(f"  {'#':<4} {'White':<22} {'Black':<22} {'Result':<10} {'Plies':<6}")
        recent = games[-6:]
        start_idx = max(1, total_played - len(recent) + 1)
        for i, g in enumerate(recent):
            w = g.headers.get("White", "White")
            b = g.headers.get("Black", "Black")
            res = g.headers.get("Result", "*")
            moves_count = len(list(g.mainline_moves()))
            print(f"  {start_idx + i:<4} {w[:21]:<22} {b[:21]:<22} {res:<10} {moves_count:<6}")
        print("\033[1;36m-----------------------------------------------------------------------\033[0m")

        # Board display for the latest game
        latest_game = games[-1]
        final_board = latest_game.end().board()
        latest_w = latest_game.headers.get("White", "White")
        latest_b = latest_game.headers.get("Black", "Black")
        latest_res = latest_game.headers.get("Result", "*")
        latest_moves = len(list(latest_game.mainline_moves()))

        print(f"\033[1;35mFinal Position of Last Game (#{total_played}: {latest_w} vs {latest_b} -> {latest_res}, {latest_moves} plies):\033[0m")
        print(format_board_ascii(final_board))
        print(f"  FEN: {final_board.fen()}")

    print("\033[1;36m=======================================================================\033[0m")


def main():
    parser = argparse.ArgumentParser(description="GiraffeGPU Live Match & Gauntlet Terminal Viewer")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuously refresh and watch live match in terminal")
    parser.add_argument("--interval", "-i", type=float, default=3.0, help="Refresh interval in seconds (default: 3.0)")
    parser.add_argument("--pgn", default=None, help="Specific PGN file to view (default: latest in testing/results/)")
    parser.add_argument("--games", type=int, default=40, help="Expected total games in match (default: 40)")

    args = parser.parse_args()

    if not args.watch:
        pgn = args.pgn or find_latest_match_file()
        render_match(pgn, total_games=args.games)
        return

    try:
        while True:
            pgn = args.pgn or find_latest_match_file()
            # Clear screen ANSI escape
            sys.stdout.write("\033[H\033[J")
            render_match(pgn, total_games=args.games)
            print(f"Watching live... Refreshing every {args.interval}s (Press Ctrl+C to exit)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExited match viewer.")


if __name__ == "__main__":
    main()
