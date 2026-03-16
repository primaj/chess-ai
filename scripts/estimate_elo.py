"""
Estimate ELO rating of the trained transformer model by playing vs Stockfish at set ELO levels.
Run from project root: python scripts/estimate_elo.py --stockfish /path/to/stockfish [options]
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from src.inference import load_model
    from src.game_runner import (
        play_game,
        make_model_player,
        make_uci_engine_player,
        find_stockfish,
    )

    parser = argparse.ArgumentParser(
        description="Estimate model ELO by playing vs Stockfish at configurable ELO levels."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/minichess_transformer_best.pt",
        help="Path to model checkpoint (default: models/minichess_transformer_best.pt)",
    )
    parser.add_argument(
        "--stockfish",
        type=str,
        default=None,
        help="Path to Stockfish binary (default: auto-detect from PATH or /usr/bin/stockfish)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per ELO level (default: 100)",
    )
    parser.add_argument(
        "--elo",
        type=int,
        nargs="+",
        default=[1000, 1200, 1400],
        help="Stockfish ELO levels to test (default: 1000 1200 1400)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path to write results",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.1,
        help="Time per move for Stockfish in seconds (default: 0.1)",
    )
    args = parser.parse_args()

    stockfish_path = args.stockfish or find_stockfish()
    if not stockfish_path or not os.path.isfile(stockfish_path):
        print("Stockfish not found. Install it (e.g. sudo apt install stockfish) and pass --stockfish /path/to/stockfish")
        sys.exit(1)

    if not os.path.isfile(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model, info = load_model(args.model)
    move_encoder = info.get("move_encoder")
    if move_encoder is None:
        print("Warning: No move encoder in checkpoint; move quality may be poor.")
    model_player = make_model_player(model, move_encoder)

    results_per_level = []
    games_per_level = args.games
    elo_levels = sorted(args.elo)

    for elo in elo_levels:
        print(f"\n--- Stockfish ELO {elo} ---")
        engine_player, engine = make_uci_engine_player(stockfish_path, elo=elo, time_limit=args.time)
        wins = draws = losses = 0
        n = games_per_level
        half = n // 2
        try:
            for i in range(n):
                model_is_white = i < half
                white = model_player if model_is_white else engine_player
                black = engine_player if model_is_white else model_player
                result, move_count, term = play_game(white, black)
                if result == "1-0":
                    if model_is_white:
                        wins += 1
                    else:
                        losses += 1
                elif result == "0-1":
                    if model_is_white:
                        losses += 1
                    else:
                        wins += 1
                else:
                    draws += 1
                if (i + 1) % 10 == 0 or i == 0:
                    print(f"  Games {i + 1}/{n} (W/D/L): {wins}/{draws}/{losses}")
        finally:
            engine.quit()

        total = wins + draws + losses
        score = (wins + 0.5 * draws) / total if total else 0.0
        s_clip = max(0.01, min(0.99, score))
        try:
            est_elo = elo + 400 * math.log10(s_clip / (1 - s_clip))
        except (ZeroDivisionError, ValueError):
            est_elo = float("nan")
        results_per_level.append({
            "elo": elo,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": score,
            "est_model_elo": round(est_elo, 0),
        })
        print(f"  Result: W={wins} D={draws} L={losses}  Score={score:.2%}  Est. model ELO: {est_elo:.0f}")

    # Report table
    print("\n" + "=" * 60)
    print("ELO estimation summary")
    print("=" * 60)
    print(f"{'Opponent ELO':<14} {'W':<6} {'D':<6} {'L':<6} {'Score':<8} {'Est. model ELO':<16}")
    print("-" * 60)
    for r in results_per_level:
        print(f"{r['elo']:<14} {r['wins']:<6} {r['draws']:<6} {r['losses']:<6} {r['score']:<8.2%} {r['est_model_elo']:<16.0f}")
    if len(results_per_level) > 1:
        avg_elo = sum(r["est_model_elo"] for r in results_per_level) / len(results_per_level)
        print("-" * 60)
        print(f"  Average estimated model ELO: {avg_elo:.0f}")

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["elo", "wins", "draws", "losses", "score", "est_model_elo"])
            w.writeheader()
            w.writerows(results_per_level)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
