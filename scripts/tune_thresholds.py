"""Grid-search the non-answer band (p1, p2) for a trained model on a labelled split.

    python scripts/tune_thresholds.py --model runs/all-MiniLM-L12-v2/best --pairs data/prepared/val.jsonl --save
"""
import argparse
import json
from pathlib import Path

from av_sbert.data import read_pair_table
from av_sbert.metrics import evaluate_all
from av_sbert.model import Verifier
from av_sbert.scoring import correct_scores, grid_search_p1p2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--save", action="store_true", help="write thresholds.json next to the model")
    args = ap.parse_args()

    df = read_pair_table(args.pairs)
    sims = Verifier(args.model, args.chunk_len).score_frame(df)
    no_band = evaluate_all(df["label"], correct_scores(sims, 0.0, 0.0))
    p1, p2, best = grid_search_p1p2(sims, df["label"], step=args.step)
    print("without non-answers:", no_band)
    print(f"best p1={p1:.2f} p2={p2:.2f}:", best)
    if args.save:
        Path(args.model, "thresholds.json").write_text(json.dumps({"p1": p1, "p2": p2}))


if __name__ == "__main__":
    main()
