"""PAN/TIRA entry point: read ``pairs.jsonl`` from an input folder, write ``answers.jsonl``.

    av-predict -i $inputDataset -o $outputDir --model runs/all-MiniLM-L12-v2/best [--p1 0.45 --p2 0.54] [--soft]

The three submitted runs map to:
    resolving-globe  default thresholds, hard chunking
    reduced-graph    same model, thresholds passed with --p1/--p2
    golden-ottoman   same model, --soft (sentence-aware chunking)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_pan, write_answers
from .scoring import correct_scores

DEFAULT_P1, DEFAULT_P2 = 0.45, 0.54


def predict_main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, type=Path, help="folder with pairs.jsonl")
    ap.add_argument("-o", "--output", required=True, type=Path, help="folder for answers.jsonl")
    ap.add_argument("--model", default="runs/all-MiniLM-L12-v2/best")
    ap.add_argument("--p1", type=float)
    ap.add_argument("--p2", type=float)
    ap.add_argument("--soft", action="store_true", help="sentence-aware chunking")
    ap.add_argument("--chunk-len", type=int, default=256)
    args = ap.parse_args(argv)

    from .model import Verifier

    p1, p2 = DEFAULT_P1, DEFAULT_P2
    saved = Path(args.model) / "thresholds.json"
    if saved.exists():
        t = json.loads(saved.read_text())
        p1, p2 = t["p1"], t["p2"]
    p1 = args.p1 if args.p1 is not None else p1
    p2 = args.p2 if args.p2 is not None else p2

    df = load_pan(args.input, truth=False)
    verifier = Verifier(args.model, args.chunk_len, "soft" if args.soft else "hard")
    scores = correct_scores(verifier.score_frame(df), p1, p2)
    write_answers(df["id"].tolist(), scores, args.output / "answers.jsonl")
    print(f"wrote {len(df)} answers (p1={p1}, p2={p2}, {'soft' if args.soft else 'hard'} chunking)")


if __name__ == "__main__":
    predict_main()
