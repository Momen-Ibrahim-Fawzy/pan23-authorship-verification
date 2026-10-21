"""Score an answers.jsonl against a PAN truth.jsonl with the official measures.

    python scripts/evaluate.py --truth data/prepared/val_pan/truth.jsonl --answers results/val/answers.jsonl
"""
import argparse
import json
from pathlib import Path

from av_sbert.data import read_answers, read_jsonl
from av_sbert.metrics import evaluate_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args()

    truth = {r["id"]: int(r["same"]) for r in read_jsonl(args.truth)}
    answers = read_answers(args.answers)
    missing = [i for i in truth if i not in answers]
    ids = list(truth)
    # a missing answer counts as a non-answer, as in the official evaluator
    scores = [answers.get(i, 0.5) for i in ids]
    result = evaluate_all([truth[i] for i in ids], scores)
    result["missing"] = len(missing)
    result["unanswered"] = sum(1 for s in scores if s == 0.5)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
