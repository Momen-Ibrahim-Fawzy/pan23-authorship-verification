"""Train and evaluate the stylometric feature-difference baseline on the prepared split.

    python scripts/train_stylometric.py --train data/prepared/train.jsonl --val data/prepared/val.jsonl
"""
import argparse
import json
import pickle
from pathlib import Path

from av_sbert.data import read_pair_table
from av_sbert.metrics import evaluate_all
from av_sbert.stylometric import StylometricVerifier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/stylometric"))
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    train, val = read_pair_table(args.train), read_pair_table(args.val)
    model = StylometricVerifier(epochs=args.epochs).fit(train["text1"], train["text2"], train["label"])
    probs = model.predict_proba(val["text1"], val["text2"])
    result = evaluate_all(val["label"], probs)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    (args.out / "val_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
