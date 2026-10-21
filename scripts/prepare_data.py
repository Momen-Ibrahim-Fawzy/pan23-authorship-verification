"""Combine the PAN 2022 and PAN 2023 training sets and build the author-disjoint split.

    python scripts/prepare_data.py \
        --pan22 data/pan22-authorship-verification-training-dataset \
        --pan23 data/pan23-authorship-verification-training-dataset \
        --out data/prepared

Writes combined.jsonl, train.jsonl, val.jsonl and val_pan/ (the validation split in PAN layout, for
scripts/evaluate.py and the stylometric baseline), then prints the dataset tables of the paper.
"""
import argparse
import json
from pathlib import Path

from av_sbert.data import combine, discourse_combinations, load_pan, summary, to_pan_dir, write_pair_table
from av_sbert.split import author_disjoint_split, check_disjoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pan22", type=Path, required=True)
    ap.add_argument("--pan23", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/prepared"))
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=18968)
    ap.add_argument("--n-val", type=int, default=2132)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = {"pan22": load_pan(args.pan22), "pan23": load_pan(args.pan23)}
    for name, df in frames.items():
        print(name, summary(df))
    combined = combine(frames)
    print("combined", summary(combined))
    print(discourse_combinations(combined).to_string(), "\n")

    train, val = author_disjoint_split(combined, args.val_fraction, args.n_train, args.n_val, seed=args.seed)
    leaks = check_disjoint(train, val)
    assert leaks == {"shared_authors": 0, "shared_texts": 0}, leaks

    write_pair_table(combined, args.out / "combined.jsonl")
    write_pair_table(train, args.out / "train.jsonl")
    write_pair_table(val, args.out / "val.jsonl")
    to_pan_dir(val, args.out / "val_pan")
    stats = {"train": summary(train), "val": summary(val)}
    (args.out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
