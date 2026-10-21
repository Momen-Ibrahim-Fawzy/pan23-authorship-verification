"""Fine-tune the Sentence-Transformer verifier on the prepared split.

    python scripts/train.py --config configs/minilm_l12.yaml
    python scripts/train.py --config configs/minilm_l12.yaml --model sentence-transformers/nli-distilroberta-base-v2 \
        --chunk-len 512 --output-dir runs/nli-distilroberta-base-v2
"""
import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from av_sbert.data import read_pair_table
from av_sbert.model import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/minilm_l12.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--chunk-len", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = raw.pop("data")
    cfg = TrainConfig(**{k: v for k, v in raw.items() if k in {f.name for f in fields(TrainConfig)}})
    for key in ("model", "chunk_len", "epochs", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    train_df, val_df = read_pair_table(data["train"]), read_pair_table(data["val"])
    history = train(cfg, train_df, val_df)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    best = max(history, key=lambda r: r["val_overall"])
    print("best epoch:", json.dumps(best))


if __name__ == "__main__":
    main()
