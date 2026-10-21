"""Siamese Sentence-Transformer: contrastive fine-tuning on chunk pairs and chunk-averaged verification.

Training (paper, Section 5.3): all-MiniLM-L12-v2 (mean pooling, 384-d, 256-token window), ContrastiveLoss
on cosine distance, batch 32, AdamW at 2e-5 with eps 1e-6 and a linear schedule with no warm-up.
Every document pair is expanded into its chunk pairs, and each chunk pair is a training example
carrying the pair's label.

Inference: the two documents are chunked the same way, every chunk pair is embedded, and the pair
score is the mean cosine similarity of its chunk pairs. ``scoring.correct_scores`` then maps it to a
PAN answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .chunking import Chunker
from .metrics import evaluate_all
from .scoring import correct_scores, grid_search_p1p2


@dataclass
class TrainConfig:
    model: str = "sentence-transformers/all-MiniLM-L12-v2"
    output_dir: str = "runs/all-MiniLM-L12-v2"
    chunk_len: int = 256
    batch_size: int = 32
    epochs: int = 189
    learning_rate: float = 2e-5
    eps: float = 1e-6
    scheduler: str = "WarmupLinear"
    warmup_steps: int = 0
    eval_every: int = 1           # epochs between validation runs
    eval_batch_size: int = 64
    seed: int = 42


def expand_to_chunk_pairs(df: pd.DataFrame, chunker: Chunker) -> pd.DataFrame:
    """One row per chunk pair, each carrying its document pair's label (training data)."""
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="chunking pairs"):
        for a, b in chunker.pair(r.text1, r.text2):
            rows.append({"label": int(r.label), "text1": a, "text2": b, "pair_id": r.id})
    return pd.DataFrame(rows)


class Verifier:
    """Scores document pairs with a (fine-tuned) Sentence-Transformer."""

    def __init__(self, model_or_path: str, chunk_len: int = 256, chunking: str = "hard",
                 device: Optional[str] = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_or_path, device=device)
        self.model.max_seq_length = chunk_len
        self.chunker = Chunker(self.model.tokenizer, chunk_len, chunking)
        self.batch_size = batch_size

    @classmethod
    def from_model(cls, model, chunker: Chunker, batch_size: int = 64) -> "Verifier":
        """Wrap an in-memory model (used for validation during training)."""
        obj = cls.__new__(cls)
        obj.model, obj.chunker, obj.batch_size = model, chunker, batch_size
        return obj

    def similarities(self, pairs: Sequence[Tuple[str, str]], progress: bool = True) -> np.ndarray:
        chunk_a, chunk_b, owner = [], [], []
        for k, (a, b) in enumerate(tqdm(pairs, desc="chunking", disable=not progress)):
            for x, y in self.chunker.pair(a, b):
                chunk_a.append(x)
                chunk_b.append(y)
                owner.append(k)
        if not owner:
            return np.zeros(len(pairs))
        enc = dict(batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True,
                   show_progress_bar=progress)
        ea, eb = self.model.encode(chunk_a, **enc), self.model.encode(chunk_b, **enc)
        sims = (ea * eb).sum(axis=1)
        owner = np.asarray(owner)
        sums = np.bincount(owner, weights=sims, minlength=len(pairs))
        counts = np.bincount(owner, minlength=len(pairs))
        return np.divide(sums, counts, out=np.zeros(len(pairs)), where=counts > 0)

    def score_frame(self, df: pd.DataFrame, progress: bool = True) -> np.ndarray:
        return self.similarities(list(zip(df["text1"], df["text2"])), progress)


def evaluate_similarities(sims: np.ndarray, labels: Sequence[int], p1: Optional[float] = None,
                          p2: Optional[float] = None) -> Dict:
    """PAN metrics after correction; p1/p2 are searched on these problems when not given."""
    if p1 is None or p2 is None:
        p1, p2, _ = grid_search_p1p2(sims, labels)
    return {"p1": p1, "p2": p2, **evaluate_all(labels, correct_scores(sims, p1, p2))}


def train(cfg: TrainConfig, train_df: pd.DataFrame, val_df: pd.DataFrame,
          train_chunks: Optional[pd.DataFrame] = None) -> List[Dict]:
    """Fine-tune and keep the epoch with the best validation overall score in ``output_dir/best``.

    Thresholds are tuned on the training problems and applied unchanged to validation, so the validation
    score stays an honest estimate of the test score.
    """
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = SentenceTransformer(cfg.model)
    model.max_seq_length = cfg.chunk_len
    chunker = Chunker(model.tokenizer, cfg.chunk_len, "hard")
    if train_chunks is None:
        train_chunks = expand_to_chunk_pairs(train_df, chunker)
    examples = [InputExample(texts=[r.text1, r.text2], label=int(r.label)) for r in train_chunks.itertuples()]
    loader = DataLoader(examples, shuffle=True, batch_size=cfg.batch_size)
    loss = losses.ContrastiveLoss(model=model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE)

    out = Path(cfg.output_dir)
    history, best = [], -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.fit(train_objectives=[(loader, loss)], epochs=1, scheduler=cfg.scheduler,
                  warmup_steps=cfg.warmup_steps, optimizer_params={"lr": cfg.learning_rate, "eps": cfg.eps},
                  show_progress_bar=True)
        if epoch % cfg.eval_every and epoch != cfg.epochs:
            continue
        verifier = Verifier.from_model(model, chunker, cfg.eval_batch_size)
        train_sims = verifier.score_frame(train_df, progress=False)
        p1, p2, train_metrics = grid_search_p1p2(train_sims, train_df["label"])
        val_metrics = evaluate_similarities(verifier.score_frame(val_df, progress=False), val_df["label"], p1, p2)
        record = {"epoch": epoch, "train_overall": train_metrics["overall"], **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(record)
        if val_metrics["overall"] > best:
            best = val_metrics["overall"]
            model.save(str(out / "best"))
            (out / "best" / "thresholds.json").write_text(f'{{"p1": {p1}, "p2": {p2}}}')
    return history
