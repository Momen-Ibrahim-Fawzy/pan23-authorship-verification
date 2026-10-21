"""Author-disjoint, balanced train/validation split (after Martinez-Galicia et al., PAN 2022).

Validating on problems whose authors also appear in training rewards memorising authors instead of
learning style. Here no author, and therefore no text, is shared between the two splits:

1. Collect every unique text with its author and discourse type (DT) from the problems.
2. Assign whole authors to train or validation.
3. Inside each split, with ``A`` the texts grouped by author:
   positives ``P = A x A`` (two different texts of one author) and negatives ``N = A x B`` (texts of two
   different authors).
4. Drop pairs whose two texts share a DT. PAN 2022/2023 problems are always cross-DT.
5. Sample the same number of positives and negatives.

On PAN 2022 + 2023 this gives the 18,968 / 2,132 problems (101 / 11 authors) used for model selection.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def collect_texts(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique text: ``text, author, dt``. Needs ``author1/2`` and ``dt1/2`` columns."""
    a = df[["text1", "author1", "dt1"]].set_axis(["text", "author", "dt"], axis=1)
    b = df[["text2", "author2", "dt2"]].set_axis(["text", "author", "dt"], axis=1)
    texts = pd.concat([a, b], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    return texts


def split_authors(authors: Sequence[str], val_fraction: float, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    authors = sorted(set(authors))
    n_val = max(1, int(round(len(authors) * val_fraction)))
    val = set(rng.choice(authors, size=n_val, replace=False).tolist())
    return [a for a in authors if a not in val], sorted(val)


def candidate_pairs(texts: pd.DataFrame, cross_dt: bool = True) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """All positive and negative index pairs among ``texts`` (each unordered pair once)."""
    positives, negatives = [], []
    by_author: Dict[str, List[int]] = texts.groupby("author").indices
    by_author = {k: list(v) for k, v in by_author.items()}
    dts = texts["dt"].tolist()

    def ok(i: int, j: int) -> bool:
        return not cross_dt or dts[i] != dts[j]

    for idx in by_author.values():
        positives += [(i, j) for i, j in itertools.combinations(idx, 2) if ok(i, j)]
    for (_, ia), (_, ib) in itertools.combinations(by_author.items(), 2):
        negatives += [(i, j) for i in ia for j in ib if ok(i, j)]
    return positives, negatives


def balanced_sample(positives, negatives, n_problems: Optional[int], rng: np.random.Generator):
    n_each = min(len(positives), len(negatives))
    if n_problems is not None:
        n_each = min(n_each, n_problems // 2)
    pos = [positives[i] for i in rng.choice(len(positives), size=n_each, replace=False)]
    neg = [negatives[i] for i in rng.choice(len(negatives), size=n_each, replace=False)]
    return pos, neg


def build_split(texts: pd.DataFrame, n_problems: Optional[int], rng: np.random.Generator,
                cross_dt: bool = True, prefix: str = "p") -> pd.DataFrame:
    positives, negatives = candidate_pairs(texts, cross_dt)
    pos, neg = balanced_sample(positives, negatives, n_problems, rng)
    records = texts.to_dict("records")
    rows = []
    for label, pairs in ((1, pos), (0, neg)):
        for i, j in pairs:
            if rng.random() < 0.5:  # random order inside a pair
                i, j = j, i
            a, b = records[i], records[j]
            rows.append({"label": label, "text1": a["text"], "text2": b["text"], "dt1": a["dt"], "dt2": b["dt"],
                         "author1": a["author"], "author2": b["author"]})
    out = pd.DataFrame(rows).sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    out.insert(0, "id", [f"{prefix}{k:06d}" for k in range(len(out))])
    return out


def author_disjoint_split(df: pd.DataFrame, val_fraction: float = 0.1, n_train: Optional[int] = 18968,
                          n_val: Optional[int] = 2132, cross_dt: bool = True,
                          seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    texts = collect_texts(df)
    train_authors, val_authors = split_authors(texts["author"], val_fraction, rng)
    train_texts = texts[texts["author"].isin(train_authors)].reset_index(drop=True)
    val_texts = texts[texts["author"].isin(val_authors)].reset_index(drop=True)
    train = build_split(train_texts, n_train, rng, cross_dt, prefix="train-")
    val = build_split(val_texts, n_val, rng, cross_dt, prefix="val-")
    return train, val


def check_disjoint(train: pd.DataFrame, val: pd.DataFrame) -> Dict[str, int]:
    """Counts of authors and texts shared between splits (both must be 0)."""
    ta = set(train["author1"]) | set(train["author2"])
    va = set(val["author1"]) | set(val["author2"])
    tt = set(train["text1"]) | set(train["text2"])
    vt = set(val["text1"]) | set(val["text2"])
    return {"shared_authors": len(ta & va), "shared_texts": len(tt & vt)}
