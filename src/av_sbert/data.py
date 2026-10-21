"""PAN authorship-verification data: reading, combining PAN 2022 + 2023, and the pair-table format.

PAN releases a problem set as two JSON-lines files::

    pairs.jsonl   {"id": ..., "discourse_types": [dt1, dt2], "pair": [text1, text2]}
    truth.jsonl   {"id": ..., "same": true|false, "authors": [a1, a2]}

and expects predictions as ``answers.jsonl`` with ``{"id": ..., "value": score}``, where ``score`` is in
[0, 1] and exactly 0.5 marks a problem left unanswered.

In this code base a set of problems is handled as a *pair table*: one row per problem with
``id, label, text1, text2`` (+ ``dt1, dt2, author1, author2`` when known), ``label = 1`` for same author.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: Iterable[dict], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_pan(directory: PathLike, truth: bool = True) -> pd.DataFrame:
    """Load ``pairs.jsonl`` (and ``truth.jsonl`` if present) from a PAN dataset folder into a pair table."""
    directory = Path(directory)
    rows = []
    for rec in read_jsonl(directory / "pairs.jsonl"):
        dts = rec.get("discourse_types") or [None, None]
        rows.append({"id": rec["id"], "text1": rec["pair"][0], "text2": rec["pair"][1],
                     "dt1": dts[0], "dt2": dts[1]})
    df = pd.DataFrame(rows)
    truth_path = directory / "truth.jsonl"
    if truth and truth_path.exists():
        t = pd.DataFrame(read_jsonl(truth_path))
        t["label"] = t["same"].astype(int)
        t["author1"] = t["authors"].map(lambda a: a[0])
        t["author2"] = t["authors"].map(lambda a: a[1])
        df = df.merge(t[["id", "label", "author1", "author2"]], on="id", how="left", validate="one_to_one")
    return df


def combine(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate several PAN editions; author ids are prefixed with the edition to keep them distinct.

    PAN 2022 and PAN 2023 both name authors ``en_<n>`` but they are different people, so ``en_12`` of 2022
    and ``en_12`` of 2023 must not be merged (the combined set has 56 + 56 = 112 authors).
    """
    parts = []
    for name, df in frames.items():
        df = df.copy()
        df["source"] = name
        for col in ("author1", "author2"):
            if col in df:
                df[col] = name + ":" + df[col].astype(str)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    if out["id"].duplicated().any():
        raise ValueError("duplicate problem ids across editions")
    return out


def read_pair_table(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_json(path, lines=True)
    missing = {"label", "text1", "text2"} - set(df.columns)
    if missing:
        raise KeyError(f"{path} lacks columns {missing}")
    df["text1"] = df["text1"].fillna("").astype(str)
    df["text2"] = df["text2"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def write_pair_table(df: pd.DataFrame, path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_json(path, orient="records", lines=True, force_ascii=False)


def to_pan_dir(df: pd.DataFrame, directory: PathLike) -> None:
    """Write a pair table back into PAN's ``pairs.jsonl`` / ``truth.jsonl`` layout (e.g. a validation split)."""
    directory = Path(directory)
    write_jsonl(({"id": r.id, "discourse_types": [r.dt1, r.dt2], "pair": [r.text1, r.text2]}
                 for r in df.itertuples()), directory / "pairs.jsonl")
    if "label" in df:
        write_jsonl(({"id": r.id, "same": bool(r.label),
                      "authors": [getattr(r, "author1", None), getattr(r, "author2", None)]}
                     for r in df.itertuples()), directory / "truth.jsonl")


def write_answers(ids: Sequence[str], scores: Sequence[float], path: PathLike) -> None:
    write_jsonl(({"id": i, "value": round(float(s), 6)} for i, s in zip(ids, scores)), path)


def read_answers(path: PathLike) -> Dict[str, float]:
    return {r["id"]: float(r["value"]) for r in read_jsonl(path)}


def discourse_combinations(df: pd.DataFrame) -> pd.Series:
    """Number of problems per (unordered) discourse-type pair, as in the paper's Tables 1, 2 and 4."""
    combo = df.apply(lambda r: tuple(sorted((str(r.dt1), str(r.dt2)))), axis=1)
    return combo.value_counts()


def summary(df: pd.DataFrame, name: Optional[str] = None) -> dict:
    texts = pd.concat([df["text1"], df["text2"]]).nunique()
    authors = pd.concat([df["author1"], df["author2"]]).nunique() if "author1" in df else None
    out = {"problems": len(df), "positives": int(df["label"].sum()) if "label" in df else None,
           "unique_texts": int(texts), "authors": authors}
    return {"name": name, **out} if name else out
