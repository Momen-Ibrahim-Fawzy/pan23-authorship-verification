import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from av_sbert.chunking import hard_chunks, pair_chunks, soft_chunks
from av_sbert.data import combine, discourse_combinations, load_pan, read_answers, summary, to_pan_dir, write_answers
from av_sbert.metrics import auc, brier, c_at_1, evaluate_all, f1, f_05_u
from av_sbert.scoring import correct_scores, grid_search_p1p2, rescale
from av_sbert.split import author_disjoint_split, check_disjoint, collect_texts

DTS = ["essay", "email", "interview", "speech transcription"]


def synthetic_problems(n_authors=20, texts_per_author=4, n_problems=120, seed=0, prefix="en"):
    rng = np.random.default_rng(seed)
    texts = [(f"{prefix}_{a}", DTS[k % len(DTS)], f"text {k} by {prefix} author {a}.")
             for a in range(n_authors) for k in range(texts_per_author)]
    rows = []
    for i in range(n_problems):
        a, b = rng.choice(len(texts), 2, replace=False)
        t1, t2 = texts[a], texts[b]
        rows.append({"id": f"{prefix}-{i}", "text1": t1[2], "text2": t2[2], "dt1": t1[1], "dt2": t2[1],
                     "label": int(t1[0] == t2[0]), "author1": t1[0], "author2": t2[0]})
    return pd.DataFrame(rows)


class DataTests(unittest.TestCase):
    def test_pan_round_trip(self):
        df = synthetic_problems(n_problems=10)
        with tempfile.TemporaryDirectory() as tmp:
            to_pan_dir(df, tmp)
            back = load_pan(tmp)
            self.assertEqual(back["label"].tolist(), df["label"].tolist())
            self.assertEqual(back["dt1"].tolist(), df["dt1"].tolist())
            write_answers(back["id"], [0.1] * len(back), Path(tmp) / "out" / "answers.jsonl")
            self.assertEqual(len(read_answers(Path(tmp) / "out" / "answers.jsonl")), 10)

    def test_combine_keeps_authors_of_editions_apart(self):
        a = synthetic_problems(n_authors=5, prefix="en", seed=1)
        b = synthetic_problems(n_authors=5, prefix="en", seed=2)
        b["id"] = "b" + b["id"]
        combined = combine({"pan22": a, "pan23": b})
        self.assertEqual(summary(combined)["authors"], summary(a)["authors"] + summary(b)["authors"])
        with self.assertRaises(ValueError):
            combine({"x": a, "y": a})

    def test_discourse_combinations_are_unordered(self):
        df = pd.DataFrame({"dt1": ["essay", "email"], "dt2": ["email", "essay"]})
        self.assertEqual(discourse_combinations(df).to_dict(), {("email", "essay"): 2})


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_problems(n_authors=30, texts_per_author=4, n_problems=400)

    def test_author_disjoint_balanced_cross_dt(self):
        train, val = author_disjoint_split(self.df, val_fraction=0.2, n_train=200, n_val=40, seed=3)
        self.assertEqual(check_disjoint(train, val), {"shared_authors": 0, "shared_texts": 0})
        for split in (train, val):
            self.assertEqual(split["label"].sum() * 2, len(split))          # balanced
            self.assertTrue((split["dt1"] != split["dt2"]).all())           # cross-DT only
            same = split["author1"] == split["author2"]
            self.assertTrue((same == split["label"].astype(bool)).all())    # labels follow authors
            self.assertTrue((split["text1"] != split["text2"]).all())
        self.assertEqual(len(train), 200)

    def test_split_is_reproducible(self):
        a = author_disjoint_split(self.df, 0.2, 100, 20, seed=5)[1]
        b = author_disjoint_split(self.df, 0.2, 100, 20, seed=5)[1]
        pd.testing.assert_frame_equal(a, b)

    def test_collect_texts_dedups(self):
        self.assertEqual(collect_texts(self.df)["text"].is_unique, True)


class ChunkingTests(unittest.TestCase):
    tokenize = staticmethod(lambda t: t.split())
    detok = staticmethod(lambda toks: " ".join(toks))
    count = staticmethod(lambda t: len(t.split()))

    def test_hard_chunks(self):
        text = " ".join(f"w{i}" for i in range(10))
        self.assertEqual(hard_chunks(text, self.tokenize, self.detok, 4), ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"])
        self.assertEqual(hard_chunks("short", self.tokenize, self.detok, 4), ["short"])

    def test_soft_chunks_keep_sentences_whole(self):
        text = "One two three. Four five. Six seven eight nine. Ten."
        chunks = soft_chunks(text, self.count, 5)
        self.assertEqual(chunks, ["One two three. Four five.", "Six seven eight nine. Ten."])

    def test_soft_chunks_split_only_overlong_sentences(self):
        chunks = soft_chunks("a b c d e f g. h.", self.count, 3)
        self.assertTrue(all(self.count(c) <= 3 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), "a b c d e f g. h.".split())

    def test_pairing_wraps_shorter_document(self):
        self.assertEqual(pair_chunks(["a1", "a2", "a3"], ["b1"]), [("a1", "b1"), ("a2", "b1"), ("a3", "b1")])
        self.assertEqual(len(pair_chunks(["a1", "a2"], ["b1", "b2", "b3", "b4", "b5"])), 5)
        self.assertEqual(pair_chunks([], ["b"]), [])


class MetricTests(unittest.TestCase):
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.5, 0.2, 0.6, 0.7, 0.5]

    def test_against_hand_computed_values(self):
        # answered: (1,.9) (0,.2) (0,.6) (1,.7) -> 3 correct of 6, 2 unanswered
        self.assertAlmostEqual(c_at_1(self.y, self.s), (3 + 2 * 3 / 6) / 6)
        # tp=2 fp=1 fn=0 on answered
        self.assertAlmostEqual(f1(self.y, self.s), 4 / 5)
        # F0.5u: tp=2 fp=1 fn=0 nu=2
        self.assertAlmostEqual(f_05_u(self.y, self.s), 2.5 / (2.5 + 0.25 * 2 + 1))
        expected_brier = 1 - np.mean((np.array(self.s) - np.array(self.y)) ** 2)
        self.assertAlmostEqual(brier(self.y, self.s), expected_brier)

    def test_auc(self):
        self.assertAlmostEqual(auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertAlmostEqual(auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertEqual(auc([1, 1], [0.2, 0.9]), 0.0)
        try:
            from sklearn.metrics import roc_auc_score
        except Exception:
            return
        rng = np.random.default_rng(0)
        y, s = rng.integers(0, 2, 200), np.round(rng.random(200), 2)
        self.assertAlmostEqual(auc(y, s), roc_auc_score(y, s))

    def test_overall_is_mean(self):
        r = evaluate_all(self.y, self.s, digits=None)
        self.assertAlmostEqual(r["overall"], np.mean([r[k] for k in ("auc", "c@1", "f_05_u", "F1", "brier")]))


class ScoringTests(unittest.TestCase):
    def test_correct_scores_bands(self):
        out = correct_scores([-1.0, 0.45, 0.5, 0.54, 1.0], p1=0.45, p2=0.54)
        np.testing.assert_allclose(out, [0.0, 0.49, 0.5, 0.51, 1.0])
        self.assertAlmostEqual(rescale(0, -1, 1, 0, 1), 0.5)

    def test_monotonic_outside_band(self):
        s = np.linspace(-1, 1, 101)
        out = correct_scores(s, 0.3, 0.6)
        self.assertTrue(np.all(np.diff(out[s <= 0.3]) >= 0))
        self.assertTrue(np.all(np.diff(out[s >= 0.6]) >= 0))

    def test_grid_search_learns_a_useful_band(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 400)
        sims = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.15, 400)
        sims[:40] = 0.5 + rng.normal(0, 0.01, 40)  # an ambiguous block
        p1, p2, best = grid_search_p1p2(sims, y, step=0.05)
        self.assertLess(p1, p2)
        baseline = evaluate_all(y, correct_scores(sims, 0.0, 0.0))
        self.assertGreaterEqual(best["overall"], baseline["overall"])


if __name__ == "__main__":
    unittest.main()
