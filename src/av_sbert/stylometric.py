"""Stylometric baseline: feature-vector difference + logistic regression (after Weerasinghe et al., 2021).

Each text becomes a vector of style features. A pair is represented by the absolute difference of its
two scaled vectors, and a logistic-regression classifier trained with SGD maps that difference to
P(same author).

The paper evaluates the stylometric model of Weerasinghe et al. as a baseline. This module implements
the same feature-vector-difference scheme, self-contained and using the feature families that need no
external tagger:

* character 1–3-grams (TF-IDF, top ``max_char_features``)
* punctuation and special-character frequencies
* function-word frequencies
* simple shape statistics (word/sentence length, upper-case ratio, digit ratio, line breaks)
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

FUNCTION_WORDS = (
    "a about above after again against all am an and any are as at be because been before being below "
    "between both but by can could did do does doing down during each few for from further had has have "
    "having he her here hers herself him himself his how i if in into is it its itself just me more most "
    "my myself no nor not now of off on once only or other our ours ourselves out over own same she should "
    "so some such than that the their theirs them themselves then there these they this those through to "
    "too under until up very was we were what when where which while who whom why will with would you your "
    "yours yourself yourselves um erm yeah like okay"
).split()
PUNCT = list(string.punctuation) + ["\n", "\t", "...", "--"]
_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")


def shape_features(text: str) -> List[float]:
    words = _WORD.findall(text)
    n_chars = max(len(text), 1)
    n_words = max(len(words), 1)
    sentences = [s for s in _SENT.split(text) if s.strip()]
    return [
        np.mean([len(w) for w in words]) if words else 0.0,
        n_words / max(len(sentences), 1),
        sum(c.isupper() for c in text) / n_chars,
        sum(c.isdigit() for c in text) / n_chars,
        text.count("\n") / n_chars,
        len(set(w.lower() for w in words)) / n_words,   # type-token ratio
    ]


def frequency_features(text: str) -> List[float]:
    n_chars = max(len(text), 1)
    words = [w.lower() for w in _WORD.findall(text)]
    n_words = max(len(words), 1)
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return [text.count(p) / n_chars for p in PUNCT] + [counts.get(w, 0) / n_words for w in FUNCTION_WORDS]


@dataclass
class StylometricVerifier:
    max_char_features: int = 3000
    alpha: float = 1e-4
    epochs: int = 20
    seed: int = 42

    def _dense(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([shape_features(t) + frequency_features(t) for t in texts], dtype=float)

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        chars = self.char_vectorizer.transform(texts).toarray()
        return self.scaler.transform(np.hstack([chars, self._dense(texts)]))

    def _pair_features(self, texts1: Sequence[str], texts2: Sequence[str]) -> np.ndarray:
        return self.diff_scaler.transform(np.abs(self._vectors(texts1) - self._vectors(texts2)))

    def fit(self, texts1: Sequence[str], texts2: Sequence[str], labels: Sequence[int]) -> "StylometricVerifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        texts = list(texts1) + list(texts2)
        self.char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=self.max_char_features,
                                               sublinear_tf=True, lowercase=False)
        chars = self.char_vectorizer.fit_transform(texts).toarray()
        self.scaler = StandardScaler().fit(np.hstack([chars, self._dense(texts)]))
        diff = np.abs(self._vectors(texts1) - self._vectors(texts2))
        self.diff_scaler = StandardScaler().fit(diff)
        X, y = self.diff_scaler.transform(diff), np.asarray(labels, dtype=int)
        self.clf = SGDClassifier(loss="log_loss", alpha=self.alpha, random_state=self.seed)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(order), 128):
                idx = order[i:i + 128]
                self.clf.partial_fit(X[idx], y[idx], classes=[0, 1])
        return self

    def predict_proba(self, texts1: Sequence[str], texts2: Sequence[str]) -> np.ndarray:
        return self.clf.predict_proba(self._pair_features(texts1, texts2))[:, 1]
