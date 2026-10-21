import unittest

import numpy as np

from av_sbert.stylometric import StylometricVerifier, frequency_features, shape_features

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

STYLES = [
    lambda i: f"Honestly, I think it's fine... like, really fine!!! number {i} lol",
    lambda i: f"The committee hereby confirms that item {i} has been approved; minutes follow.",
    lambda i: f"erm so yeah, um, item {i} was, erm, okay I guess, yeah",
]


class FeatureTests(unittest.TestCase):
    def test_feature_lengths_are_fixed(self):
        a = shape_features("Short text.") + frequency_features("Short text.")
        b = shape_features("A much longer text, with commas; and more!") + frequency_features("x")
        self.assertEqual(len(a), len(b))

    def test_empty_text(self):
        self.assertTrue(all(np.isfinite(shape_features(""))))


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class BaselineTests(unittest.TestCase):
    def test_separates_distinct_styles(self):
        rng = np.random.default_rng(0)
        t1, t2, y = [], [], []
        for k in range(300):
            a = rng.integers(3)
            same = rng.random() < 0.5
            b = a if same else (a + rng.integers(1, 3)) % 3
            t1.append(STYLES[a](k))
            t2.append(STYLES[b](k + 1000))
            y.append(int(same))
        model = StylometricVerifier(max_char_features=300, epochs=5).fit(t1[:200], t2[:200], y[:200])
        probs = model.predict_proba(t1[200:], t2[200:])
        acc = np.mean((probs > 0.5) == np.array(y[200:]))
        self.assertGreater(acc, 0.8)


if __name__ == "__main__":
    unittest.main()
