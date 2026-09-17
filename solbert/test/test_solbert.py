import unittest

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import solbert
from solbert.forest import (
    RandomForestEncoder,
    RandomForestExplainer,
    RandomForestWrapper,
)
from solbert.tree import DecisionTreeExplainer, DecisionTreeWrapper


class SolbertApiTest(unittest.TestCase):
    def test_exports(self):
        self.assertTrue(callable(solbert.compute_prime_implicants))
        self.assertTrue(callable(solbert.compute_prime_implicants2))
        self.assertTrue(callable(solbert.enumerate_models))
        self.assertTrue(hasattr(solbert, "model_iterator"))
        self.assertTrue(hasattr(solbert, "monotonic_circuit"))

    def test_prime_implicants(self):
        self.assertEqual(solbert.compute_prime_implicants([[1]], [1]), [[-1]])
        self.assertEqual(solbert.compute_prime_implicants([[1], [-1]], [1]), [])

    def test_model_enumeration(self):
        self.assertEqual(solbert.enumerate_models([[1]], [1]), [[1]])
        self.assertEqual(solbert.enumerate_models([[1], [-1]], [1]), [])

    def test_python_package_exports(self):
        from solbert.forest import (
            RandomForestEncoder,
            RandomForestExplainer,
            RandomForestWrapper,
        )
        from solbert.tree import (
            DecisionTreeEncoder,
            DecisionTreeExplainer,
            DecisionTreeWrapper,
        )

        for exported_class in (
            DecisionTreeEncoder,
            DecisionTreeExplainer,
            DecisionTreeWrapper,
            RandomForestEncoder,
            RandomForestExplainer,
            RandomForestWrapper,
        ):
            self.assertTrue(callable(exported_class))

    def test_random_forest_encoder_ignores_raw_samples(self):
        lhs = pl.DataFrame({"feature": ["empty", "1", "2", "3"]})
        rhs = pl.Series("target", ["a", "a", "b", "b"]).cast(pl.Categorical)
        classifier = RandomForestClassifier(n_estimators=2, random_state=0)
        classifier.fit(
            np.array([[-1.0], [1.0], [2.0], [3.0]]),
            rhs.to_physical().to_numpy(),
        )

        encoder = RandomForestEncoder(RandomForestWrapper(classifier, lhs, rhs))
        self.addCleanup(encoder.pool.terminate)

        self.assertGreater(len(encoder.clauses), 0)

    def test_decision_tree_explains_prediction_from_cached_implicants(self):
        samples = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        lhs = pl.DataFrame(samples, schema=["a", "b"])
        rhs = pl.Series(
            "target", ["negative", "positive", "positive", "positive"]
        ).cast(pl.Categorical)
        classifier = DecisionTreeClassifier(max_depth=2, random_state=0)
        classifier.fit(samples, rhs.to_physical().to_numpy())
        explainer = DecisionTreeExplainer(
            "", None, DecisionTreeWrapper(classifier, lhs, rhs)
        )
        self.assertEqual(explainer.implicants, {})

        category, reasons = explainer.explain_prediction([1.0, -1.0])

        self.assertEqual(category, "positive")
        self.assertEqual(set(explainer.implicants), {category})
        self.assertGreater(len(reasons), 0)
        self.assertLess(len(reasons), len(explainer.implicants[category]))
        sample_literals = explainer.encoder.sample_literals([1.0, -1.0])
        self.assertTrue(all(set(reason).issubset(sample_literals) for reason in reasons))
        cached = explainer.implicants[category]
        explainer.explain_prediction([1.0, -1.0])
        self.assertIs(explainer.implicants[category], cached)

    def test_random_forest_explains_prediction_from_cached_implicants(self):
        samples = np.array([[-1.0], [1.0], [2.0], [3.0]])
        lhs = pl.DataFrame({"feature": ["empty", "1", "2", "3"]})
        rhs = pl.Series("target", ["a", "a", "b", "b"]).cast(pl.Categorical)
        classifier = RandomForestClassifier(n_estimators=2, random_state=0)
        classifier.fit(samples, rhs.to_physical().to_numpy())
        wrapper = RandomForestWrapper(classifier, lhs, rhs)
        self.assertEqual(
            wrapper.feature_values(0), sorted(set(wrapper.feature_values(0)))
        )
        explainer = RandomForestExplainer("", None, wrapper)
        self.addCleanup(explainer.encoder.pool.terminate)
        self.assertEqual(explainer.implicants, {})

        sample = [-1.0]
        category, reasons = explainer.explain_prediction(sample)

        self.assertEqual(set(explainer.implicants), {category})
        self.assertGreater(len(reasons), 0)
        sample_literals = explainer.encoder.sample_literals(sample)
        self.assertTrue(all(set(reason).issubset(sample_literals) for reason in reasons))
        cached = explainer.implicants[category]
        explainer.explain_prediction(sample)
        self.assertIs(explainer.implicants[category], cached)


if __name__ == "__main__":
    unittest.main()