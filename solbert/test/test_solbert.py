import unittest

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier

import solbert
from solbert.forest import RandomForestEncoder, RandomForestWrapper


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


if __name__ == "__main__":
    unittest.main()