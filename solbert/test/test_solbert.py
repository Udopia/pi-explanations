import unittest

import solbert


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


if __name__ == "__main__":
    unittest.main()