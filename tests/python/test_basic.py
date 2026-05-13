"""Skeleton tests — fill these in as you implement the C++ core."""

import numpy as np
import pytest

from pypolca._core import Data, fit_em, compute_ylik, compute_prior_from_beta


class TestMathOps:
    """Unit tests for core math operations."""

    def test_ylik_signature(self):
        """Smoke test: compute_ylik runs and returns correct shape."""
        data = Data()
        data.y = np.array([[1, 1], [2, 2]], dtype=np.int32)
        data.x = np.ones((2, 1), dtype=np.float64)
        data.num_choices = [2, 2]

        from pypolca._core import Params
        p = Params()
        p.vecprobs = np.ones(8, dtype=np.float64) * 0.5
        p.beta = np.array([], dtype=np.float64)

        lik = compute_ylik(data, p, 2)
        assert lik.shape == (2, 2)
        # TODO: once implemented, assert actual values

    def test_prior_from_beta_shape(self):
        """compute_prior_from_beta returns correct shape."""
        x = np.ones((5, 1), dtype=np.float64)
        beta = np.zeros(2, dtype=np.float64)
        prior = compute_prior_from_beta(x, beta, 3)
        assert prior.shape == (5, 3)
        # TODO: assert correct probabilities once implemented


class TestFitEM:
    """Integration tests for the EM engine."""

    def test_em_runs(self):
        """fit_em executes without crashing."""
        np.random.seed(123)
        N = 50
        y = np.random.randint(1, 3, size=(N, 2), dtype=np.int32)

        data = Data()
        data.y = y
        data.x = np.ones((N, 1), dtype=np.float64)
        data.num_choices = [2, 2]

        res = fit_em(data, nclass=2, maxiter=5, tol=1e-6)
        assert res.posterior.shape == (N, 2)
        # TODO: assert convergence behavior once EM is implemented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
