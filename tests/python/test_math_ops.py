"""Concise unit tests for C++ math_ops exposed via pybind11."""

import numpy as np
import pytest

from pypolca._core import (
    Data,
    Params,
    compute_ylik,
    e_step,
    m_step_probs,
    compute_prior_from_beta,
    update_beta,
    compute_beta_derivatives,
)


def make_data(y, x, num_choices):
    """Helper to build a pypolca.Data object."""
    d = Data()
    d.y = np.array(y, dtype=np.int32)
    d.x = np.array(x, dtype=np.float64)
    d.num_choices = list(num_choices)
    return d


def make_params(vecprobs, beta=None):
    """Helper to build a pypolca.Params object."""
    p = Params()
    p.vecprobs = np.array(vecprobs, dtype=np.float64)
    p.beta = np.array(beta if beta is not None else [], dtype=np.float64)
    return p


class TestComputeYlik:
    """Tests for compute_log_ylik."""

    def test_shape(self):
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        p = make_params([0.5] * 8)
        lik = compute_ylik(d, p, 2)
        assert lik.shape == (2, 2)

    def test_uniform_probs(self):
        """With all probs = 0.5 and 2 binary items, log-lik = 2 * log(0.5)."""
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        p = make_params([0.5] * 8)
        lik = compute_ylik(d, p, 2)
        expected = 2 * np.log(0.5)
        np.testing.assert_allclose(lik, np.full((2, 2), expected))

    def test_zero_prob_gives_neg_inf(self):
        """If observed category has probability 0, log-likelihood is -inf."""
        # Class 0: item 0 cat 1 = 1.0, item 0 cat 2 = 0.0, item 1 all 0.5
        vec = [1.0, 0.0, 0.5, 0.5,   # class 0
               0.5, 0.5, 0.5, 0.5]   # class 1
        d = make_data([[2, 1]], [[1.0]], [2, 2])
        p = make_params(vec)
        lik = compute_ylik(d, p, 2)
        assert lik[0, 0] == -np.inf
        assert np.isfinite(lik[0, 1])

    def test_missing_data_skipped(self):
        """Missing values (0) do not contribute to likelihood."""
        d = make_data([[1, 0]], [[1.0]], [2, 2])
        p = make_params([0.9, 0.1, 0.5, 0.5,
                         0.5, 0.5, 0.5, 0.5])
        lik = compute_ylik(d, p, 2)
        # class 0: log(0.9) + log(1.0) (missing -> no contribution)
        np.testing.assert_allclose(lik[0, 0], np.log(0.9))
        np.testing.assert_allclose(lik[0, 1], np.log(0.5))

    def test_hand_rolled_values(self):
        """Compare against explicit product-of-probs calculation."""
        # 2 items, 3 categories each, 2 classes
        # Class 0: [0.7, 0.2, 0.1,  0.1, 0.8, 0.1]
        # Class 1: [0.1, 0.1, 0.8,  0.7, 0.2, 0.1]
        vec = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1,
               0.1, 0.1, 0.8, 0.7, 0.2, 0.1]
        d = make_data([[1, 2], [3, 1]], [[1.0], [1.0]], [3, 3])
        p = make_params(vec)
        lik = compute_ylik(d, p, 2)

        expected = np.array([
            [np.log(0.7) + np.log(0.8), np.log(0.1) + np.log(0.2)],
            [np.log(0.1) + np.log(0.1), np.log(0.8) + np.log(0.7)],
        ])
        np.testing.assert_allclose(lik, expected)


class TestComputePriorFromBeta:
    """Tests for softmax prior computation."""

    def test_shape(self):
        x = np.ones((5, 2), dtype=np.float64)
        beta = np.zeros(4, dtype=np.float64)
        prior = compute_prior_from_beta(x, beta, 3)
        assert prior.shape == (5, 3)

    def test_rows_sum_to_one(self):
        x = np.random.randn(10, 3)
        beta = np.random.randn(6)
        prior = compute_prior_from_beta(x, beta, 3)
        np.testing.assert_allclose(prior.sum(axis=1), np.ones(10), atol=1e-12)

    def test_all_non_negative(self):
        x = np.random.randn(10, 3)
        beta = np.random.randn(6)
        prior = compute_prior_from_beta(x, beta, 3)
        assert (prior >= 0).all()

    def test_zero_beta_uniform(self):
        """With beta = 0, all priors equal 1 / nclass."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        beta = np.zeros(4, dtype=np.float64)
        prior = compute_prior_from_beta(x, beta, 3)
        np.testing.assert_allclose(prior, np.full((2, 3), 1.0 / 3.0))

    def test_nclass_one(self):
        prior = compute_prior_from_beta(np.ones((5, 2)), np.array([]), 1)
        np.testing.assert_allclose(prior, np.ones((5, 1)))

    def test_invalid_beta_size(self):
        with pytest.raises(ValueError, match="beta size"):
            compute_prior_from_beta(np.ones((5, 2)), np.zeros(3), 3)


class TestEStep:
    """Tests for posterior computation."""

    def test_posterior_rows_sum_to_one(self):
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        p = make_params([0.5] * 8)
        prior = np.full((2, 2), 0.5)
        post, loglik = e_step(d, p, prior, 2)
        np.testing.assert_allclose(post.sum(axis=1), np.ones(2), atol=1e-12)

    def test_equal_priors_equal_likelihoods(self):
        """When priors and likelihoods are identical across classes,
        posteriors are uniform."""
        d = make_data([[1, 1]], [[1.0]], [2, 2])
        p = make_params([0.5] * 8)
        prior = np.full((1, 2), 0.5)
        post, loglik = e_step(d, p, prior, 2)
        np.testing.assert_allclose(post, np.full((1, 2), 0.5))

    def test_extreme_prior_dominates(self):
        """If prior is [0.99, 0.01] and likelihoods are equal,
        posterior should be close to prior."""
        d = make_data([[1, 1]], [[1.0]], [2, 2])
        p = make_params([0.5] * 8)
        prior = np.array([[0.99, 0.01]])
        post, loglik = e_step(d, p, prior, 2)
        np.testing.assert_allclose(post[0], prior[0], atol=0.01)


class TestMStepProbs:
    """Tests for response probability updates."""

    def test_simple_two_class(self):
        """Hand-calculated M-step for 2 obs, 2 items, 2 classes."""
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        # posterior[i, r] = 0.5 for all
        posterior = np.full((2, 2), 0.5)
        vec = m_step_probs(d, posterior, [2, 2], 2)
        # Item 0: obs 0 chose 1, obs 1 chose 2
        # class 0: count cat 1 = 0.5, cat 2 = 0.5  -> [0.5, 0.5]
        # class 1: same
        # Item 1: same
        assert vec.shape == (8,)
        np.testing.assert_allclose(vec, np.full(8, 0.5))

    def test_asymmetric_posterior(self):
        """When obs 0 is definitely class 0 and obs 1 is definitely class 1."""
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        posterior = np.array([[1.0, 0.0], [0.0, 1.0]])
        vec = m_step_probs(d, posterior, [2, 2], 2)
        # class 0 item 0: only obs 0, chose 1 -> [1.0, 0.0]
        # class 0 item 1: only obs 0, chose 1 -> [1.0, 0.0]
        # class 1 item 0: only obs 1, chose 2 -> [0.0, 1.0]
        # class 1 item 1: only obs 1, chose 2 -> [0.0, 1.0]
        expected = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
        np.testing.assert_allclose(vec, expected)

    def test_missing_data_excluded(self):
        """Missing observation should not count in numerator or denominator."""
        d = make_data([[1, 0], [2, 1]], [[1.0], [1.0]], [2, 2])
        posterior = np.full((2, 1), 1.0)
        vec = m_step_probs(d, posterior, [2, 2], 1)
        # class 0 item 0: obs 0 (cat 1), obs 1 (cat 2) -> [0.5, 0.5]
        # class 0 item 1: obs 0 missing, obs 1 (cat 1) -> [1.0, 0.0]
        expected = [0.5, 0.5, 1.0, 0.0]
        np.testing.assert_allclose(vec, expected, atol=1e-12)


class TestBetaDerivativesAndUpdate:
    """Tests for Newton-Raphson beta update."""

    def test_nclass_one_returns_zero(self):
        d = make_data([[1, 1]], [[1.0]], [2, 2])
        post = np.ones((1, 1))
        prior = np.ones((1, 1))
        beta = np.array([], dtype=np.float64)
        grad, hess = compute_beta_derivatives(d, post, prior, beta, 1)
        assert grad.size == 0
        assert hess.size == 0

    def test_update_beta_changes_beta(self):
        """When gradient is non-zero, beta should change."""
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        # Start with uniform prior -> beta = 0
        beta = np.zeros(2, dtype=np.float64)
        prior = compute_prior_from_beta(d.x, beta, 3)
        # Make posterior different from prior
        posterior = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        new_beta, new_prior = update_beta(d, posterior, prior, beta, 3)
        assert not np.allclose(new_beta, beta)

    def test_updated_prior_is_stochastic(self):
        d = make_data([[1, 1], [2, 2]], [[1.0], [1.0]], [2, 2])
        beta = np.zeros(2, dtype=np.float64)
        prior = compute_prior_from_beta(d.x, beta, 3)
        posterior = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
        new_beta, new_prior = update_beta(d, posterior, prior, beta, 3)
        np.testing.assert_allclose(new_prior.sum(axis=1), np.ones(2), atol=1e-12)
        assert (new_prior >= 0).all()

    def test_invalid_beta_size(self):
        d = make_data([[1, 1]], [[1.0]], [2, 2])
        with pytest.raises(ValueError, match="beta size"):
            update_beta(d, np.ones((1, 2)), np.full((1, 2), 0.5),
                        np.zeros(2), 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
