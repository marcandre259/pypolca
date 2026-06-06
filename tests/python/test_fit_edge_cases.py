"""Edge-case and integration tests: covariates, nrep, polytomous items, missing data.

Tests the fit() function with more complex configurations.
"""

import numpy as np
import polars as pl
import pytest

from pypolca import fit
from pypolca._core import Data, Params, compute_ylik, m_step_probs

# ── helpers ────────────────────────────────────────────────────────


def make_data(y, x, num_choices):
    d = Data()
    d.y = np.array(y, dtype=np.int32)
    d.x = np.array(x, dtype=np.float64)
    d.num_choices = list(num_choices)
    return d


def make_params(vecprobs, beta=None):
    p = Params()
    p.vecprobs = np.array(vecprobs, dtype=np.float64)
    p.beta = np.array(beta if beta is not None else [], dtype=np.float64)
    return p


# ── covariates ─────────────────────────────────────────────────────


class TestFitWithCovariates:
    def test_binary_covariate_converges(self):
        """Fit with a binary covariate that predicts class membership."""
        np.random.seed(42)
        X = np.concatenate([np.zeros(100), np.ones(100)])
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.85, 100) + 1, np.random.binomial(1, 0.15, 100) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.80, 100) + 1, np.random.binomial(1, 0.20, 100) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "X": X})
        result = fit("cbind(Y1, Y2) ~ X", df, nclass=2, seed=42)
        assert result.converged
        assert np.isfinite(result.loglik)
        assert result.coeff.shape == (2, 1)

    def test_continuous_covariate_converges(self):
        """Fit with a continuous covariate."""
        np.random.seed(42)
        N = 200
        X = np.random.randn(N)
        # Higher X → higher prob of class 1 (which shows as Y=2)
        logit = X
        prob_class1 = 1.0 / (1.0 + np.exp(-logit))
        classes = np.random.binomial(1, prob_class1, N)
        Y1 = np.where(
            classes == 0, np.random.binomial(1, 0.85, N) + 1, np.random.binomial(1, 0.15, N) + 1
        )
        Y2 = np.where(
            classes == 0, np.random.binomial(1, 0.80, N) + 1, np.random.binomial(1, 0.20, N) + 1
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "X": X})
        result = fit("cbind(Y1, Y2) ~ X", df, nclass=2, seed=42)
        assert result.converged
        assert result.coeff.shape == (2, 1)

    def test_covariate_increases_loglik(self):
        """A predictive covariate should improve log-likelihood vs intercept-only."""
        np.random.seed(42)
        X = np.concatenate([np.zeros(100), np.ones(100)])
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.85, 100) + 1, np.random.binomial(1, 0.15, 100) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.80, 100) + 1, np.random.binomial(1, 0.20, 100) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "X": X})

        res_null = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        res_cov = fit("cbind(Y1, Y2) ~ X", df, nclass=2, seed=42)

        # Model with covariate should fit at least as well
        assert res_cov.loglik >= res_null.loglik - 1e-6

    def test_coeff_se_valid(self):
        """coeff_se should have same shape as coeff and be non-negative."""
        np.random.seed(42)
        X = np.concatenate([np.zeros(100), np.ones(100)])
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.85, 100) + 1, np.random.binomial(1, 0.15, 100) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.80, 100) + 1, np.random.binomial(1, 0.20, 100) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "X": X})
        result = fit("cbind(Y1, Y2) ~ X", df, nclass=2, seed=42)
        assert result.coeff_se.shape == result.coeff.shape
        assert (result.coeff_se >= 0).all()


# ── nrep (multiple replications) ───────────────────────────────────


class TestNrep:
    def test_nrep_converges(self):
        """nrep > 1 should converge and return a result."""
        np.random.seed(42)
        N = 100
        Y1 = np.random.binomial(1, 0.5, N) + 1
        Y2 = np.random.binomial(1, 0.5, N) + 1
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, nrep=5, seed=42)
        assert result.converged
        assert np.isfinite(result.loglik)

    def test_nrep_best_not_worse_than_single(self):
        """Best of nrep should be >= a single replication."""
        np.random.seed(42)
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.8, 50) + 1, np.random.binomial(1, 0.2, 50) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.7, 50) + 1, np.random.binomial(1, 0.3, 50) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})

        res_single = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, nrep=1, seed=42)
        res_multi = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, nrep=5, seed=42)

        assert res_multi.loglik >= res_single.loglik - 1e-8

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        np.random.seed(42)
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.8, 50) + 1, np.random.binomial(1, 0.2, 50) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.7, 50) + 1, np.random.binomial(1, 0.3, 50) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})

        r1 = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=123, nrep=1)
        r2 = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=123, nrep=1)

        assert r1.loglik == pytest.approx(r2.loglik)
        np.testing.assert_allclose(r1.posterior, r2.posterior, atol=1e-10)
        np.testing.assert_allclose(r1.P, r2.P, atol=1e-10)


# ── polytomous items ───────────────────────────────────────────────


class TestPolytomousItems:
    def test_three_categories_per_item(self):
        """Items with 3 response categories."""
        np.random.seed(42)
        # Class 0: high prob of category 1; Class 1: high prob of category 3
        Y1 = np.concatenate(
            [
                np.random.choice([1, 2, 3], 75, p=[0.8, 0.1, 0.1]),
                np.random.choice([1, 2, 3], 75, p=[0.1, 0.1, 0.8]),
            ]
        )
        Y2 = np.concatenate(
            [
                np.random.choice([1, 2, 3], 75, p=[0.7, 0.2, 0.1]),
                np.random.choice([1, 2, 3], 75, p=[0.1, 0.2, 0.7]),
            ]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        assert result.converged
        assert len(result.probs) == 2
        # Each item has 2 classes × 3 categories
        assert result.probs[0].shape == (2, 3)
        assert result.probs[1].shape == (2, 3)
        for p in result.probs:
            np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-10)

    def test_mixed_categories(self):
        """Items with different numbers of categories (2 and 3)."""
        np.random.seed(42)
        Y1 = np.concatenate(
            [
                np.random.choice([1, 2], 75, p=[0.85, 0.15]),
                np.random.choice([1, 2], 75, p=[0.15, 0.85]),
            ]
        )
        Y2 = np.concatenate(
            [
                np.random.choice([1, 2, 3], 75, p=[0.8, 0.1, 0.1]),
                np.random.choice([1, 2, 3], 75, p=[0.1, 0.1, 0.8]),
            ]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        assert result.converged
        assert len(result.probs) == 2
        assert result.probs[0].shape == (2, 2)
        assert result.probs[1].shape == (2, 3)
        # npar: 2*(2-1) + 2*(3-1) + (2-1) = 2 + 4 + 1 = 7
        assert result.npar == 7

    def test_npar_mixed_categories(self):
        """npar with mixed category counts."""
        np.random.seed(42)
        N = 100
        Y1 = np.random.choice([1, 2], N, p=[0.5, 0.5])
        Y2 = np.random.choice([1, 2, 3], N, p=[0.4, 0.3, 0.3])
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=3, seed=42)
        # npar: 3*(2-1) + 3*(3-1) + (3-1) = 3 + 6 + 2 = 11
        assert result.npar == 11


# ── missing data ───────────────────────────────────────────────────


class TestMissingData:
    def test_missing_values_converge(self):
        """Data with missing responses (coded as 0) should converge."""
        np.random.seed(42)
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.85, 75) + 1, np.random.binomial(1, 0.15, 75) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.80, 75) + 1, np.random.binomial(1, 0.20, 75) + 1]
        )
        # Punch some holes
        Y1[10] = 0
        Y2[25] = 0
        Y1[50] = 0
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        assert result.converged
        assert np.isfinite(result.loglik)
        # Nobs should exclude rows with any missing
        assert result.Nobs < 150
        assert result.Nobs > 0

    def test_missing_data_posterior_valid(self):
        """Posterior should sum to 1 even with missing data."""
        np.random.seed(42)
        N = 100
        Y1 = np.random.binomial(1, 0.5, N) + 1
        Y2 = np.random.binomial(1, 0.5, N) + 1
        Y1[5] = 0
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        np.testing.assert_allclose(result.posterior.sum(axis=1), np.ones(N), atol=1e-10)

    def test_missing_probs_se_with_missing(self):
        """SE computation should not crash with missing data."""
        np.random.seed(42)
        N = 100
        Y1 = np.random.binomial(1, 0.5, N) + 1
        Y2 = np.random.binomial(1, 0.5, N) + 1
        Y1[3] = 0
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42, calc_se=True)
        # SE should exist (not crash)
        assert len(result.probs_se) == 2
        assert result.P_se.shape == (2,)


# ── math ops edge cases ────────────────────────────────────────────


class TestMathOpsEdgeCases:
    def test_ylik_three_categories(self):
        """compute_ylik with 3 response categories."""
        d = make_data([[1, 2], [3, 1]], [[1.0], [1.0]], [3, 3])
        # Class 0: [0.7, 0.2, 0.1, 0.1, 0.8, 0.1]
        # Class 1: [0.1, 0.1, 0.8, 0.7, 0.2, 0.1]
        vec = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8, 0.7, 0.2, 0.1]
        p = make_params(vec)
        lik = compute_ylik(d, p, 2)
        expected = np.array(
            [
                [np.log(0.7) + np.log(0.8), np.log(0.1) + np.log(0.2)],
                [np.log(0.1) + np.log(0.1), np.log(0.8) + np.log(0.7)],
            ]
        )
        np.testing.assert_allclose(lik, expected)

    def test_mstep_three_categories(self):
        """m_step_probs with 3 categories."""
        d = make_data([[1, 1], [2, 2], [3, 3]], [[1.0], [1.0], [1.0]], [3, 3])
        # Each obs gets a different class
        posterior = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        vec = m_step_probs(d, posterior, [3, 3], 2)
        # Class 0 (obs 0,2): item 0 → [0.5, 0.0, 0.5], item 1 → [0.5, 0.0, 0.5]
        # Class 1 (obs 1):   item 0 → [0.0, 1.0, 0.0], item 1 → [0.0, 1.0, 0.0]
        expected = np.array(
            [0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        np.testing.assert_allclose(vec, expected, atol=1e-12)

    def test_mstep_three_classes(self):
        """m_step_probs with 3 classes."""
        d = make_data([[1, 1], [2, 2], [1, 2]], [[1.0], [1.0], [1.0]], [2, 2])
        # Each obs in a different class
        posterior = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        vec = m_step_probs(d, posterior, [2, 2], 3)
        # Class 0 (obs 0): [1, 1] → [1, 0], [1, 0]
        # Class 1 (obs 1): [2, 2] → [0, 1], [0, 1]
        # Class 2 (obs 2): [1, 2] → [1, 0], [0, 1]
        expected = np.array(
            [
                1.0,
                0.0,
                1.0,
                0.0,  # class 0
                0.0,
                1.0,
                0.0,
                1.0,  # class 1
                1.0,
                0.0,
                0.0,
                1.0,  # class 2
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(vec, expected, atol=1e-12)


# ── single class ───────────────────────────────────────────────────


class TestNclassOne:
    def test_single_class_converges(self):
        """nclass=1 should converge trivially."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.binomial(1, 0.5, 50) + 1,
                "Y2": np.random.binomial(1, 0.5, 50) + 1,
            }
        )
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        assert result.converged
        assert np.isfinite(result.loglik)

    def test_posterior_all_ones(self):
        """With one class, all posterior probabilities are 1."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.binomial(1, 0.5, 30) + 1,
                "Y2": np.random.binomial(1, 0.5, 30) + 1,
            }
        )
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        assert result.posterior.shape == (30, 1)
        np.testing.assert_allclose(result.posterior, np.ones((30, 1)), atol=1e-10)

    def test_probs_match_marginals(self):
        """With one class, probs should match observed marginal frequencies."""
        df = pl.DataFrame({"Y1": [1, 1, 2, 1, 1], "Y2": [2, 1, 1, 2, 1]})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        # Y1: 4 ones, 1 two → [0.8, 0.2]; Y2: 3 ones, 2 twos → [0.6, 0.4]
        np.testing.assert_allclose(result.probs[0][0], [0.8, 0.2], atol=1e-10)
        np.testing.assert_allclose(result.probs[1][0], [0.6, 0.4], atol=1e-10)

    def test_npar_single_class(self):
        """npar for nclass=1 is sum of (K_j - 1)."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.binomial(1, 0.5, 50) + 1,
                "Y2": np.random.binomial(1, 0.5, 50) + 1,
            }
        )
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        # 2 binary items: (2-1) + (2-1) = 2
        assert result.npar == 2

    def test_coeff_empty(self):
        """nclass=1 → no class contrast → empty coeff."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.binomial(1, 0.5, 50) + 1,
                "Y2": np.random.binomial(1, 0.5, 50) + 1,
            }
        )
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        assert result.coeff.size == 0
        assert result.coeff_se.size == 0

    def test_predict_posterior(self):
        """predict_posterior with nclass=1 should return all ones."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.binomial(1, 0.5, 30) + 1,
                "Y2": np.random.binomial(1, 0.5, 30) + 1,
            }
        )
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=1, seed=42)
        predicted = result.predict_posterior(df)
        np.testing.assert_allclose(predicted, np.ones((30, 1)), atol=1e-10)


# ── error handling ─────────────────────────────────────────────────


class TestFitErrors:
    def test_emtpy_data_raises(self):
        """Empty DataFrame should raise."""
        df = pl.DataFrame(
            {"Y1": pl.Series([], dtype=pl.Int64), "Y2": pl.Series([], dtype=pl.Int64)}
        )
        with pytest.raises((ValueError, RuntimeError)):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=2)

    def test_all_missing_raises(self):
        """All-zero y matrix should raise."""
        df = pl.DataFrame({"Y1": [0, 0, 0], "Y2": [0, 0, 0]})
        with pytest.raises((ValueError, RuntimeError)):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=2)

    def test_negative_values_raises(self):
        """Negative response values should raise."""
        df = pl.DataFrame({"Y1": [1, -1, 1], "Y2": [1, 2, 2]})
        with pytest.raises(ValueError, match="non-negative"):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=2)

    def test_nclass_too_large(self):
        """nclass=3 with N=2 is overparameterized → pre-fit ValueError."""
        np.random.seed(42)
        df = pl.DataFrame({"Y1": [1, 2], "Y2": [2, 1]})
        # nclass=3, N=2: npar = 3*2 + (3-1) = 8 > 2 → rejected before fitting
        with pytest.raises(ValueError, match="Number of parameters"):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=3, seed=42, nrep=1, max_restarts=1)


# ── large-ish data ─────────────────────────────────────────────────


class TestLargerData:
    def test_medium_dataset(self):
        """500 obs, 2 classes, 3 binary items — should converge quickly."""
        np.random.seed(42)
        Y1 = np.concatenate(
            [np.random.binomial(1, 0.9, 250) + 1, np.random.binomial(1, 0.1, 250) + 1]
        )
        Y2 = np.concatenate(
            [np.random.binomial(1, 0.8, 250) + 1, np.random.binomial(1, 0.2, 250) + 1]
        )
        Y3 = np.concatenate(
            [np.random.binomial(1, 0.7, 250) + 1, np.random.binomial(1, 0.3, 250) + 1]
        )
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "Y3": Y3})
        result = fit("cbind(Y1, Y2, Y3) ~ 1", df, nclass=2, seed=42)
        assert result.converged
        assert len(result.probs) == 3
        assert result.posterior.shape == (500, 2)


# ── pre-fit degrees of freedom check ───────────────────────────────


class TestPreFitDFCheck:
    """Pre-fit check: reject when npar > N (R poLCA prints ALERT post-fit)."""

    def test_npar_exceeds_N_intercept_only(self):
        """npar > N should raise ValueError before fitting (intercept-only)."""
        # 2 binary items, 3 classes: npar = 3*2 + 2 = 8, with N=7 → reject
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 7),
                "Y2": np.random.choice([1, 2], 7),
            }
        )
        with pytest.raises(ValueError, match="Number of parameters"):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=3, seed=42)

    def test_npar_exceeds_N_with_covariates(self):
        """npar > N should raise ValueError before fitting (with covariates)."""
        # 1 binary item, 2 classes, 1 covariate:
        # npar = 2*1 + (1+1)*1 = 2 + 2 = 4, with N=3 → reject
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 3),
                "X1": np.random.randn(3),
            }
        )
        with pytest.raises(ValueError, match="Number of parameters"):
            fit("cbind(Y1) ~ X1", df, nclass=2, seed=42)

    def test_npar_exceeds_N_two_covariates(self):
        """npar > N with multiple covariates."""
        # 2 binary items, 2 classes, 2 covariates:
        # npar = 2*2 + (1+2)*1 = 4 + 3 = 7, with N=6 → reject
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 6),
                "Y2": np.random.choice([1, 2], 6),
                "X1": np.random.randn(6),
                "X2": np.random.randn(6),
            }
        )
        with pytest.raises(ValueError, match="Number of parameters"):
            fit("Y1 + Y2 ~ X1 + X2", df, nclass=2, seed=42)

    def test_npar_equals_N_passes(self):
        """npar == N should not raise (barely identified model)."""
        # 2 binary items, 2 classes: npar = 2*2 + 1 = 5, N=5 → passes
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 5),
                "Y2": np.random.choice([1, 2], 5),
            }
        )
        # Should not raise
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42, nrep=1)
        assert result is not None
        assert np.isfinite(result.loglik)

    def test_npar_well_within_N_passes(self):
        """Normal case: npar << N should fit without issues."""
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 100),
                "Y2": np.random.choice([1, 2], 100),
            }
        )
        # npar = 5, N = 100 — well within bounds
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
        assert result.converged
        assert result.npar == 5

    def test_npar_equals_N_covariates_passes(self):
        """npar == N with covariates should pass."""
        # 1 binary item, 2 classes, 1 covariate:
        # npar = 2*1 + 2*1 = 4, N=4 → passes
        np.random.seed(42)
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 4),
                "X1": np.random.randn(4),
            }
        )
        result = fit("cbind(Y1) ~ X1", df, nclass=2, seed=42, nrep=1)
        assert result is not None
        assert np.isfinite(result.loglik)

    def test_error_message_includes_counts(self):
        """Error message should include both npar and N values."""
        df = pl.DataFrame(
            {
                "Y1": np.random.choice([1, 2], 3),
                "Y2": np.random.choice([1, 2], 3),
            }
        )
        # 2 binary items, nclass=3: npar = 3*2 + 2 = 8, N=3
        with pytest.raises(ValueError, match=r"Number of parameters \(8\).*observations \(3\)"):
            fit("cbind(Y1, Y2) ~ 1", df, nclass=3, seed=42)
