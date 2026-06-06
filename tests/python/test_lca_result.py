"""Tests for LCAResult properties: probs, coeff, aic/bic, SE, predict, predcell.

These test the Python-side LCAResult wrapper in pypolca/api.py.
"""

import numpy as np
import polars as pl
import pytest

from pypolca import fit

# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_model():
    """2-class intercept-only model on clean synthetic data."""
    np.random.seed(123)
    # First 50 obs: class 0 (high prob of Y=1), last 50: class 1 (high prob of Y=2)
    Y1 = np.concatenate([np.random.binomial(1, 0.85, 50) + 1, np.random.binomial(1, 0.15, 50) + 1])
    Y2 = np.concatenate([np.random.binomial(1, 0.80, 50) + 1, np.random.binomial(1, 0.20, 50) + 1])
    df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
    result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, seed=42)
    return result, df


@pytest.fixture
def covariate_model():
    """2-class model with a binary covariate that predicts class membership."""
    np.random.seed(42)
    X = np.concatenate([np.zeros(100), np.ones(100)])
    # X=0: mostly Y=1; X=1: mostly Y=2
    Y1 = np.concatenate(
        [np.random.binomial(1, 0.85, 100) + 1, np.random.binomial(1, 0.15, 100) + 1]
    )
    Y2 = np.concatenate(
        [np.random.binomial(1, 0.80, 100) + 1, np.random.binomial(1, 0.20, 100) + 1]
    )
    df = pl.DataFrame({"Y1": Y1, "Y2": Y2, "X": X})
    result = fit("cbind(Y1, Y2) ~ X", df, nclass=2, seed=42)
    return result, df


# ── probs property ─────────────────────────────────────────────────


class TestProbs:
    def test_shape(self, simple_model):
        result, _df = simple_model
        assert len(result.probs) == 2
        assert result.probs[0].shape == (2, 2)
        assert result.probs[1].shape == (2, 2)

    def test_valid_probabilities(self, simple_model):
        result, _df = simple_model
        for p in result.probs:
            assert (p >= 0).all() and (p <= 1).all()
            np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-10)


# ── coeff property ─────────────────────────────────────────────────


class TestCoeff:
    def test_shape_intercept_only(self, simple_model):
        result, _df = simple_model
        # Intercept-only has beta of length (R-1), coeff shape (1, R-1), values near 0
        assert result.coeff.shape == (1, 1)
        assert np.allclose(result.coeff, 0, atol=1e-10)

    def test_shape_with_covariates(self, covariate_model):
        result, _df = covariate_model
        # 2 covariates (intercept + X), 2 classes → (2, 1)
        assert result.coeff.shape == (2, 1)

    def test_covariate_coefficient_nonzero(self, covariate_model):
        result, _df = covariate_model
        # The coefficient for X (row 1) should be non-zero
        assert abs(result.coeff[1, 0]) > 0.1

    def test_coeff_se_matches_shape(self, covariate_model):
        result, _df = covariate_model
        assert result.coeff_se.shape == result.coeff.shape


# ── fit statistics ─────────────────────────────────────────────────


class TestFitStatistics:
    def test_aic_formula(self, simple_model):
        result, _df = simple_model
        expected = -2 * result.loglik + 2 * result.npar
        assert result.aic == pytest.approx(expected)

    def test_bic_formula(self, simple_model):
        result, _df = simple_model
        N = result.posterior.shape[0]
        expected = -2 * result.loglik + np.log(N) * result.npar
        assert result.bic == pytest.approx(expected)

    def test_bic_default_nobs(self, simple_model):
        result, _df = simple_model
        # bic (no arg) uses posterior.shape[0]
        assert result.bic == result.bic  # just ensure it doesn't crash

    def test_bic_formula_manual(self, simple_model):
        result, _df = simple_model
        N = result.posterior.shape[0]
        expected = -2 * result.loglik + np.log(N) * result.npar
        assert result.bic == pytest.approx(expected)

    def test_npar_intercept_only(self, simple_model):
        result, _df = simple_model
        # 2 classes, 2 items, 2 categories each:
        # probs: 2*1 + 2*1 = 4, class shares: (2-1) = 1 → 5 total
        assert result.npar == 5

    def test_npar_with_covariates(self, covariate_model):
        result, _df = covariate_model
        # 2 classes, 2 items, 2 categories each + 2*1 betas = 4 + 2 = 6
        assert result.npar == 6

    def test_loglik_finite(self, simple_model):
        result, _df = simple_model
        assert np.isfinite(result.loglik)
        assert result.loglik < 0

    def test_converged(self, simple_model):
        result, _df = simple_model
        assert result.converged

    def test_iterations_positive(self, simple_model):
        result, _df = simple_model
        assert result.iterations > 0

    def test_gsq_nonnegative(self, simple_model):
        result, _df = simple_model
        assert result.Gsq >= 0

    def test_chisq_nonnegative(self, simple_model):
        result, _df = simple_model
        assert result.Chisq >= 0

    def test_predclass_range(self, simple_model):
        result, _df = simple_model
        pc = result.predclass
        assert pc.min() >= 1
        assert pc.max() <= 2
        assert len(pc) == 100

    def test_P_sums_to_one(self, simple_model):
        result, _df = simple_model
        np.testing.assert_allclose(result.P.sum(), 1.0, atol=1e-10)

    def test_prior_shape(self, simple_model):
        result, _df = simple_model
        assert result.prior.shape == (100, 2)
        np.testing.assert_allclose(result.prior.sum(axis=1), np.ones(100), atol=1e-10)

    def test_Nobs(self, simple_model):
        result, _df = simple_model
        # All data is complete → Nobs == N
        assert result.Nobs == 100


# ── predict_posterior ──────────────────────────────────────────────


class TestPredictPosterior:
    def test_predict_on_training_data(self, simple_model):
        result, df = simple_model
        predicted = result.predict_posterior(df)
        # C++ round-trip through predict_posterior may differ by ~1e-7
        np.testing.assert_allclose(predicted, result.posterior, atol=1e-6)
        np.testing.assert_allclose(predicted.sum(axis=1), np.ones(100), atol=1e-10)

    def test_predict_new_data_no_covariates(self, simple_model):
        result, df = simple_model
        # Predict using flipped responses — posterior should change
        df_flipped = df.with_columns(
            pl.col("Y1").replace({1: 2, 2: 1}), pl.col("Y2").replace({1: 2, 2: 1})
        )
        predicted = result.predict_posterior(df_flipped)
        assert predicted.shape == (100, 2)
        np.testing.assert_allclose(predicted.sum(axis=1), np.ones(100), atol=1e-10)
        # Flipped responses should give different posteriors
        assert not np.allclose(predicted, result.posterior, atol=1e-5)

    def test_predict_with_covariates_same_x(self, covariate_model):
        result, df = covariate_model
        predicted = result.predict_posterior(df, newx=df)
        np.testing.assert_allclose(predicted, result.posterior, atol=1e-10)
        np.testing.assert_allclose(predicted.sum(axis=1), np.ones(200), atol=1e-10)

    def test_predict_with_covariates_different_x(self, covariate_model):
        result, df = covariate_model
        # Set all X to 0 → posterior should change
        df_new_x = df.with_columns(pl.lit(0.0).alias("X"))
        predicted = result.predict_posterior(df, newx=df_new_x)
        assert predicted.shape == (200, 2)
        np.testing.assert_allclose(predicted.sum(axis=1), np.ones(200), atol=1e-10)
        assert not np.allclose(predicted, result.posterior, atol=1e-5)

    def test_predict_missing_newx_raises(self, covariate_model):
        result, df = covariate_model
        with pytest.raises(ValueError, match="newx must be provided"):
            result.predict_posterior(df, newx=None)

    def test_predict_invalid_category_raises(self, simple_model):
        result, _df = simple_model
        df_bad = pl.DataFrame({"Y1": [3], "Y2": [1]})  # 3 is invalid for binary item
        with pytest.raises(ValueError, match="exceeds"):
            result.predict_posterior(df_bad)


# ── standard errors ────────────────────────────────────────────────


class TestStandardErrors:
    def test_probs_se_shape(self, simple_model):
        result, _df = simple_model
        assert len(result.probs_se) == 2
        assert result.probs_se[0].shape == (2, 2)
        assert result.probs_se[1].shape == (2, 2)

    def test_probs_se_nonnegative(self, simple_model):
        result, _df = simple_model
        for pse in result.probs_se:
            assert (pse >= 0).all()

    def test_P_se_shape(self, simple_model):
        result, _df = simple_model
        assert result.P_se.shape == (2,)
        assert (result.P_se >= 0).all()

    def test_coeff_se_intercept_only(self, simple_model):
        result, _df = simple_model
        # Intercept-only still has a beta SE; shape (1, R-1)
        assert result.coeff_se.shape == (1, 1)
        assert (result.coeff_se >= 0).all()

    def test_coeff_V_shape(self, covariate_model):
        result, _df = covariate_model
        # beta_V is the full covariance for beta params
        beta = np.array(result.params.beta)
        assert result.coeff_V.shape == (len(beta), len(beta))

    def test_no_se_when_calc_se_false(self):
        np.random.seed(99)
        N = 50
        Y1 = np.random.binomial(1, 0.5, N) + 1
        Y2 = np.random.binomial(1, 0.5, N) + 1
        df = pl.DataFrame({"Y1": Y1, "Y2": Y2})
        result = fit("cbind(Y1, Y2) ~ 1", df, nclass=2, calc_se=False, seed=42)
        # When calc_se=False, P_se and coeff_se should be empty/zero
        assert result.P_se.size == 0
        assert result.coeff_se.size == 0


# ── predcell / expected frequencies ────────────────────────────────


class TestPredcell:
    def test_returns_tuple(self, simple_model):
        result, _df = simple_model
        obs, exp, pats = result.predcell
        assert isinstance(obs, np.ndarray)
        assert isinstance(exp, np.ndarray)
        assert isinstance(pats, np.ndarray)

    def test_lengths_match(self, simple_model):
        result, _df = simple_model
        obs, exp, pats = result.predcell
        assert len(obs) == len(exp) == len(pats)

    def test_total_observed_matches(self, simple_model):
        result, _df = simple_model
        obs, exp, _pats = result.predcell
        # Total observed complete cases = Nobs
        assert obs.sum() == pytest.approx(result.Nobs)

    def test_resid_df_is_int(self, simple_model):
        result, _df = simple_model
        # resid_df can be negative for saturated models; just check type
        assert isinstance(result.resid_df, int)


# ── misc ────────────────────────────────────────────────────────────


class TestMisc:
    def test_repr(self, simple_model):
        result, _df = simple_model
        r = repr(result)
        assert "LCAResult" in r
        assert "loglik" in r

    def test_posterior_shape(self, simple_model):
        result, _df = simple_model
        assert result.posterior.shape == (100, 2)
        np.testing.assert_allclose(result.posterior.sum(axis=1), np.ones(100), atol=1e-10)
