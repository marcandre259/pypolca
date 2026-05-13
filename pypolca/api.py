"""High-level Python API for pypoLCA."""

from typing import Optional
import numpy as np
import polars as pl

from pypolca._core import Data, fit_em
from pypolca.utils import build_design_matrix


class LCAResult:
    """Python-friendly wrapper around the C++ Results struct."""

    def __init__(self, raw_results, formula=None, data=None):
        self._raw = raw_results
        self.formula = formula
        self.data = data

    @property
    def loglik(self) -> float:
        return self._raw.loglik

    @property
    def iterations(self) -> int:
        return self._raw.iterations

    @property
    def converged(self) -> bool:
        return self._raw.converged

    @property
    def posterior(self) -> np.ndarray:
        return np.array(self._raw.posterior)

    @property
    def prior(self) -> np.ndarray:
        return np.array(self._raw.prior)

    @property
    def predclass(self) -> np.ndarray:
        return self.posterior.argmax(axis=1) + 1  # 1-based class labels

    @property
    def params(self):
        """Raw fitted parameters (vecprobs, beta)."""
        return self._raw.params

    @property
    def P(self) -> np.ndarray:
        """Class population shares (marginal prior probabilities)."""
        return self.posterior.mean(axis=0)

    @property
    def npar(self) -> int:
        # TODO: compute properly once covariates/SEs are implemented
        return 0

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.npar

    @property
    def bic(self, nobs: Optional[int] = None) -> float:
        n = nobs or self.posterior.shape[0]
        return -2 * self.loglik + np.log(n) * self.npar

    def __repr__(self):
        return (
            f"LCAResult(loglik={self.loglik:.4f}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )


def fit(
    formula: str,
    data: pl.DataFrame,
    nclass: int = 2,
    maxiter: int = 1000,
    tol: float = 1e-10,
    verbose: bool = False,
    na_rm: bool = True,
    probs_start: Optional[np.ndarray] = None,
    beta_start: Optional[np.ndarray] = None,
) -> LCAResult:
    """Fit a latent class model.

    Parameters
    ----------
    formula : str
        Patsy-style formula, e.g. "cbind(Y1, Y2, Y3) ~ 1" or "Y1 + Y2 ~ X1 + X2".
        Left-hand side gives manifest variables; right-hand side gives covariates.
    data : pd.DataFrame
        Data frame containing all variables.
    nclass : int
        Number of latent classes.
    maxiter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence tolerance.
    verbose : bool
        Print iteration progress.
    na_rm : bool
        Drop rows with any missing values.
    probs_start : np.ndarray, optional
        Starting values for class-conditional response probabilities.
    beta_start : np.ndarray, optional
        Starting values for covariate coefficients.

    Returns
    -------
    LCAResult
        Fitted model result object.
    """
    if na_rm:
        data = data.drop_nulls()

    # Parse formula to get y and x matrices
    y_mat, x_mat, num_choices = build_design_matrix(formula, data)

    # Ensure y is integer and 1-based
    y_int = y_mat.astype(np.int32)
    if y_int.min() < 0:
        raise ValueError("Response variables must be non-negative integers (0 = missing).")

    cpp_data = Data()
    cpp_data.y = y_int
    cpp_data.x = x_mat.astype(np.float64)
    cpp_data.num_choices = num_choices

    pstart = probs_start if probs_start is not None else np.array([], dtype=np.float64)
    bstart = beta_start if beta_start is not None else np.array([], dtype=np.float64)

    raw = fit_em(
        cpp_data,
        nclass=nclass,
        maxiter=maxiter,
        tol=tol,
        probs_start=pstart,
        beta_start=bstart,
    )

    return LCAResult(raw, formula=formula, data=data)
