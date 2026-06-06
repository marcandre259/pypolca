"""High-level Python API for pypoLCA."""

from typing import Any, cast

import numpy as np
import polars as pl

from pypolca._core import Data, compute_prior_from_beta, e_step, fit_em
from pypolca.utils import build_design_matrix


class LCAResult:
    """Python-friendly wrapper around the C++ Results struct."""

    def __init__(
        self,
        raw_results: Any,
        formula: str | None = None,
        data: pl.DataFrame | None = None,
        num_choices: list[int] | None = None,
        y_mat: np.ndarray | None = None,
    ) -> None:
        self._raw = raw_results
        self.formula = formula
        self.data = data
        self._num_choices = num_choices
        self._y_mat = y_mat

    @property
    def probs(self) -> list[np.ndarray]:
        """Class-conditional response probabilities.

        Returns: list[J] of ndarray of shape (R, K_j).
        """
        vecprobs = self.params.vecprobs
        if not isinstance(self._num_choices, list):
            return []

        vecprobs_arr = np.array(vecprobs)

        total_choices = sum(self._num_choices)
        R = self.posterior.shape[1]
        vecprobs_arr = vecprobs_arr.reshape((R, total_choices))

        probs_arr = []

        pos = 0
        for k in self._num_choices:
            probs_arr.append(vecprobs_arr[:, pos : pos + k])
            pos += k

        return probs_arr

    @property
    def coeff(self) -> np.ndarray:
        """Covariate coefficients.

        Returns: ndarray of shape (S, R-1). Column r-1 = coefficients for class r (r >= 2).
        None if no covariates.
        """
        R = self._raw.posterior.shape[1]
        beta_coefs = self.params.beta
        S = len(beta_coefs)

        if S == 0 or R <= 1:
            return np.array([])

        n_covariates = S // (R - 1)

        return np.array(beta_coefs).reshape((n_covariates, R - 1), order="F")

    # Prob params and beta params
    @property
    def npar(self) -> int:
        num_choices = self._num_choices
        if num_choices is None:
            return 0

        n_classes = self.posterior.shape[1]
        n_probs = sum(n_classes * (k - 1) for k in num_choices)

        beta = np.array(self.params.beta)
        n_beta = len(beta)
        S = n_beta // (n_classes - 1) if n_classes > 1 and n_beta > 0 else 0

        if S <= 1:
            return int(n_probs)

        n_betas = S * (n_classes - 1)

        return int(n_probs + n_betas)

    @property
    def probs_se(self) -> list[np.ndarray]:
        if not isinstance(self._num_choices, list):
            return []

        vecprobs_se = np.array(self._raw.vecprobs_se)
        if vecprobs_se.size == 0:
            n_classes = self.posterior.shape[1]
            return [np.zeros((n_classes, k)) for k in self._num_choices]

        se_list = []

        n_classes = self.posterior.shape[1]

        pos = 0
        for k in self._num_choices:
            block_len = n_classes * k
            block = vecprobs_se[pos : pos + block_len]
            se_list.append(block.reshape((n_classes, k)))
            pos += block_len

        return se_list

    @property
    def P_se(self) -> np.ndarray:
        return np.array(self._raw.P_se)

    @property
    def coeff_se(self) -> np.ndarray:
        """Standard errors of covariate coefficients.

        Returns: ndarray of shape (S, R-1) matching .coeff layout.
        Empty array if no covariates.
        """
        if len(self._raw.coeff_se) == 0:
            return np.array([])

        n_classes = self.posterior.shape[1]
        beta = np.array(self.params.beta)
        S = len(beta) // (n_classes - 1) if n_classes > 1 else 0

        return np.array(self._raw.coeff_se).reshape((S, n_classes - 1))

    @property
    def coeff_V(self) -> np.ndarray:
        return np.array(self._raw.beta_V)

    @property
    def loglik(self) -> float:
        return float(self._raw.loglik)

    @property
    def iterations(self) -> int:
        return int(self._raw.iterations)

    @property
    def converged(self) -> bool:
        return bool(self._raw.converged)

    @property
    def posterior(self) -> np.ndarray:
        """Training-data posterior class membership probabilities."""
        return np.array(self._raw.posterior)

    @property
    def prior(self) -> np.ndarray:
        return np.array(self._raw.prior)

    @property
    def predclass(self) -> np.ndarray:
        return cast(np.ndarray, self.posterior.argmax(axis=1) + 1)  # 1-based class labels

    def predict_posterior(
        self, newdata: pl.DataFrame, newx: pl.DataFrame | None = None
    ) -> np.ndarray:
        """Compute posterior class membership probabilities for new data."""
        if self.formula is None:
            raise ValueError(
                "Model was not fitted with a formula; cannot extract manifest variable names."
            )

        if self._num_choices is None:
            raise ValueError("Model has no manifest variables.")

        num_choices = self._num_choices

        lhs = self.formula.split("~")[0].strip()
        if lhs.startswith("cbind(") and lhs.endswith(")"):
            inner = lhs[6:-1]
            y_names = [v.strip() for v in inner.split(",")]
        else:
            y_names = [v.strip() for v in lhs.split("+")]

        y = newdata.select(y_names).fill_null(0).to_numpy().astype(np.int32)

        for j in range(y.shape[1]):
            valid = y[:, j][y[:, j] > 0]
            if len(valid) > 0 and int(valid.max()) > num_choices[j]:
                raise ValueError(
                    f"Variable '{y_names[j]}': category {int(valid.max())} "
                    f"exceeds model's {num_choices[j]} categories"
                )

        N_new = y.shape[0]
        R = self.posterior.shape[1]

        rhs = self.formula.split("~")[1].strip()
        has_covariates = rhs != "1"

        if has_covariates and newx is not None:
            x_names = [v.strip() for v in rhs.split("+")]
            x = newx.select(x_names).to_numpy().astype(np.float64)
            x_with_intercept = np.column_stack([np.ones(x.shape[0], dtype=np.float64), x])
            beta = np.array(self.params.beta, dtype=np.float64)
            prior = compute_prior_from_beta(x_with_intercept, beta, R)
        elif has_covariates and newx is None:
            raise ValueError("Model was fitted with covariates; newx must be provided.")
        else:
            prior = np.tile(self.P.reshape(1, -1), (N_new, 1))

        data_new = Data()
        data_new.y = y
        data_new.x = (
            x_with_intercept.copy()
            if (has_covariates and newx is not None)
            else np.ones((N_new, 1), dtype=np.float64)
        )
        data_new.num_choices = list(num_choices)

        posterior_mat, _ = e_step(data_new, self.params, prior, R)
        return np.array(posterior_mat)

    @property
    def params(self) -> Any:
        """Raw fitted parameters (vecprobs, beta)."""
        return self._raw.params

    @property
    def P(self) -> np.ndarray:
        """Class population shares (marginal prior probabilities)."""
        return cast(np.ndarray, self.posterior.mean(axis=0))

    @property
    def aic(self) -> float:
        return float(-2 * self.loglik + 2 * self.npar)

    @property
    def bic(self, nobs: int | None = None) -> float:
        n = nobs or self.posterior.shape[0]
        return float(-2 * self.loglik + np.log(n) * self.npar)

    @property
    def Gsq(self) -> float:
        """Likelihood ratio deviance (G-squared) vs saturated model."""
        obs, exp, _ = self._compute_expected_frequencies()
        if len(obs) == 0:
            return 0.0
        mask = obs > 0
        return float(2.0 * np.sum(obs[mask] * np.log(obs[mask] / exp[mask])))

    @property
    def Chisq(self) -> float:
        """Pearson chi-square goodness-of-fit.

        Includes correction term (N - sum(exp)) for unobserved response
        patterns where O=0 and E>0, matching R poLCA behavior.
        """
        obs, exp, _ = self._compute_expected_frequencies()
        if len(obs) == 0:
            return 0.0
        N = self._y_mat.shape[0] if self._y_mat is not None else obs.sum()
        return float(np.sum((obs - exp) ** 2 / exp) + (N - np.sum(exp)))

    @property
    def predcell(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (observed, expected, patterns) for each unique complete response pattern."""
        return self._compute_expected_frequencies()

    @property
    def resid_df(self) -> int:
        """Residual degrees of freedom for GOF tests (intercept-only models)."""
        _, exp, _ = self._compute_expected_frequencies()
        num_cells = int(np.sum(exp > 0))
        return int(num_cells - self.npar - 1)

    @property
    def Nobs(self) -> int:
        """Number of fully observed cases (no missing in any manifest variable)."""
        if self._y_mat is None:
            return int(self.posterior.shape[0])
        return int(np.all(self._y_mat > 0, axis=1).sum())

    def __repr__(self) -> str:
        return (
            f"LCAResult(loglik={self.loglik:.4f}, "
            f"iterations={self.iterations}, converged={self.converged})"
        )

    def _compute_expected_frequencies(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute observed and expected counts for each unique complete response pattern.

        Only complete response patterns (no missing values, i.e., y > 0 on all items)
        are included. For intercept-only models, expected count for pattern a is:

            N * Σ_r P_r * Π_j ρ_{jr}(a_j - 1)
        """
        if self._y_mat is None:
            return (np.array([]), np.array([]), np.array([]))

        N = self._y_mat.shape[0]
        J = self._y_mat.shape[1]
        R = self.posterior.shape[1]
        P = self.P

        # Count observed frequencies of each complete pattern (skip rows with missing)
        patterns: dict[tuple[int, ...], int] = {}
        for i in range(N):
            if np.any(self._y_mat[i] == 0):
                continue
            key = tuple(self._y_mat[i].tolist())
            patterns[key] = patterns.get(key, 0) + 1

        if not patterns:
            return (np.array([]), np.array([]), np.array([]))

        obs = np.array(list(patterns.values()), dtype=np.float64)
        pats = np.array(list(patterns.keys()), dtype=np.int32)  # (n_patterns, J), 1-based

        # Expected: N * Σ_r P_r * Π_j ρ_{jr}(a_j - 1)
        exp = np.zeros(len(obs), dtype=np.float64)
        for r in range(R):
            contrib = np.ones(len(obs), dtype=np.float64)
            for j in range(J):
                rho = self.probs[j][r]  # (K_j,) vector for item j, class r
                contrib *= rho[pats[:, j] - 1]  # 1-based pattern → 0-based index
            exp += P[r] * contrib
        exp *= N

        return (obs, exp, pats)


def fit(
    formula: str,
    data: pl.DataFrame,
    nclass: int = 2,
    maxiter: int = 1000,
    tol: float = 1e-10,
    verbose: bool = False,
    na_rm: bool = True,
    probs_start: np.ndarray | None = None,
    beta_start: np.ndarray | None = None,
    nrep: int = 1,
    seed: int | None = None,
    max_restarts: int = 100,
    calc_se: bool = True,
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
    nrep : int
        Number of replications with different random starting values (like R's `nrep`).
    seed : int, optional
        Random seed for the first replication. If None, a random seed is drawn.
    max_restarts : int
        Maximum restarts per replication when a likelihood drop occurs (R retries
        indefinitely; this is a safety cap).
    calc_se : bool
        Whether to compute standard errors (default True).

    Returns
    -------
    LCAResult
        Fitted model result object.
    """
    # Drop rows where covariates are missing (regardless of na_rm)
    # Manifest missingness is handled inside build_design_matrix
    rhs = formula.split("~")[1].strip()
    if rhs != "1":
        x_names = [v.strip() for v in rhs.split("+")]
        for col in x_names:
            data = data.filter(pl.col(col).is_not_null())
        if data.is_empty():
            raise ValueError("No observations remain after dropping rows with missing covariates.")

    if na_rm:
        data = data.drop_nulls()

    # Parse formula to get y and x matrices
    y_mat, x_mat, num_choices = build_design_matrix(formula, data, na_rm=na_rm)

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

    rng = np.random.default_rng(seed)

    best_result: LCAResult | None = None
    best_loglik = -np.inf

    for rep in range(nrep):
        base_seed = int(rng.integers(0, 2**32))

        for attempt in range(max_restarts):
            current_seed = base_seed + attempt

            # On retry, discard user-provided starting probs and generate
            # fresh random starts (matching R poLCA's !firstrun behavior).
            probs_for_attempt = pstart if attempt == 0 else np.array([], dtype=np.float64)

            raw = fit_em(
                cpp_data,
                nclass=nclass,
                maxiter=maxiter,
                tol=tol,
                probs_start=probs_for_attempt,
                beta_start=bstart,
                seed=current_seed,
                calc_se=calc_se,
            )

            if not raw.error:
                candidate = LCAResult(
                    raw,
                    formula=formula,
                    data=data,
                    num_choices=num_choices,
                    y_mat=y_int,
                )
                if raw.loglik > best_loglik:
                    best_loglik = raw.loglik
                    best_result = candidate
                if verbose:
                    print(
                        f"Rep {rep + 1}/{nrep}: llik = {raw.loglik:.4f} ... best = {best_loglik:.4f}"
                    )
                break
            else:
                if verbose:
                    print(f"Rep {rep + 1}/{nrep}: attempt {attempt + 1} dropped, retrying ...")
        else:
            # All restarts for this rep failed
            if verbose:
                print(f"Rep {rep + 1}/{nrep}: all attempts failed.")

    if best_result is None:
        raise RuntimeError(
            f"EM failed to converge after {nrep} replication(s) and "
            f"up to {max_restarts} restarts each"
        )

    return best_result
