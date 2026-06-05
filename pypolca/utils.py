"""Utility functions for formula parsing and data preparation."""

import numpy as np
import polars as pl


def build_design_matrix(
    formula: str, data: pl.DataFrame, na_rm: bool = True
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Parse a simple formula and build design matrices.

    Supports:
        "Y1 + Y2 + Y3 ~ 1"          -> intercept only (no covariates)
        "Y1 + Y2 ~ X1 + X2"         -> covariates
        "cbind(Y1, Y2, Y3) ~ 1"     -> R-style cbind on LHS

    Returns
    -------
    y : np.ndarray, shape (N, J)
    x : np.ndarray, shape (N, S)
    num_choices : list of int
        Number of categories for each manifest variable.
    """
    lhs, rhs = formula.split("~")
    lhs = lhs.strip()
    rhs = rhs.strip()

    # --- Parse LHS (manifest variables) ---
    if lhs.startswith("cbind(") and lhs.endswith(")"):
        inner = lhs[6:-1]
        y_names = [v.strip() for v in inner.split(",")]
    else:
        y_names = [v.strip() for v in lhs.split("+")]

    if na_rm:
        y = data[y_names].to_numpy().astype(np.int32)
    else:
        # Nulls in manifest variables → 0 (missing indicator for EM engine)
        y = data[y_names].fill_null(0).to_numpy().astype(np.int32)

    # Determine number of choices per item
    num_choices = []
    for j in range(y.shape[1]):
        valid = y[:, j][y[:, j] > 0]
        if len(valid) == 0:
            raise ValueError(f"Variable {y_names[j]} has no valid responses.")
        num_choices.append(int(valid.max()))

    # --- Parse RHS (covariates) ---
    if rhs == "1":
        # Intercept only: x is a column of ones, S = 1
        x = np.ones((y.shape[0], 1), dtype=np.float64)
    else:
        x_names = [v.strip() for v in rhs.split("+")]
        x = data[x_names].to_numpy().astype(np.float64)
        # Prepend intercept
        intercept = np.ones((x.shape[0], 1), dtype=np.float64)
        x = np.hstack([intercept, x])

    return y, x, num_choices
