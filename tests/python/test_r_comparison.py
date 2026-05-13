"""Compare pypolca EM engine against reference R poLCA implementation.

These tests dynamically invoke R (if available) and assert that pypolca
produces numerically identical results on the same data with the same
deterministic starting values.
"""

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import pytest

from pypolca import fit


def _r_available() -> bool:
    try:
        subprocess.run(["Rscript", "-e", "library(poLCA)"],
                       capture_output=True, check=True, timeout=30)
        return True
    except Exception:
        return False


R_AVAILABLE = _r_available()


def _build_r_script(
    data_json: str,
    formula_items: List[str],
    nclass: int,
    probs_start: Optional[List[List[List[float]]]],
    maxiter: int,
    tol: float,
) -> str:
    """Construct an R script that fits poLCA and prints JSON results."""
    probs_start_r = "NULL"
    if probs_start is not None:
        mats = []
        for mat in probs_start:
            rows = ", ".join(
                "c(" + ", ".join(str(v) for v in row) + ")"
                for row in mat
            )
            mats.append(f"matrix(c({rows}), nrow={len(mat)}, byrow=TRUE)")
        probs_start_r = "list(" + ", ".join(mats) + ")"

    script = f'''
library(poLCA)
library(jsonlite)

data <- fromJSON('{data_json}')
formula <- as.formula(paste("cbind(", paste(c({",".join(repr(x) for x in formula_items)}), collapse=","), ") ~ 1"))

probs_start <- {probs_start_r}

res <- poLCA(formula, data=data, nclass={nclass}, maxiter={maxiter}, tol={tol}, verbose=FALSE, probs.start=probs_start)

# Extract results
out <- list(
  loglik = res$llik,
  aic = res$aic,
  bic = res$bic,
  P = as.numeric(res$P),
  predclass = as.integer(res$predclass),
  posterior = head(as.matrix(res$posterior), 20),
  probs = lapply(res$probs, as.matrix)
)

cat(toJSON(out, digits=12, matrix="rowmajor"))
'''
    return script


def run_r_polca(
    df: pl.DataFrame,
    formula_items: List[str],
    nclass: int,
    probs_start: Optional[List[List[List[float]]]] = None,
    maxiter: int = 1000,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Run R poLCA and return parsed results."""
    # Convert polars DataFrame to JSON (list of dicts)
    records = df.select(formula_items).to_dicts()
    data_json = json.dumps(records)

    script = _build_r_script(data_json, formula_items, nclass, probs_start, maxiter, tol)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        r_file = f.name

    try:
        result = subprocess.run(
            ["Rscript", r_file],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        raw = json.loads(result.stdout)
        # jsonlite serialises length-1 R vectors as JSON arrays; unwrap scalars.
        for key in ("loglik", "aic", "bic"):
            if key in raw and isinstance(raw[key], list) and len(raw[key]) == 1:
                raw[key] = raw[key][0]
        return raw
    finally:
        os.unlink(r_file)


def _flatten_probs_start(
    probs_start_list: List[List[List[float]]]
) -> np.ndarray:
    """Convert R-style list-of-matrices to pypolca flat vector.

    R format: list of J matrices, each nclass x K_j
    pypolca format: class-major, item-major, category-minor flat vector.
    """
    nclass = len(probs_start_list[0])
    flat = []
    for r in range(nclass):
        for item_mat in probs_start_list:
            flat.extend(item_mat[r])
    return np.array(flat, dtype=np.float64)


def _align_classes(py_probs: np.ndarray, r_probs, nclass: int, num_choices: List[int]) -> np.ndarray:
    """Return a permutation array that reorders pypolca classes to match R classes.

    r_probs may be a list of matrices or a dict (from JSON) of matrices.
    """
    # Normalise r_probs to a list of matrices in item order
    if isinstance(r_probs, dict):
        r_probs = list(r_probs.values())

    # Build pypolca probs into J matrices of shape (nclass, K_j)
    py_mats = []
    pos = 0
    total_choices = sum(num_choices)
    for K in num_choices:
        mat = np.zeros((nclass, K))
        for r in range(nclass):
            mat[r, :] = py_probs[r * total_choices + pos : r * total_choices + pos + K]
        py_mats.append(mat)
        pos += K

    # Compute distance between each pypolca class and each R class
    dist = np.zeros((nclass, nclass))
    for r_py in range(nclass):
        for r_r in range(nclass):
            d = 0.0
            for py_mat, r_mat in zip(py_mats, r_probs):
                r_vec = np.array(r_mat[r_r], dtype=float)
                d += np.sum((py_mat[r_py] - r_vec) ** 2)
            dist[r_py, r_r] = d

    # Greedy matching (simple, works for small nclass)
    perm = np.zeros(nclass, dtype=int)
    used = set()
    for r_r in range(nclass):
        best = None
        best_d = float('inf')
        for r_py in range(nclass):
            if r_py not in used and dist[r_py, r_r] < best_d:
                best_d = dist[r_py, r_r]
                best = r_py
        perm[r_r] = best
        used.add(best)
    return perm


@pytest.mark.skipif(not R_AVAILABLE, reason="R and poLCA not available")
def test_synthetic_2class():
    """Small synthetic dataset: 2 classes, 3 binary items, 10 observations."""
    df = pl.DataFrame({
        "Y1": [1, 1, 1, 2, 2, 2, 1, 2, 1, 2],
        "Y2": [1, 1, 2, 1, 2, 2, 2, 1, 1, 2],
        "Y3": [1, 2, 1, 1, 2, 2, 2, 2, 1, 1],
    })

    # Deterministic starting values (R format: list of J matrices, each nclass x K_j)
    probs_start_r = [
        [[0.9, 0.1], [0.3, 0.7]],  # item 0
        [[0.8, 0.2], [0.2, 0.8]],  # item 1
        [[0.7, 0.3], [0.3, 0.7]],  # item 2
    ]

    # Run R reference
    r_res = run_r_polca(df, ["Y1", "Y2", "Y3"], nclass=2,
                        probs_start=probs_start_r, maxiter=1000, tol=1e-10)

    # Run pypolca
    probs_start_py = _flatten_probs_start(probs_start_r)
    py_res = fit("cbind(Y1, Y2, Y3) ~ 1", df, nclass=2,
                 maxiter=1000, tol=1e-10, probs_start=probs_start_py)

    # --- Compare log-likelihood ---
    assert py_res.loglik == pytest.approx(r_res["loglik"], abs=1e-5)

    # --- Align classes and compare P ---
    perm = _align_classes(py_res.params.vecprobs, r_res["probs"], 2, [2, 2, 2])
    py_P = py_res.P[perm]
    r_P = np.array(r_res["P"])
    np.testing.assert_allclose(py_P, r_P, atol=1e-4)

    # --- Compare predicted classes ---
    py_pred = py_res.predclass - 1  # 0-based
    py_pred_aligned = perm[py_pred] + 1  # back to 1-based, aligned to R
    r_pred = np.array(r_res["predclass"])
    np.testing.assert_array_equal(py_pred_aligned, r_pred)

    # --- Compare posterior probabilities ---
    r_post = np.array(r_res["posterior"])
    py_post = py_res.posterior[: r_post.shape[0]][:, perm]
    np.testing.assert_allclose(py_post, r_post, atol=1e-5)


@pytest.mark.skipif(not R_AVAILABLE, reason="R and poLCA not available")
def test_cheating_dataset():
    """Real-world cheating dataset shipped with R poLCA."""
    # Export cheating data from R to a temp CSV
    r_export = '''
library(poLCA)
data(cheating)
write.csv(cheating, "/tmp/polca_cheating_test.csv", row.names=FALSE)
'''
    subprocess.run(["Rscript", "-e", r_export], check=True, capture_output=True)
    df = pl.read_csv("/tmp/polca_cheating_test.csv")

    # Clean up any NA string values that polars reads as strings
    for col in ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Int64))

    probs_start_r = [
        [[0.9, 0.1], [0.4, 0.6]],  # item 0
        [[0.9, 0.1], [0.4, 0.6]],  # item 1
        [[0.9, 0.1], [0.4, 0.6]],  # item 2
        [[0.9, 0.1], [0.4, 0.6]],  # item 3
    ]

    r_res = run_r_polca(df, ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"],
                        nclass=2, probs_start=probs_start_r, maxiter=1000, tol=1e-10)

    probs_start_py = _flatten_probs_start(probs_start_r)
    py_res = fit("cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1", df,
                 nclass=2, maxiter=1000, tol=1e-10, probs_start=probs_start_py)

    assert py_res.loglik == pytest.approx(r_res["loglik"], abs=1e-4)

    perm = _align_classes(py_res.params.vecprobs, r_res["probs"], 2, [2, 2, 2, 2])
    py_P = py_res.P[perm]
    r_P = np.array(r_res["P"])
    np.testing.assert_allclose(py_P, r_P, atol=1e-4)

    py_pred = py_res.predclass - 1
    py_pred_aligned = perm[py_pred] + 1
    r_pred = np.array(r_res["predclass"])
    np.testing.assert_array_equal(py_pred_aligned, r_pred)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
