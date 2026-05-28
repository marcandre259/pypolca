#!/usr/bin/env python3
"""Parity test: run pypolca on cheating data, compare standard errors with R poLCA."""
import sys
import polars as pl
import numpy as np
from pypolca._core import Data, fit_em
from pypolca.api import LCAResult

# Load cheating data
df = pl.read_csv("/tmp/polca_cheating_test.csv")

# Same deterministic starting values as R
probs_start_r_list = [
    [[0.9, 0.1], [0.4, 0.6]],  # item 1
    [[0.9, 0.1], [0.4, 0.6]],  # item 2
    [[0.9, 0.1], [0.4, 0.6]],  # item 3
    [[0.9, 0.1], [0.4, 0.6]],  # item 4
]
# Flatten: class-major, item-major, category-minor
flat = []
for r in range(2):
    for item_mat in probs_start_r_list:
        flat.extend(item_mat[r])
probs_start_py = np.array(flat, dtype=np.float64)

# Build Data
y_names = ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]
y_mat = df[y_names].to_numpy().astype(np.int32)
num_choices = [int(y_mat[:, j][y_mat[:, j] > 0].max()) for j in range(len(y_names))]

cpp_data = Data()
cpp_data.y = y_mat
cpp_data.x = np.ones((y_mat.shape[0], 1), dtype=np.float64)  # intercept only
cpp_data.num_choices = num_choices

# Fit with seed=42 (as in R)
raw = fit_em(cpp_data, nclass=2, maxiter=1000, tol=1e-10,
             probs_start=probs_start_py, seed=42, calc_se=True)

result = LCAResult(raw, num_choices=num_choices)

# --- Print pypolca results ---
print("=== LOGLIK ===")
print(result.loglik)

print("=== NPAR ===")
print(result.npar)

print("=== P (class shares) ===")
print(result.P)

print("=== PROBS (vecprobs) ===")
print(result.params.vecprobs)

print("=== PROBS SE ===")
print(raw.vecprobs_se)

print("=== P SE ===")
print(raw.P_se)

print("=== POSTERIOR (first 5 rows) ===")
print(raw.posterior[:5, :])

print("=== PRIOR (first 5 rows) ===")
print(raw.prior[:5, :])

print("=== PREDCLASS (first 20) ===")
print(result.predclass[:20])

print("=== CONVERGED ===")
print(raw.converged)

print("=== ITERATIONS ===")
print(raw.iterations)

print("=== ERROR ===")
print(raw.error)
print("=== BETA ===")
print(raw.params.beta)
print("=== BETA SE ===")
print(raw.beta_se)
print("=== BETA V ===")
print(raw.beta_V)
