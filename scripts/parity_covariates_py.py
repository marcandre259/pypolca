#!/usr/bin/env python3
"""Parity: pypolca with covariates (GPA) on cheating data."""

import numpy as np
import polars as pl

from pypolca._core import Data, fit_em

df = pl.read_csv("pypolca/data/cheating.csv", null_values=["NA"])

# Drop rows with NA in GPA
df = df.drop_nulls(subset=["GPA"])

# Remove quotes from column names if present
df.columns = [c.strip('"') for c in df.columns]

y_names = ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]
y_mat = df[y_names].to_numpy().astype(np.int32)
num_choices = [2, 2, 2, 2]

# Build x matrix: intercept + GPA (same as R's model.matrix(formula, data))
gpa = df["GPA"].to_numpy().astype(np.float64)
N = len(gpa)
x_mat = np.column_stack([np.ones(N, dtype=np.float64), gpa])
# S = 2

# Same starting values as R
probs_start_r_list = [
    [[0.9, 0.1], [0.4, 0.6]],
    [[0.9, 0.1], [0.4, 0.6]],
    [[0.9, 0.1], [0.4, 0.6]],
    [[0.9, 0.1], [0.4, 0.6]],
]
flat = []
for r in range(2):
    for item_mat in probs_start_r_list:
        flat.extend(item_mat[r])
probs_start_py = np.array(flat, dtype=np.float64)

cpp_data = Data()
cpp_data.y = y_mat
cpp_data.x = x_mat
cpp_data.num_choices = num_choices

raw = fit_em(
    cpp_data, nclass=2, maxiter=1000, tol=1e-10, probs_start=probs_start_py, seed=42, calc_se=True
)

print("=== LOGLIK ===")
print(raw.loglik)

print("\n=== P (class shares) ===")
print(np.array(raw.posterior).mean(axis=0))

print("\n=== P_se ===")
print(raw.P_se)

print("\n=== params.beta ===")
print(raw.params.beta)

print("\n=== coeff_se ===")
print(raw.coeff_se)

print("\n=== beta_V ===")
print(raw.beta_V)

print("\n=== vecprobs_se ===")
print(raw.vecprobs_se)

print("\n=== posterior first 5 rows ===")
print(raw.posterior[:5, :])

print("\n=== prior first 5 rows ===")
print(raw.prior[:5, :])

print("\n=== converged ===")
print(raw.converged)
print(raw.iterations)

# R got:
# Loglik: -429.6384     P: 0.8218932 0.1781068     P.se: 0.04506223 0.04506223
# coeff: 0.1134289, -0.8424836
# coeff.se: 0.5099159, 0.2813142
# coeff.V: [[0.2600143, -0.109632], [-0.109632, 0.07913766]]
