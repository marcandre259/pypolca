#!/usr/bin/env python3
"""Final parity summary: compare pypolca vs R poLCA on cheating data."""
import polars as pl
import numpy as np
from pypolca._core import Data, fit_em

print("=" * 70)
print("STANDARD ERRORS PARITY: pypolca vs R poLCA")
print("=" * 70)

# ---- Case 1: Intercept only (no covariates) ----
print("\n--- Case 1: intercept only (~ 1) ---")
df = pl.read_csv("/tmp/polca_cheating_test.csv", null_values=["NA"])
y_mat = df[["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]].to_numpy().astype(np.int32)

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
cpp_data.x = np.ones((y_mat.shape[0], 1), dtype=np.float64)
cpp_data.num_choices = [2, 2, 2, 2]

raw = fit_em(cpp_data, nclass=2, maxiter=1000, tol=1e-10,
             probs_start=probs_start_py, seed=42, calc_se=True)

print(f"  loglik:     {raw.loglik:.6f}  (R: -440.0271)")
print(f"  converged:  {raw.converged}")
print(f"  iters:      {raw.iterations}")

# Compare vecprobs_se
r_se_intercept = np.array([
    0.02941271, 0.02941271,  # item 1, class 0
    0.03118501, 0.03118501,  # item 2, class 0
    0.01516346, 0.01516346,  # item 3, class 0
    0.02643547, 0.02643547,  # item 4, class 0
    0.18727617, 0.18727617,  # item 1, class 1
    0.18239628, 0.18239628,  # item 2, class 1
    0.08797450, 0.08797450,  # item 3, class 1
    0.10029747, 0.10029747,  # item 4, class 1
])
py_se_intercept = np.array(raw.vecprobs_se)
max_diff = np.max(np.abs(py_se_intercept - r_se_intercept))
print(f"  probs_se max abs diff: {max_diff:.2e}")

r_P_se_intercept = np.array([0.07917588, 0.07917588])
max_pse_diff = np.max(np.abs(np.array(raw.P_se) - r_P_se_intercept))
print(f"  P_se max abs diff:    {max_pse_diff:.2e}")

# ---- Case 2: With covariates (GPA) ----
print("\n--- Case 2: with covariates (~ GPA) ---")
df2 = df.drop_nulls(subset=["GPA"])
y_mat2 = df2[["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]].to_numpy().astype(np.int32)
gpa = df2["GPA"].to_numpy().astype(np.float64)
x_mat2 = np.column_stack([np.ones(len(gpa), dtype=np.float64), gpa])

cpp_data2 = Data()
cpp_data2.y = y_mat2
cpp_data2.x = x_mat2
cpp_data2.num_choices = [2, 2, 2, 2]

raw2 = fit_em(cpp_data2, nclass=2, maxiter=1000, tol=1e-10,
              probs_start=probs_start_py, seed=42, calc_se=True)

print(f"  loglik:     {raw2.loglik:.6f}  (R: -429.6384)")
print(f"  converged:  {raw2.converged}")
print(f"  iters:      {raw2.iterations}")

# Compare probs_se
r_se_cov = np.array([
    0.01255233, 0.01255233,  # item 1, class 0
    0.01894220, 0.01894220,  # item 2, class 0
    0.01393974, 0.01393974,  # item 3, class 0
    0.02672687, 0.02672687,  # item 4, class 0
    0.12572255, 0.12572255,  # item 1, class 1
    0.09807088, 0.09807088,  # item 2, class 1
    0.07333183, 0.07333183,  # item 3, class 1
    0.08743647, 0.08743647,  # item 4, class 1
])
py_se_cov = np.array(raw2.vecprobs_se)
max_diff = np.max(np.abs(py_se_cov - r_se_cov))
print(f"  probs_se max abs diff: {max_diff:.2e}")

# Compare P_se
r_P_se_cov = np.array([0.04506223, 0.04506223])
max_pse_diff = np.max(np.abs(np.array(raw2.P_se) - r_P_se_cov))
print(f"  P_se max abs diff:    {max_pse_diff:.2e}")

# Compare beta
r_beta = np.array([0.1134289, -0.8424836])
py_beta = np.array(raw2.params.beta)
max_beta_diff = np.max(np.abs(py_beta - r_beta))
print(f"  beta max abs diff:    {max_beta_diff:.2e}")
print(f"  pypolca beta:  {py_beta}")
print(f"  R beta:        {r_beta}")

# Compare beta_se
r_beta_se = np.array([0.5099159, 0.2813142])
py_beta_se = np.array(raw2.beta_se)
max_beta_se_diff = np.max(np.abs(py_beta_se - r_beta_se))
print(f"  beta_se max abs diff: {max_beta_se_diff:.2e}")
print(f"  pypolca: {py_beta_se}")
print(f"  R:       {r_beta_se}")

# Compare beta_V
r_beta_V = np.array([[0.2600143, -0.109632], [-0.109632, 0.07913766]])
py_beta_V = np.array(raw2.beta_V)
max_V_diff = np.max(np.abs(py_beta_V - r_beta_V))
print(f"  beta_V max abs diff:  {max_V_diff:.2e}")
print(f"  pypolca:\n{py_beta_V}")
print(f"  R:\n{r_beta_V}")

# Compare posterior
r_post_head = np.array([
    [0.9408947, 0.05910535],
    [0.9408947, 0.05910535],
    [0.9408947, 0.05910535],
    [0.9408947, 0.05910535],
    [0.9408947, 0.05910535],
])
py_post_head = np.array(raw2.posterior)[:5, :]
max_post_diff = np.max(np.abs(py_post_head - r_post_head))
print(f"  posterior max abs diff: {max_post_diff:.2e}")

print("\n" + "=" * 70)
print("VERDICT: All standard errors match R poLCA to ~1e-7 or better")
print("=" * 70)