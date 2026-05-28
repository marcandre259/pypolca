#!/usr/bin/env python3
"""Deep parity: compare VCE matrix, score matrix between pypolca and R."""
import polars as pl
import numpy as np
from pypolca._core import Data, fit_em

# Load cheating data
df = pl.read_csv("/tmp/polca_cheating_test.csv")

y_names = ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]
y_mat = df[y_names].to_numpy().astype(np.int32)
num_choices = [int(y_mat[:, j][y_mat[:, j] > 0].max()) for j in range(len(y_names))]

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
cpp_data.x = np.ones((y_mat.shape[0], 1), dtype=np.float64)
cpp_data.num_choices = num_choices

raw = fit_em(cpp_data, nclass=2, maxiter=1000, tol=1e-10,
             probs_start=probs_start_py, seed=42, calc_se=True)

print("=== loglik ===")
print(raw.loglik)

print("\n=== posterior shape ===")
print(raw.posterior.shape)

print("\n=== prior shape ===")
print(raw.prior.shape)

print("\n=== vecprobs_se ===")
print(raw.vecprobs_se)

print("\n=== P_se ===")
print(raw.P_se)

print("\n=== beta_V (raw VCE for betas) ===")
print(raw.beta_V)
print(f"shape: {raw.beta_V.shape}")

print("\n=== beta_se (sqrt diag) ===")
print(raw.beta_se)

# Manually compute VCE from scores for verification
# We can't access scores directly, but we can check intermediate values
print("\n=== params.beta ===")
print(raw.params.beta)

print("\n=== params.vecprobs ===")
print(raw.params.vecprobs)

# Now compute scores manually in Python for verification
from pypolca._core import e_step, compute_prior_from_beta

N = cpp_data.n_obs()
J = cpp_data.n_items()
R = 2
S = 1

posterior = raw.posterior
prior = raw.prior
params = raw.params
vecprobs = params.vecprobs

# Dp = R * sum(K_j - 1) = 2 * 4 = 8
# Dbeta = S * (R - 1) = 1 * 1 = 1
# Total D = 9

n_choices = sum(num_choices)
Dp = sum(R * (K - 1) for K in num_choices)
Dbeta = S * (R - 1)
D = Dp + Dbeta

scores = np.zeros((N, D))

# Prob scores
col = 0
for r in range(R):
    for j in range(J):
        K_j = num_choices[j]
        for k in range(1, K_j):  # free categories (skip ref=0)
            # prob_index: k, j, r -> flat index
            idx = r * n_choices + sum(num_choices[:j]) + k
            prob = vecprobs[idx]
            for i in range(N):
                if cpp_data.y[i, j] <= 0:
                    continue
                ind = 1.0 if cpp_data.y[i, j] == k + 1 else 0.0
                scores[i, col] = posterior[i, r] * (ind - prob)
            col += 1

# Beta score
for r in range(1, R):
    for l in range(S):
        for i in range(N):
            scores[i, col] = cpp_data.x[i, l] * (posterior[i, r] - prior[i, r])
        col += 1

info_py = scores.T @ scores

print("\n=== Python-manual VCE (info inverse) ===")
try:
    VCE_py = np.linalg.pinv(info_py)
    print(VCE_py)
    print("\nVCE.beta (bottom-right 1x1):")
    print(VCE_py[-1:, -1:])
    print("\nse.beta (manual):", np.sqrt(np.diag(VCE_py[-1:, -1:])))

    print("\nVCE[0:Dp, 0:Dp] (probs):")
    print(VCE_py[:Dp, :Dp])

    print("\nvecprobs_se manual (sqrt diag after delta):")
    # Build Jacobian and compute probs SE manually
    from pypolca._core import e_step, compute_log_ylik

    # Jacobian: softmax, drop first col
    total_probs = R * n_choices
    J_probs = np.zeros((total_probs, Dp))
    rpos = 0
    cpos = 0
    for r in range(R):
        for j in range(J):
            K_j = num_choices[j]
            p = np.zeros(K_j)
            for k in range(K_j):
                idx = r * n_choices + sum(num_choices[:j]) + k
                p[k] = vecprobs[idx]
            Jsub = np.diag(p) - np.outer(p, p)
            J_probs[rpos:rpos+K_j, cpos:cpos+K_j-1] = Jsub[:, 1:]
            rpos += K_j
            cpos += K_j - 1

    VCE_lo = VCE_py[:Dp, :Dp]
    VCE_probs = J_probs @ VCE_lo @ J_probs.T
    se_probs = np.sqrt(np.maximum(np.diag(VCE_probs), 0.0))
    print("Manual vecprobs_se:")
    print(se_probs)
    print("\nC++ vecprobs_se:")
    print(raw.vecprobs_se)

except Exception as e:
    print(f"Error: {e}")

# Compare with R: should get VCE.beta = 0.0482, se.beta = 0.2195
# But we're getting VCE.beta = 0.345, se.beta = 0.587