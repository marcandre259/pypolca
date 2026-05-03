"""Example: fit a basic latent class model with pypoLCA."""

import numpy as np
import pandas as pd

# Import the low-level C++ bindings directly
from polca._core import Data, fit_em

# --- Create synthetic data ---
np.random.seed(42)
N = 500

# 4 manifest variables, each with 3 categories
J = 4
K = 3

# True latent class labels
true_class = np.random.choice([0, 1], size=N, p=[0.4, 0.6])

# Class-conditional response probabilities
# Class 0: tends to answer 1
# Class 1: tends to answer 3
probs_c0 = np.array([0.6, 0.3, 0.1])
probs_c1 = np.array([0.1, 0.3, 0.6])

y = np.zeros((N, J), dtype=np.int32)
for i in range(N):
    for j in range(J):
        if true_class[i] == 0:
            y[i, j] = np.random.choice([1, 2, 3], p=probs_c0) + 1  # 1-based
        else:
            y[i, j] = np.random.choice([1, 2, 3], p=probs_c1) + 1

# Build Data object
data = Data()
data.y = y
data.x = np.ones((N, 1), dtype=np.float64)  # intercept only (no covariates)
data.num_choices = [K] * J

# --- Fit model ---
print("Fitting 2-class LCA model...")
result = fit_em(data, nclass=2, maxiter=200, tol=1e-8, verbose=True)

print(f"\nConverged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Log-likelihood: {result.loglik:.4f}")
print(f"Class shares: {result.posterior.mean(axis=0)}")
print(f"Predclass counts: {np.bincount(result.posterior.argmax(axis=1))}")

# --- Try the high-level API ---
try:
    from polca.api import fit

    df = pd.DataFrame(
        y,
        columns=[f"Y{j+1}" for j in range(J)],
    )
    result2 = fit("Y1 + Y2 + Y3 + Y4 ~ 1", df, nclass=2, verbose=False)
    print(f"\nHigh-level API result: {result2}")
    print(f"Class shares: {result2.P}")
except Exception as e:
    print(f"High-level API not yet functional: {e}")
