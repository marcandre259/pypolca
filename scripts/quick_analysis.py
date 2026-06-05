#!/usr/bin/env python3
"""Quick analysis: poLCA with GPA covariate on cheating data."""

import polars as pl

from pypolca.api import fit

df = pl.read_csv("pypolca/data/cheating.csv", null_values=["NA"])
df = df.drop_nulls()

result = fit(
    "cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA", df, nclass=2, seed=42, nrep=1, verbose=True
)

print("\n=== coeff ===")
print(result.coeff)
print("\n=== coeff_se ===")
print(result.coeff_se)
print("\n=== coeff_V ===")
print(result.coeff_V)
print("\n=== loglik ===")
print(result.loglik)
print("\n=== P ===")
print(result.P)
print("\n=== P_se ===")
print(result.P_se)
