"""Benchmark: pypolca (C++) vs R poLCA on election dataset with covariates.

Full model: 12 polytomous items (4-point scale), 5 covariates, nclass=3.
N=1,785 observations. Timings are means over multiple runs.
"""

import json
import os
import subprocess
import tempfile
import time
from statistics import mean, median

import numpy as np
import polars as pl

from pypolca import fit

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
MANIFEST = [
    "MORALG",
    "CARESG",
    "KNOWG",
    "LEADG",
    "DISHONG",
    "INTELG",
    "MORALB",
    "CARESB",
    "KNOWB",
    "LEADB",
    "DISHONB",
    "INTELB",
]
COVARIATES = ["VOTE3", "AGE", "EDUC", "GENDER", "PARTY"]
NCLASS = 3
NCAT = 4  # each manifest item has 4 categories
J = len(MANIFEST)  # 12 items
S = len(COVARIATES) + 1  # 6 columns in design matrix (intercept + 5 covariates)

# R formula string
R_FORMULA = "cbind(" + ",".join(MANIFEST) + ") ~ " + " + ".join(COVARIATES)
PY_FORMULA = "cbind(" + ",".join(MANIFEST) + ") ~ " + " + ".join(COVARIATES)

# Fixed starting values: for each item, 3 rows (classes) × 4 categories.
# Rows are perturbed uniform to create separation between classes.
_PROBS_START_ONE_ITEM = [
    [0.40, 0.30, 0.20, 0.10],
    [0.25, 0.25, 0.25, 0.25],
    [0.10, 0.20, 0.30, 0.40],
]


def _make_r_probs_start() -> str:
    """Generate R code for probs.start list-of-matrices."""
    rows = []
    for _ in range(J):
        vals = []
        for r in range(NCLASS):
            row = _PROBS_START_ONE_ITEM[r]
            vals.append(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}")
        rows.append(f"  matrix(c({', '.join(vals)}), nrow={NCLASS}, byrow=TRUE)")
    return "probs_start <- list(\n" + ",\n".join(rows) + "\n)"


def _make_py_probs_start() -> np.ndarray:
    """Flat probs_start vector for pypolca (R-major order)."""
    flat = []
    for r in range(NCLASS):
        for _ in range(J):
            flat.extend(_PROBS_START_ONE_ITEM[r])
    return np.array(flat, dtype=np.float64)


def _extract_factor_code(df: pl.DataFrame) -> pl.DataFrame:
    """Convert R factor label strings like '3 Not too well' → integer 3."""
    df = df.clone()
    for col in MANIFEST:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.extract(r"^(\d+)", group_index=1).cast(pl.Int64).alias(col)
            )
    return df


# ---------------------------------------------------------------------------
# R benchmark
# ---------------------------------------------------------------------------


def build_r_benchmark_script(n_runs: int) -> str:
    probs_code = _make_r_probs_start()
    return f"""
library(poLCA)
library(jsonlite)

data(election)

{probs_code}

f <- {R_FORMULA}

times <- numeric({n_runs})
logliks <- numeric({n_runs})
for (i in seq_len({n_runs})) {{
  t0 <- Sys.time()
  res <- poLCA(f, data=election, nclass={NCLASS}, maxiter=1000, tol=1e-10,
               verbose=FALSE, probs.start=probs_start, nrep=1)
  t1 <- Sys.time()
  times[i] <- as.numeric(t1 - t0, units="secs")
  logliks[i] <- res$llik
}}

out <- jsonlite::toJSON(list(
  mean_time=mean(times), median_time=median(times),
  sd_time=sd(times), loglik=mean(logliks)
), digits=12, auto_unbox=TRUE)
cat(out)
"""


def run_r_benchmark(n_runs: int = 20) -> dict:
    script = build_r_benchmark_script(n_runs)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(script)
        r_file = f.name
    try:
        result = subprocess.run(
            ["Rscript", r_file],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        raw = json.loads(result.stdout)
        for key in ("mean_time", "median_time", "sd_time", "loglik"):
            if isinstance(raw[key], list) and len(raw[key]) == 1:
                raw[key] = raw[key][0]
        return raw
    finally:
        os.unlink(r_file)


# ---------------------------------------------------------------------------
# Python benchmark
# ---------------------------------------------------------------------------


def run_python_benchmark(df: pl.DataFrame, n_runs: int = 20, calc_se: bool = True) -> dict:
    probs_start = _make_py_probs_start()
    times = []
    logliks = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = fit(
            PY_FORMULA,
            df,
            nclass=NCLASS,
            maxiter=1000,
            tol=1e-10,
            probs_start=probs_start,
            calc_se=calc_se,
            nrep=1,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
        logliks.append(res.loglik)

    return {
        "mean_time": mean(times),
        "median_time": median(times),
        "sd_time": float(np.std(times, ddof=1)),
        "loglik": mean(logliks),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Export election data from R (clean numeric codes)
    r_export = """
library(poLCA)
data(election)
write.csv(election, "/tmp/polca_election_bench.csv", row.names=FALSE)
"""
    subprocess.run(["Rscript", "-e", r_export], check=True, capture_output=True)
    df = pl.read_csv("/tmp/polca_election_bench.csv", null_values="NA")
    df = _extract_factor_code(df)

    n_runs = 20
    print(f"Benchmarking {n_runs} runs on election dataset")
    print(f"  N={df.shape[0]}, J={J}, R={NCLASS}, S={S}")
    print(f"  Model: {PY_FORMULA}")
    print()

    print("Running R poLCA ...")
    r_res = run_r_benchmark(n_runs)
    print(f"  mean   = {r_res['mean_time']:.4f} s")
    print(f"  median = {r_res['median_time']:.4f} s")
    print(f"  sd     = {r_res['sd_time']:.4f} s")
    print(f"  loglik = {r_res['loglik']:.4f}")
    print()

    print("Running pypolca (C++, with SE) ...")
    py_res = run_python_benchmark(df, n_runs, calc_se=True)
    print(f"  mean   = {py_res['mean_time']:.4f} s")
    print(f"  median = {py_res['median_time']:.4f} s")
    print(f"  sd     = {py_res['sd_time']:.4f} s")
    print(f"  loglik = {py_res['loglik']:.4f}")
    print()

    speedup = r_res["mean_time"] / py_res["mean_time"]
    print(f"Speed-up (with SE): {speedup:.1f}\u00d7")

    print("\nRunning pypolca (C++, without SE) ...")
    py_res_nose = run_python_benchmark(df, n_runs, calc_se=False)
    print(f"  mean   = {py_res_nose['mean_time']:.4f} s")
    print(f"  median = {py_res_nose['median_time']:.4f} s")
    print(f"  sd     = {py_res_nose['sd_time']:.4f} s")
    print(f"  loglik = {py_res_nose['loglik']:.4f}")
    print()

    speedup_nose = r_res["mean_time"] / py_res_nose["mean_time"]
    print(f"Speed-up (without SE): {speedup_nose:.1f}\u00d7")


if __name__ == "__main__":
    main()
