"""Benchmark: pypolca (C++) vs R poLCA on the cheating dataset."""

import json
import os
import subprocess
import tempfile
import time
from statistics import mean, median

import numpy as np
import polars as pl

from pypolca import fit


def _flatten_probs_start(probs_start_list):
    """R-style list-of-matrices → pypolca flat vector."""
    nclass = len(probs_start_list[0])
    flat = []
    for r in range(nclass):
        for item_mat in probs_start_list:
            flat.extend(item_mat[r])
    return np.array(flat, dtype=np.float64)


def build_r_benchmark_script(n_runs: int) -> str:
    return f"""
library(poLCA)
library(jsonlite)

# Load cheating data
data(cheating)

probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)

f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1

times <- numeric({n_runs})
logliks <- numeric({n_runs})
for (i in seq_len({n_runs})) {{
  t0 <- Sys.time()
  res <- poLCA(f, data=cheating, nclass=2, maxiter=1000, tol=1e-10,
               verbose=FALSE, probs.start=probs_start)
  t1 <- Sys.time()
  times[i] <- as.numeric(t1 - t0, units="secs")
  logliks[i] <- res$llik
}}

out <- list(mean_time=mean(times), median_time=median(times),
            sd_time=sd(times), loglik=mean(logliks))
cat(toJSON(out, digits=12))
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
            timeout=120,
        )
        raw = json.loads(result.stdout)
        # jsonlite may wrap length-1 vectors
        for key in ("mean_time", "median_time", "sd_time", "loglik"):
            if isinstance(raw[key], list) and len(raw[key]) == 1:
                raw[key] = raw[key][0]
        return raw
    finally:
        os.unlink(r_file)


def run_python_benchmark(df: pl.DataFrame, n_runs: int = 20, calc_se: bool = True) -> dict:
    probs_start_py = _flatten_probs_start(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.9, 0.1], [0.4, 0.6]],
        ]
    )

    times = []
    logliks = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        res = fit(
            "cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1",
            df,
            nclass=2,
            maxiter=1000,
            tol=1e-10,
            probs_start=probs_start_py,
            calc_se=calc_se,
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


def main():
    # Export cheating data from R
    r_export = """
library(poLCA)
data(cheating)
write.csv(cheating, "/tmp/polca_cheating_bench.csv", row.names=FALSE)
"""
    subprocess.run(["Rscript", "-e", r_export], check=True, capture_output=True)
    df = pl.read_csv("/tmp/polca_cheating_bench.csv")
    for col in ["LIEEXAM", "LIEPAPER", "FRAUD", "COPYEXAM"]:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Int64))

    n_runs = 20
    print(f"Benchmarking {n_runs} runs on cheating dataset (n={df.shape[0]}, J=4, R=2)")
    print()

    print("Running R poLCA …")
    r_res = run_r_benchmark(n_runs)
    print(f"  mean   = {r_res['mean_time']:.4f} s")
    print(f"  median = {r_res['median_time']:.4f} s")
    print(f"  sd     = {r_res['sd_time']:.4f} s")
    print(f"  loglik = {r_res['loglik']:.4f}")
    print()

    print("Running pypolca (C++, with SE) …")
    py_res = run_python_benchmark(df, n_runs, calc_se=True)
    print(f"  mean   = {py_res['mean_time']:.4f} s")
    print(f"  median = {py_res['median_time']:.4f} s")
    print(f"  sd     = {py_res['sd_time']:.4f} s")
    print(f"  loglik = {py_res['loglik']:.4f}")
    print()

    speedup = r_res["mean_time"] / py_res["mean_time"]
    print(f"Speed-up (with SE): {speedup:.1f}×")

    print("\nRunning pypolca (C++, without SE) …")
    py_res_nose = run_python_benchmark(df, n_runs, calc_se=False)
    print(f"  mean   = {py_res_nose['mean_time']:.4f} s")
    print(f"  median = {py_res_nose['median_time']:.4f} s")
    print(f"  sd     = {py_res_nose['sd_time']:.4f} s")
    print(f"  loglik = {py_res_nose['loglik']:.4f}")
    print()

    speedup_nose = r_res["mean_time"] / py_res_nose["mean_time"]
    print(f"Speed-up (without SE): {speedup_nose:.1f}×")


if __name__ == "__main__":
    main()
