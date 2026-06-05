"""Scaling benchmark: pypolca (C++) vs R poLCA on larger synthetic data."""

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
    nclass = len(probs_start_list[0])
    flat = []
    for r in range(nclass):
        for item_mat in probs_start_list:
            flat.extend(item_mat[r])
    return np.array(flat, dtype=np.float64)


def build_r_benchmark_script(data_json: str, n_runs: int) -> str:
    return f"""
library(poLCA)
library(jsonlite)

data <- fromJSON('{data_json}')
probs_start <- list(
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE),
  matrix(c(0.9, 0.1, 0.4, 0.6), nrow=2, byrow=TRUE)
)
f <- cbind(Y1, Y2, Y3, Y4, Y5) ~ 1

times <- numeric({n_runs})
for (i in seq_len({n_runs})) {{
  t0 <- Sys.time()
  res <- poLCA(f, data=data, nclass=2, maxiter=100, tol=1e-8,
               verbose=FALSE, probs.start=probs_start)
  t1 <- Sys.time()
  times[i] <- as.numeric(t1 - t0, units="secs")
}}
cat(mean(times), "\n")
"""


def run_r_benchmark(df: pl.DataFrame, n_runs: int = 10) -> dict:
    records = df.select(["Y1", "Y2", "Y3", "Y4", "Y5"]).to_dicts()
    data_json = json.dumps(records)
    script = build_r_benchmark_script(data_json, n_runs)
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
        times = [float(x) for x in result.stdout.strip().split()]
        return {
            "mean_time": mean(times),
            "median_time": median(times),
            "sd_time": float(np.std(times, ddof=1)),
        }
    finally:
        os.unlink(r_file)


def run_python_benchmark(df: pl.DataFrame, n_runs: int = 10, calc_se: bool = True) -> dict:
    probs_start_py = _flatten_probs_start([[[0.9, 0.1], [0.4, 0.6]] for _ in range(5)])
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fit(
            "cbind(Y1, Y2, Y3, Y4, Y5) ~ 1",
            df,
            nclass=2,
            maxiter=100,
            tol=1e-8,
            probs_start=probs_start_py,
            calc_se=calc_se,
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "mean_time": mean(times),
        "median_time": median(times),
        "sd_time": float(np.std(times, ddof=1)),
    }


def main():
    sizes = [500, 2000, 10000]
    for n in sizes:
        np.random.seed(42)
        # Generate synthetic 5-item binary data
        y = np.random.randint(1, 3, size=(n, 5), dtype=np.int64)
        df = pl.DataFrame(
            {
                "Y1": y[:, 0],
                "Y2": y[:, 1],
                "Y3": y[:, 2],
                "Y4": y[:, 3],
                "Y5": y[:, 4],
            }
        )

        n_runs = 5 if n >= 5000 else 10
        print(f"\nn={n}, 5 binary items, 2 classes ({n_runs} runs)")
        print("-" * 40)

        r_res = run_r_benchmark(df, n_runs)
        print(
            f"R poLCA              mean={r_res['mean_time']:.4f}s  median={r_res['median_time']:.4f}s"
        )

        py_res_se = run_python_benchmark(df, n_runs, calc_se=True)
        print(
            f"pypolca (with SE)    mean={py_res_se['mean_time']:.4f}s  median={py_res_se['median_time']:.4f}s"
        )

        py_res_nose = run_python_benchmark(df, n_runs, calc_se=False)
        print(
            f"pypolca (no SE)      mean={py_res_nose['mean_time']:.4f}s  median={py_res_nose['median_time']:.4f}s"
        )

        speedup_se = r_res["mean_time"] / py_res_se["mean_time"]
        speedup_nose = r_res["mean_time"] / py_res_nose["mean_time"]
        print(f"Speed-up (with SE): {speedup_se:.1f}×  (no SE): {speedup_nose:.1f}×")


if __name__ == "__main__":
    main()
