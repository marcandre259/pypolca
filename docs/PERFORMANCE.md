# poLCA Performance Analysis

This document assesses the speed/performance gains from the package's C functions, identifies other components that could be sped up, and describes how to measure performance in RStudio.

---

## 1. What the C functions do (and why they matter)

poLCA offloads the **inner loops of the EM algorithm** to C via `.C()` calls. These are the hot paths:

| C function | R wrapper | What it computes |
|---|---|---|
| `ylik` | `poLCA.ylik.C` | Class-conditional response likelihoods for every observation (N × R) |
| `postclass` | `poLCA.postClass.C` | Posterior class membership probabilities (E-step) |
| `probhat` | `poLCA.probHat.C` | Updated class-conditional response probabilities (M-step) |
| `d2lldbeta2` | `poLCA.dLL2dBeta.C` | Gradient and Hessian for the Newton-Raphson covariate update |

These routines contain tight, triply-nested loops over observations (N), items (J), response categories (K), and latent classes (R). In pure R, every scalar operation carries overhead: type checking, garbage-collection guards, and bounds-checked vector indexing. The C versions use raw pointers and simple `for` loops, so per-iteration overhead is essentially zero.

### Estimated gain

For the core EM steps, the C versions are likely **1–2 orders of magnitude faster** than naive R loops would be. However, the *overall* function speedup is smaller because a lot of work still happens in R around the C calls—model-frame construction, list manipulation, standard-error calculation, etc.

---

## 2. Other components that could be sped up

### 2.1 Standard-error calculation (`R/poLCA.se.R`)

The score matrix is built iteratively with `cbind` inside nested loops:

```r
s <- NULL
for (r in 1:R) {
    for (j in 1:J) {
        s <- cbind(s, rgivy[,r] * t(t(y[[j]][,2:K.j[j]]) - probs[[j]][r,2:K.j[j]]))
    }
}
```

Growing a matrix with `cbind` inside a loop is a classic R anti-pattern. It re-allocates and copies the entire matrix on every iteration, making it **O(N²)** in memory movement. For large models this can easily dominate runtime.

**Fix:** Pre-allocate `s` with `matrix(0, nrow = N, ncol = ...)` and fill it by column index.

---

### 2.2 Data compression (`R/poLCA.compress.R`)

```r
for (i in 2:nrow(ym.sorted)) {
    if (sum(ym.sorted[i,] == ym.sorted[i-1,]) == vars) {
        freq[curpos] <- freq[curpos] + 1
    } else {
        datamat <- rbind(datamat, ym.sorted[i,])
        freq <- c(freq, 1)
        curpos <- curpos + 1
    }
}
```

Again, `rbind` and `c` in a loop are expensive. For large N this is punishing.

**Fix:** Replace with `dplyr::count(across(everything()))`, `aggregate()`, or at least pre-allocate and index into `datamat`.

---

### 2.3 Redundant `ylik` computation in the main loop

In `R/poLCA.R`, every EM iteration does:

```r
rgivy <- poLCA.postClass.C(prior, vp, y)       # C code internally calls ylik()
vp$vecprobs <- poLCA.probHat.C(rgivy, y, vp)
# ...
llik[iter] <- sum(log(rowSums(prior * poLCA.ylik.C(vp, y)) - log(.Machine$double.xmax)))
```

`postclass` already computes the observation likelihoods in C but **does not return them**. The explicit `poLCA.ylik.C` call recalculates the same quantities just to evaluate the log-likelihood.

**Fix:** Modify the C `postclass` function (or create a variant) to return the marginal log-likelihood or the raw likelihood matrix alongside the posteriors. This saves one full C call per EM iteration.

---

### 2.4 `ginv()` in the Newton-Raphson step

```r
b <- b + ginv(-dd$hess) %*% dd$grad
```

`MASS::ginv()` computes the Moore-Penrose inverse via SVD. If the Hessian is well-conditioned (it usually is here), `solve(-dd$hess, dd$grad)` is much faster and more stable.

---

### 2.5 Serial replications (`nrep`)

The `for (repl in 1:nrep)` loop in `poLCA()` is purely serial. Because each replication is independent, this is an easy target for `parallel::mclapply()` or `future.apply::future_lapply()`.

---

## 3. How to assess performance in RStudio

### 3.1 Interactive profiling with `profvis`

Install if needed:

```r
install.packages("profvis")
```

Run your model inside `profvis()`:

```r
library(poLCA)
library(profvis)

data(cheating)
f <- cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ 1

profvis({
  out <- poLCA(f, cheating, nclass = 2, nrep = 3, verbose = FALSE)
})
```

In RStudio this opens an interactive pane showing:
- **Time spent on each line** (flame graph)
- **Memory allocations** (which lines trigger GC)
- **Call stack depth**

Look for wide horizontal bars that belong to R code (not `.C` calls). If `poLCA.se` or `poLCA.compress` show up as thick blocks, they are the low-hanging fruit.

---

### 3.2 Quick line profiling with `Rprof`

```r
Rprof(tmp <- tempfile())
out <- poLCA(f, cheating, nclass = 2, nrep = 10, verbose = FALSE)
Rprof(NULL)
summaryRprof(tmp)
```

This prints a flat table of which functions consumed the most time.

---

### 3.3 Micro-benchmark specific pieces with `bench`

Isolate the cost of the `cbind` anti-pattern or compare `ginv` vs `solve`:

```r
bench::mark(
  ginv  = ginv(-hess) %*% grad,
  solve = solve(-hess, grad),
  check = FALSE,
  iterations = 100
)
```

---

## 4. Bottom line

- The C code removes the biggest bottleneck (the EM inner loops), but the **R-side standard-error and data-compression routines are the next targets**—they use grow-by-concatenation patterns that scale poorly.
- The easiest wins are:
  1. Pre-allocating matrices in `poLCA.se`
  2. Rewriting `poLCA.compress` without looped `rbind`
  3. Avoiding the redundant `ylik` call by having `postclass` return likelihoods
- Use **`profvis::profvis()`** in RStudio to see exactly which lines eat the most time and memory.
