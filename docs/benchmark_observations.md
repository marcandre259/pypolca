---
tags: [performance, benchmark, python]
---

# Benchmark Observations: pypolca vs R poLCA

## Baseline (before any fixes)

| Dataset size | R poLCA (mean) | pypolca (mean) | Speed-up |
|--------------|----------------|----------------|----------|
| n = 319      | 0.0247 s       | 0.0174 s       | **1.4×** |
| n = 500      | 0.0133 s       | 0.0104 s       | **1.3×** |
| n = 2 000    | 0.0405 s       | 0.0455 s       | **0.9×** |
| n = 10 000   | 0.1822 s       | 0.2490 s       | **0.7×** |

Both engines converged to identical log-likelihoods with the same starting values, so the slowdown was pure computational overhead, not algorithmic divergence.

## Current results (after fixes 1–3)

| Dataset size | R poLCA (mean) | pypolca (mean) | Speed-up |
|--------------|----------------|----------------|----------|
| n = 319      | 0.0237 s       | **0.0110 s**   | **2.2×** |
| n = 500      | 0.0123 s       | **0.0056 s**   | **2.2×** |
| n = 2 000    | 0.0394 s       | **0.0207 s**   | **1.9×** |
| n = 10 000   | 0.1769 s       | **0.0991 s**   | **1.8×** |

Fixes 1–3 completely reversed the scaling trend. pypolca went from **0.7× slower** than R at n = 10 000 to **1.8× faster**.

---

## Implementation status

| # | Fix | Status | Observed impact |
|---|-----|--------|-----------------|
| 1 | Eliminate duplicate `compute_log_ylik` | ✅ Done | Removed per-iteration recompute; `e_step` now returns log-likelihood directly. |
| 2 | Direct category indexing in `compute_log_ylik` | ✅ Done | Dropped inner `k` loop; complexity O(N·R·J·K) → O(N·R·J). |
| 3 | Reorder `m_step_probs` loops | ✅ Done | `r` outer, `i` inner → contiguous column reads on `posterior`. Fixed missing-data guard and `pos += K` bug. |
| 4 | Pre-allocate workspace | 🔄 Next | `e_step` still allocates `log_ylik` internally; `m_step_probs` returns a fresh vector every iteration. |
| 5 | Restructure for SIMD | ⏳ Pending | Loop order in `compute_log_ylik` is still `i` outer / `r` inner (strided writes); `std::log` still in hot path. |

---

## Detailed notes per fix

### 1. Eliminate duplicate `compute_log_ylik` call ✅

**What changed:** `e_step` now computes the total log-likelihood while it already has `log_nums` in hand, and returns it as part of its result pair. `fit_em` no longer calls `compute_log_ylik` a second time inside the EM loop.

**Code:**
```cpp
// em_engine.cpp — inside the while loop
auto [posterior, log_lik_latest] = e_step(data, p, prior, nclass);
```

**Impact:** ~20–30 % speed-up; removes one N × R allocation per iteration.

---

### 2. Direct category indexing in `compute_log_ylik` ✅

**What changed:** Replaced the branch-heavy inner `k` loop with direct indexing via `obs_value`.

**Before:**
```cpp
for (int k = 0; k < cat_choices; k++) {
    if (obs_value == k + 1) {
        ll += std::log(vecprobs(idx));
    }
    current_choice_pos++;
}
```

**After:**
```cpp
int idx = r * sum_choices + current_choice_pos + (obs_value - 1);
ll += std::log(vecprobs(idx));
current_choice_pos += cat_choices;
```

**Impact:** ~15–25 % on binary data (proportional to number of categories). Eliminates data-dependent branches.

---

### 3. Reorder `m_step_probs` loops ✅

**What changed:** Swapped from `k`-outer / `i`-inner (random strided access) to `r`-outer / `j`-middle / `i`-inner (sequential down a column).

**Before:**
```cpp
for (int k = 0; k < current_choices; k++) {
    for (int i = 0; i < N; i++) {
        if (y(i, j) == k + 1) {
            vecprobs(...) += posterior(i, r);
        }
    }
}
```

**After:**
```cpp
for (int r = 0; r < nclass; r++) {
    for (int j = 0; j < M; j++) {
        int K = num_choices[j];
        std::vector<double> acc(K, 0.0);
        double sum = 0.0;
        for (int i = 0; i < N; i++) {
            int value = y(i, j);
            if (value > 0) {
                acc[value - 1] += posterior(i, r);
                sum += posterior(i, r);
            }
        }
        for (int k = 0; k < K; k++) {
            vecprobs(r * total_choices + pos + k) = acc[k] / sum;
        }
        pos += K;
    }
}
```

**Impact:** ~20–30 % on large N. The key win is reading `posterior(i, r)` sequentially down column `r` (contiguous in Eigen column-major storage).

---

### 4. Pre-allocate workspace 🔄 **Next step**

**What still happens every EM iteration:**
- `e_step` → allocates `log_ylik` (`N × R`) internally
- `e_step` → allocates `posterior` (`N × R`) and returns it by value
- `m_step_probs` → allocates `vecprobs` (`R × total_choices`) and returns it by value

For n = 10 000, R = 2, each `MatrixXd` is ~160 KB. With 40–50 iterations that is several megabytes of heap churn per fit.

**Fix:** Add a `Workspace` struct and pass `Eigen::Ref` out-parameters.

```cpp
struct Workspace {
    Eigen::MatrixXd log_ylik;
    Eigen::MatrixXd posterior;
};

// Allocate once in fit_em:
Workspace ws;
ws.log_ylik.resize(N, nclass);
ws.posterior.resize(N, nclass);

// Pass by reference into e_step and m_step_probs:
void e_step(const Data&, const Params&, const Eigen::MatrixXd&, int nclass,
            Eigen::Ref<Eigen::MatrixXd> out_posterior,
            Eigen::Ref<Eigen::MatrixXd> out_log_ylik,
            double& out_loglik);

void m_step_probs(const Data&, const Eigen::MatrixXd&,
                  const std::vector<int>&, int nclass,
                  Eigen::Ref<Eigen::VectorXd> out_vecprobs);
```

**Expected impact:** ~10–20 %, mostly visible when iteration count is high.

---

### 5. Restructure for SIMD ⏳ **Pending**

Even after fixes 1–3, the compiler cannot auto-vectorise `compute_log_ylik` for three reasons:

**A. Loop order causes strided writes.**

Current code (`i` outer, `r` inner):
```cpp
for (int i = 0; i < N; i++) {
    for (int r = 0; r < R; r++) {
        // ...
        log_ylik(i, r) = ll;   // strided by N * sizeof(double)
    }
}
```

Flip to `r` outer, `i` inner so the inner loop writes contiguously down a column:
```cpp
for (int r = 0; r < R; r++) {
    for (int i = 0; i < N; i++) {
        // ...
        log_ylik(i, r) = ll;   // contiguous
    }
}
```

**B. `std::log` is a libc call.**

The compiler will not vectorise across an external function call. Two options:

- **Option 1 (recommended):** Pre-compute `log_vecprobs` once per M-step, store it in `Params`, and replace `std::log(vecprobs(idx))` with a plain load.
- **Option 2:** Link a vector math library (SLEEF, SVML). More work, portable but requires build-system changes.

**C. Missing-data branch inside the inner loop.**

```cpp
if (obs_value < 1) { ... }
```

If Python already drops incomplete cases (`na_rm = true`), this branch can be deleted entirely in C++. Otherwise, replace with a branchless mask:
```cpp
double contrib = (obs > 0) ? log_probs(idx) : 0.0;
ll += contrib;
```

**Expected impact:** ~30–50 % on top of fixes 1–4 (requires all preceding fixes first).

---

## What the hot path should look like after all fixes

```cpp
void compute_log_ylik(const Data& data, const Params& p,
                      int nclass, Eigen::Ref<Eigen::MatrixXd> out_log_ylik)
{
    const int N = data.n_obs();
    const int R = nclass;
    const int M = data.n_items();
    const auto& log_probs = p.log_vecprobs;  // pre-computed once per M-step

    int sum_choices = 0;
    for (int j = 0; j < M; j++) sum_choices += data.num_choices[j];

    for (int r = 0; r < R; r++) {
        int r_offset = r * sum_choices;
        for (int i = 0; i < N; i++) {
            double ll = 0.0;
            int pos = 0;
            for (int j = 0; j < M; j++) {
                int obs = data.y(i, j);   // guaranteed > 0 if na_rm upstream
                ll += log_probs(r_offset + pos + obs - 1);
                pos += data.num_choices[j];
            }
            out_log_ylik(i, r) = ll;
        }
    }
}
```

- No `std::log` call in the loop.
- No category loop.
- No missing-data branch.
- `i` loop writes contiguously down a column → auto-vectorisable.

---

## Files to edit for remaining work

- `src/cpp/core/em_engine.cpp` — pass workspace buffers into `e_step` and `m_step_probs`
- `src/cpp/core/math_ops.cpp` — add out-parameter signatures; flip `compute_log_ylik` loop order; pre-compute `log_vecprobs`
- `src/cpp/include/pypolca/types.h` — add `log_vecprobs` to `Params` (or `Workspace`)
- `src/cpp/include/pypolca/math_ops.h` — update function signatures to use `Eigen::Ref`
