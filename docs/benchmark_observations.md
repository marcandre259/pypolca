# Benchmark Observations: pypolca vs R poLCA

## Raw numbers

| Dataset size | R poLCA (mean) | pypolca (mean) | Speed-up |
|--------------|----------------|----------------|----------|
| n = 319      | 0.0247 s       | 0.0174 s       | **1.4×** |
| n = 500      | 0.0133 s       | 0.0104 s       | **1.3×** |
| n = 2 000    | 0.0405 s       | 0.0455 s       | **0.9×** |
| n = 10 000   | 0.1822 s       | 0.2490 s       | **0.7×** |

Both engines converged to identical log-likelihoods with the same starting values, so the slowdown is pure computational overhead, not algorithmic divergence.

## Concrete bottlenecks (line-by-line)

### 1. `compute_log_ylik` is called **twice** per EM iteration (`em_engine.cpp`)

**Location:** `em_engine.cpp`, inside the `while` loop.

**What happens:**
1. `posterior = e_step(data, p, prior, nclass);`  ← internally calls `compute_log_ylik` and allocates a new `N × R` matrix.
2. A few lines later: `Eigen::MatrixXd log_lik_mat = compute_log_ylik(data, p, nclass);` ← **recomputes the exact same quantities** and allocates *another* `N × R` matrix.

**Cost:** For n = 10 000, R = 2, J = 4, binary items, that is 20 000 × 2 = 40 000 double allocations per iteration, plus the redundant O(N·R·J) arithmetic. R poLCA computes class-conditional likelihoods once and reuses them for both the posterior and the overall log-likelihood.

**Fix:** Refactor `e_step` to return the log-likelihood matrix as an out-parameter, or compute the overall log-likelihood directly from the quantities already inside `e_step`.

---

### 2. `compute_log_ylik` does a branch-heavy category loop (`math_ops.cpp`, lines 24-45)

**Code:**
```cpp
for (int i = 0; i < N; i++) {
  for (int r = 0; r < R; r++) {
    int current_choice_pos = 0;
    for (int j = 0; j < M; j++) {
      int obs_value = data.y(i, j);
      int cat_choices = data.num_choices[j];
      for (int k = 0; k < cat_choices; k++) {
        int idx = r * sum_choices + current_choice_pos;
        double prob_rjk = vecprobs(idx);
        if (obs_value == k + 1) {          // ← executed K times per item
          if (prob_rjk <= 0) {
            log_ylik(i, r) = -inf;
          } else {
            log_ylik(i, r) += std::log(prob_rjk);
          }
        }
        current_choice_pos++;
      }
    }
  }
}
```

**Problem:** For every observation, class, and item, the code loops over *all* categories and checks `if (obs_value == k + 1)`. On binary items that means the branch is taken once and mispredicted once per item. With four binary items, that is 8 branch mispredictions per (observation, class) pair. R poLCA almost certainly indexes directly into the probability vector using `obs_value` as an offset, eliminating both the inner `k` loop and the branch.

**Fix:** Replace the inner category loop with direct indexing:
```cpp
int idx = r * sum_choices + current_choice_pos + (obs_value - 1);
double prob_rjk = vecprobs(idx);
log_ylik(i, r) += std::log(prob_rjk);
```
This drops the complexity from O(N·R·J·K) to O(N·R·J).

---

### 3. `m_step_probs` has atrocious memory locality (`math_ops.cpp`, lines 75-100)

**Code:**
```cpp
for (int r = 0; r < nclass; r++) {
  for (int j = 0; j < M; j++) {
    // ...
    for (int k = 0; k < current_choices; k++) {
      for (int i = 0; i < N; i++) {       // ← innermost loop walks N rows
        if (y(i, j) == k + 1) {
          vecprobs(...) += posterior(i, r); // ← column-major matrix, row i varies fastest
        }
      }
    }
  }
}
```

**Problem:** `posterior` is an `Eigen::MatrixXd` which defaults to **column-major** storage. The innermost loop varies `i` (row index), so the access `posterior(i, r)` strides by `nclass` bytes (since columns are stored contiguously). For nclass = 2 that is a 16-byte stride - small, but the loop also jumps randomly depending on which observations match category `k`. The result is cache-line trashing. R poLCA's C code likely iterates observation-major: for each observation it updates all (r, j, k) counters, keeping the observation row in cache.

**Fix:** Swap the loop order. Iterate over observations in the outermost loop, accumulate into local per-class counters, and write back once per item. This also eliminates the repeated `sum_posteriors` accumulation which is currently done in a separate loop.

---

### 4. Per-iteration heap allocations everywhere

Every EM iteration allocates:
- `e_step` → `log_ylik` (N × R)
- `e_step` → `posterior` (N × R)
- `fit_em` → `log_lik_mat` (N × R)  [duplicate of the first]
- `m_step_probs` → `vecprobs` (R × sum_choices)
- `update_beta` → `updated_prior` (N × R)  [only when covariates are used]

For n = 10 000, R = 2, each `MatrixXd` is ~160 KB. With 40-50 EM iterations on a well-specified model, that is tens of megabytes of heap churn per fit. R poLCA allocates its workspace once and reuses it across iterations.

**Fix:** Add scratch buffers to the `Data` struct (or a new `Workspace` struct) and pass them as `Eigen::Ref` or raw pointers so that `e_step`, `m_step_probs`, etc. write in-place instead of returning new matrices.

---

### 5. Newton-Raphson step does redundant prior recomputation (`math_ops.cpp`, lines 175-183)

**Code:**
```cpp
auto [gradients, hessians] =
    compute_beta_derivatives(data, posterior, prior, beta, nclass);
// ...
Eigen::MatrixXd updated_prior =
    compute_prior_from_beta(x, updated_beta, nclass);
```

`compute_prior_from_beta` loops over all N observations and computes a softmax for each. Inside the EM loop this is called once per Newton-Raphson iteration. When there are no covariates (the common case, `x` is just an intercept), this is entirely unnecessary because the prior is updated from the posterior column means in `fit_em` anyway.

**Fix:** Skip `update_beta` entirely when `S == 1` (no covariates). The code already has a branch for that in `fit_em`, but only for the non-covariate prior update; the redundant call is not present for the intercept-only case, so this point is minor unless covariates are introduced.

---

### 6. The compiler cannot auto-vectorise the hot loop because of four SIMD killers

The hottest code path is the `i` loop inside `compute_log_ylik` (and again inside `e_step`). Here is the exact body the compiler sees for each iteration when `i` advances:

```cpp
int idx = r * sum_choices + current_choice_pos;
double prob_rjk = vecprobs(idx);          // 1. gather-like load
if (obs_value == k + 1) {                // 2. data-dependent branch
    if (prob_rjk <= 0) {
        log_ylik(i, r) = -inf;
    } else {
        log_ylik(i, r) += std::log(prob_rjk); // 3. non-inlined libc call
    }
}
```

**Why the compiler gives up on SIMD:**

1. **`std::log` is a libc function call.** Even with `-O3 -march=native`, `std::log` is not inlined (it lives in `libm`). A loop containing an external function call is not auto-vectorised unless the compiler has a vector math library (SVML, SLEEF, etc.) and is explicitly told to use it. The current `CMakeLists.txt` does not enable anything of the sort.

2. **Data-dependent branches.** The `if (obs_value == k + 1)` condition depends on the data matrix `y(i, j)`. In SIMD, all lanes must execute the same instruction; a data-dependent branch forces lane masking or scalar fall-back, which GCC/Clang refuse to generate automatically for this loop structure.

3. **Gather-like loads from `vecprobs`.** For a fixed observation `i`, the inner `r` loop computes `idx = r * sum_choices + current_choice_pos`. On the cheating dataset (`sum_choices = 8`), consecutive `r` values load from offsets 0, 8, 16, 24 in the `vecprobs` array. AVX2 has `_mm256_i32gather_pd`, but it is slow (high latency, low throughput). ARM NEON has no gather instruction at all; it would scalarise the load entirely. So even if you removed the branch and the `log` call, the compiler still cannot issue a single wide vector load for the probabilities.

4. **Scatter-like stores to `log_ylik`.** `log_ylik` is an `Eigen::MatrixXd` stored in column-major order. The access `log_ylik(i, r)` means that for fixed `i`, consecutive `r` values write to memory locations separated by `N * sizeof(double)` bytes (e.g., 80 000 bytes for n = 10 000). AVX2 has scatter stores, but they are even slower than gathers. NEON does not have them. The compiler will not vectorise a loop that writes to strided locations.

**What R poLCA's C code likely does instead:**  
The reference implementation uses flat C arrays and direct pointer arithmetic. Because it computes the log-likelihood in a single pass without an `r` loop (or with the classes unrolled manually), the memory access pattern is sequential: it walks the data array forward and updates contiguous output slots. There are no gather/scatter operations, and because it indexes directly by `obs_value` there is no inner `k` branch either. This is trivial for the compiler to auto-vectorise with simple `-O3`.

**What you would need to do to get SIMD:**  
Simply adding `-march=native` will not help this loop. You would need to restructure the algorithm so that:
- The inner loop over `r` (classes) is replaced by an outer loop over `r` with a sequential pass over observations. This makes both reads and writes contiguous.
- `std::log` is replaced by either a vector math library call (SLEEF, SVML) or a precomputed log-lookup table.
- Branches are eliminated by restructuring the category indexing (see point 2 above).

---

## Why R poLCA wins at scale

R poLCA's C core (`src/poLCA.c` in the R package source) is a single file with flat arrays and tight loops. It does not use an abstraction like Eigen, which means:
- No hidden heap allocations inside arithmetic expressions.
- Direct pointer arithmetic on contiguous C arrays.
- Likely a single-pass design: one loop over observations computes the likelihoods, posteriors, and log-likelihood simultaneously.

Your C++ code uses Eigen for convenience but pays for it with:
1. **Allocation overhead** (new matrix per function call).
2. **Loop abstraction overhead** (triple nested loops with strided access instead of single-pass pointer walks).
3. **Double computation** (`compute_log_ylik` called twice per iteration).

## Recommended fixes (in order of expected impact)

1. **Eliminate the duplicate `compute_log_ylik` call** in `fit_em`. Expected gain: ~20-30 %.
2. **Remove the category loop in `compute_log_ylik`**. Use `obs_value` as an index. Expected gain: ~15-25 % on binary data, proportional to the number of categories.
3. **Rewrite `m_step_probs` with observation-outer loop order** and write into pre-allocated counters. Expected gain: ~20-30 % on large N.
4. **Pre-allocate workspace matrices** and pass them by `Eigen::Ref` or pointer. Expected gain: ~10-20 %, mostly visible when the number of iterations is high.
5. **Compile with `-march=native -O3`** and check whether the compiler auto-vectorises the rewritten loops. If not, consider `#pragma omp simd` on the observation loop.

## Detailed solution designs

### 1. Eliminate the duplicate `compute_log_ylik` call

**Current problem:** `e_step` calls it, then `fit_em` calls it again 5 lines later.

**How to fix:** Change `e_step` to return *both* the posterior matrix and the log-likelihood matrix, or compute the total log-likelihood inside `e_step` and return it.

**Concrete approach A - return log-likelihood as an out-parameter:**

```cpp
void e_step(const Data &data, const Params &p,
            const Eigen::MatrixXd &prior, int nclass,
            Eigen::MatrixXd &out_posterior,   // pre-allocated N x R
            Eigen::MatrixXd &out_log_ylik);   // pre-allocated N x R
```

Inside `e_step`, after computing `log_ylik(i, r)`, write it to `out_log_ylik(i, r)` before you overwrite it with the posterior. Then in `fit_em`:

```cpp
// Before the loop, allocate once:
Eigen::MatrixXd log_ylik(N, nclass);
Eigen::MatrixXd posterior(N, nclass);

// Inside the loop:
e_step(data, p, prior, nclass, posterior, log_ylik);
// ... m-step ...
// Compute total log-likelihood directly from log_ylik and prior, no second call
```

**Concrete approach B - compute total log-likelihood inside `e_step`:**

Add a `double &out_loglik` parameter to `e_step`. While you are already looping over `i` and doing the logsumexp, accumulate the total:

```cpp
double total_loglik = 0.0;
for (int i = 0; i < N; i++) {
    // ... compute log_nums ...
    double log_denom = compute_logsumexp(log_nums);
    total_loglik += log_denom;
    for (int r = 0; r < R; r++) {
        posterior(i, r) = exp(log_nums(r) - log_denom);
    }
}
out_loglik = total_loglik;
```

This is slightly more accurate than the current manual loop in `fit_em` because you reuse the `log_nums` you already computed.

---

### 2. Replace the category loop with direct indexing

**Current problem:**
```cpp
for (int k = 0; k < cat_choices; k++) {
    int idx = r * sum_choices + current_choice_pos;
    if (obs_value == k + 1) {
        log_ylik(i, r) += std::log(vecprobs(idx));
    }
    current_choice_pos++;
}
```

**How to fix:** Jump directly to the observed category.

```cpp
for (int j = 0; j < M; j++) {
    int obs_value = data.y(i, j);
    if (obs_value == 0) continue;           // missing data
    int idx = r * sum_choices + current_choice_pos + (obs_value - 1);
    log_ylik(i, r) += std::log(vecprobs(idx));
    current_choice_pos += data.num_choices[j];
}
```

**Why this works:** You remove the inner `k` loop entirely. Complexity drops from O(N*R*J*K) to O(N*R*J). On binary data that is a 2x speedup in this function alone. The branch is now only on missing data (`obs_value == 0`), which is rare and predictable.

---

### 3. Fix `m_step_probs` loop order and eliminate the inner branch

**Current problem:** The inner loop over `i` has a data-dependent branch `if (y(i,j) == k+1)` which is unpredictable.

**How to fix:** Iterate over observations in the outer loop, accumulate into a local buffer, and write back once per item.

```cpp
Eigen::VectorXd m_step_probs(const Data &data, const Eigen::MatrixXd &posterior,
                             const std::vector<int> &num_choices, int nclass)
{
    int N = data.n_obs();
    int M = data.n_items();
    int total_choices = /* sum */;
    Eigen::VectorXd vecprobs = Eigen::VectorXd::Zero(nclass * total_choices);

    for (int j = 0; j < M; j++) {
        int K = num_choices[j];
        Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(nclass, K);

        for (int i = 0; i < N; i++) {
            int cat = data.y(i, j);
            if (cat == 0) continue;
            int c = cat - 1;
            for (int r = 0; r < nclass; r++) {
                acc(r, c) += posterior(i, r);
            }
        }

        int pos = /* cumulative offset for item j */;
        for (int r = 0; r < nclass; r++) {
            double row_sum = acc.row(r).sum();
            for (int k = 0; k < K; k++) {
                vecprobs(r * total_choices + pos + k) = acc(r, k) / row_sum;
            }
        }
    }
    return vecprobs;
}
```

**Why this works:**
- The unpredictable branch is gone. Instead you check `if (cat == 0)` once per observation. Missing data is usually rare, so this branch is highly predictable.
- You read `posterior(i, r)` sequentially down column `r` (contiguous in Eigen column-major storage) because the `r` loop is innermost.
- The `acc` matrix is tiny (e.g., 2x2 = 4 doubles) and lives in L1 cache.

**Even better variant:** If you want to avoid the `r` loop inside the `i` loop, transpose the problem:

```cpp
for (int r = 0; r < nclass; r++) {
    for (int j = 0; j < M; j++) {
        int K = num_choices[j];
        std::vector<double> acc(K, 0.0);
        for (int i = 0; i < N; i++) {
            int cat = data.y(i, j);
            if (cat > 0) acc[cat - 1] += posterior(i, r);
        }
        // normalize and write to vecprobs
    }
}
```

This is the same structure but now `posterior(i, r)` is read sequentially down a single column (perfect locality) and `acc` is a tiny stack array.

---

### 4. Pre-allocate workspace to kill heap churn

**Current problem:** Every EM iteration allocates fresh `Eigen::MatrixXd` objects.

**How to fix:** Add scratch buffers to a new `Workspace` struct and pass them by reference.

```cpp
struct Workspace {
    Eigen::MatrixXd log_ylik;
    Eigen::MatrixXd posterior;
    Eigen::MatrixXd prior;
};

// In fit_em:
Workspace ws;
ws.log_ylik.resize(N, nclass);
ws.posterior.resize(N, nclass);
ws.prior.resize(N, nclass);

while (/* ... */) {
    e_step(data, p, ws.prior, nclass, ws.posterior, ws.log_ylik);
    m_step_probs(data, ws.posterior, data.num_choices, nclass, p.vecprobs);
    // ...
}
```

**Key detail:** Change the functions to take `Eigen::Ref<Eigen::MatrixXd>` instead of returning `Eigen::MatrixXd`. This lets you write directly into the pre-allocated buffer without copies.

```cpp
void e_step(const Data &data, const Params &p,
            const Eigen::MatrixXd &prior, int nclass,
            Eigen::Ref<Eigen::MatrixXd> out_posterior,
            Eigen::Ref<Eigen::MatrixXd> out_log_ylik);
```

---

### 5. Re-structure for SIMD

Simply adding `-march=native` will **not** help the current loops because of the four SIMD killers documented above. Here is the actual restructuring you need:

**Step A - make the inner loop contiguous:**

The current `compute_log_ylik` has the class loop `r` inside the observation loop `i`. That means for a fixed `i`, you write to `log_ylik(i, 0)`, `log_ylik(i, 1)`, etc. These are `N * sizeof(double)` bytes apart because `log_ylik` is column-major. SIMD wants to load/store contiguous chunks.

Flip the loop order so `r` is outer and `i` is inner:

```cpp
for (int r = 0; r < R; r++) {
    for (int i = 0; i < N; i++) {
        double ll = 0.0;
        int pos = 0;
        for (int j = 0; j < M; j++) {
            int obs = data.y(i, j);
            if (obs > 0) {
                ll += std::log(vecprobs(r * sum_choices + pos + obs - 1));
            }
            pos += num_choices[j];
        }
        log_ylik(i, r) = ll;
    }
}
```

Now for fixed `r`, the inner `i` loop writes to `log_ylik(0, r)`, `log_ylik(1, r)`, `log_ylik(2, r)`... which is a **contiguous column** in Eigen column-major storage. The compiler can now vectorise the `i` loop with SIMD stores.

**Step B - remove `std::log` from the hot loop:**

Even with contiguous access, `std::log` is still a libc call and the compiler won't vectorise across it. You have two options:

**Option 1 - Pre-compute log-probabilities once per M-step:**

After `m_step_probs` updates `p.vecprobs`, immediately compute `log_vecprobs = vecprobs.array().log()` and store it in `Params`. Then `compute_log_ylik` becomes:

```cpp
ll += log_vecprobs(r * sum_choices + pos + obs - 1);
```

This is just a load and an add - no function call, fully vectorisable.

**Option 2 - Use a vector math library:**

Link against Intel SVML (if using ICC) or SLEEF (portable). Then `std::log` can be replaced with a vectorisable intrinsic. This is more work than Option 1.

**Step C - handle the `if (obs > 0)` branch:**

With the loop order flipped, the branch is now inside the `j` loop (which is inside the `i` loop). The compiler can vectorise the `i` loop if the `j` loop is unrolled and the branch is converted to a masked operation. But the simplest fix is to separate the missing-data handling:

```cpp
// Pre-compute a boolean mask or separate the data into complete cases
// If na_rm=true (the default), you already drop missing rows in Python.
// In C++ you can assume no missing data and skip the branch entirely.
```

If you always call `drop_nulls()` in Python before sending data to C++, then `obs > 0` is guaranteed and you can delete the `if` check. If you want to support missing data in C++, zero out the contribution instead of branching:

```cpp
int idx = r * sum_choices + pos + (obs - 1);
double contrib = (obs > 0) ? log_vecprobs(idx) : 0.0;
ll += contrib;
```

This is branchless and vectorises cleanly.

---

### 6. What the full hot path should look like after all fixes

Here is the shape of `compute_log_ylik` after applying fixes 2, 5A, and 5B:

```cpp
void compute_log_ylik(const Data &data, const Params &p,
                      int nclass, Eigen::Ref<Eigen::MatrixXd> out_log_ylik)
{
    const int N = data.n_obs();
    const int R = nclass;
    const int M = data.n_items();
    const auto& log_probs = p.log_vecprobs;  // pre-computed logs

    int sum_choices = 0;
    for (int j = 0; j < M; j++) sum_choices += data.num_choices[j];

    for (int r = 0; r < R; r++) {
        int r_offset = r * sum_choices;
        for (int i = 0; i < N; i++) {
            double ll = 0.0;
            int pos = 0;
            for (int j = 0; j < M; j++) {
                int obs = data.y(i, j);           // guaranteed > 0 if na_rm=true
                ll += log_probs(r_offset + pos + obs - 1);
                pos += data.num_choices[j];
            }
            out_log_ylik(i, r) = ll;
        }
    }
}
```

- No `std::log` call in the loop.
- No category loop (`k` is gone).
- No missing-data branch (handled upstream).
- `i` loop writes contiguously down a column.
- The compiler can now auto-vectorise the `i` loop with AVX2/NEON.

---

## Summary of expected impact

| Fix | Effort | Expected speed-up |
|-----|--------|-------------------|
| 1. Remove duplicate `compute_log_ylik` | 30 min | ~20-30% |
| 2. Direct category indexing | 15 min | ~15-25% on binary data |
| 3. Reorder `m_step_probs` | 30 min | ~20-30% on large N |
| 4. Pre-allocate workspace | 1 hour | ~10-20% (iterations-dependent) |
| 5. Restructure for SIMD | 2-3 hours | ~30-50% (requires 1-4 first) |

**Cumulative potential:** 2-4x faster on large datasets, which would put you well ahead of R poLCA.

## Files to edit

- `src/cpp/core/em_engine.cpp` - remove duplicate likelihood computation
- `src/cpp/core/math_ops.cpp` - restructure `compute_log_ylik` and `m_step_probs`
- `src/cpp/include/pypolca/types.h` - add scratch buffers if doing in-place allocation
- `src/cpp/CMakeLists.txt` - verify / add `-O3 -march=native`
