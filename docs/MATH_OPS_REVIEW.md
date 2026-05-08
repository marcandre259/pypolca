# Code Review: `math_ops.cpp` / `math_ops.h`

**Date:** 2026-05-08 (revised)  
**Scope:** Core math routines that the EM engine will call.  
**Verdict:** **All critical bugs have been fixed.** `compute_log_ylik`, `update_beta`, and `compute_beta_derivatives` are now mathematically correct. `m_step_probs` and `compute_prior_from_beta` remain sound. A handful of hygiene issues still remain (see §3).

---

## 1. Critical Bugs — RESOLVED

### 1.1 `compute_log_ylik` — ✅ Fixed
Now initializes with `setZero()` and accumulates `std::log(prob_rjk)`, with a guard against `prob_rjk <= 0`.

### 1.2 `update_beta` — ✅ Fixed
Now correctly calls `compute_prior_from_beta(x, updated_beta, nclass)` using the updated coefficients.

### 1.3 `compute_beta_derivatives` sign convention — ✅ Fixed
Signs have been flipped so the routine returns the **observed information matrix** (`I_obs = prior_term - posterior_term`), matching the `poLCA` C convention. The diagonal and off-diagonal formulas now read:

```cpp
// Diagonal
hessians(row, col) += x(i, m) * x(i, l) *
    ( prior(i, r) * (1 - prior(i, r)) - posterior(i, r) * (1 - posterior(i, r)) );

// Off-diagonal
hessians(row, col2) += x(i, m) * x(i, l) *
    ( posterior(i, r) * posterior(i, s) - prior(i, r) * prior(i, s) );
```

This is compatible with the `+` step in `update_beta`.

### 1.4 `compute_beta_derivatives` block symmetrisation — ✅ Fixed
The `l <= m` guard was removed from the inner loop, so off-diagonal `(r, s)` blocks are now written in full. Combined with the `triangularView` mirror, the Hessian is fully symmetric.

---

## 2. Moderate Issues (correctness edge cases) — Still present

### 2.1 Missing data (`y(i,j) == 0`)

- **`compute_log_ylik`:** Missing values are correctly skipped because `k + 1` ranges from `1` to `cat_choices`, so `obs_value == k + 1` never matches `0`.
- **`m_step_probs`:** Missing observations are naturally excluded from the numerator (the indicator is false), but the denominator uses the **full** posterior sum `posterior.col(r).sum()`.  Standard LCA divides by the sum of posteriors over *non-missing* observations for that item.  With no missing data this is harmless; with missing data it biases probabilities downward.

### 2.2 `log(prior(i,r))` in `e_step`

If a prior can be exactly `0.0`, `log(0)` yields `-inf`.  The `logsumexp` trick handles this gracefully (the `-inf` terms drop out), but it is worth an explicit comment or a tiny epsilon guard if priors are ever exactly zero by construction.

---

## 3. Minor / Hygiene Issues — Still present

| Issue | Location | Fix |
|-------|----------|-----|
| Includes internal Eigen headers | `math_ops.cpp` lines 2–3 (`Eigen/src/Core/...`) | Remove; `<Eigen/Dense>` from the header is sufficient |
| Unnecessary copies of large matrices | `m_step_probs` (`y = data.y`), `compute_beta_derivatives` (`x = data.x`), `update_beta` (`x = data.x`) | Use `const auto&` or `const Eigen::MatrixXd&` |
| `compute_logsumexp` not declared in header | `math_ops.cpp` only | Add to `math_ops.h` (or move to an anonymous namespace as a detail) |
| `compute_logsumexp` passes vector by value | signature: `Eigen::VectorXd x` | Use `const Eigen::VectorXd& x` |
| No bounds check on `beta.size()` | `compute_prior_from_beta`, `update_beta` | Assert `beta.size() == x.cols() * (nclass - 1)` |
| `nclass == 1` not handled in `compute_beta_derivatives` | loop `r = 1; r < 1` is fine, but `beta_map` in `update_beta` would have 0 cols | Already handled in `compute_prior_from_beta`; derivative routines should early-return empty gradient/Hessian when `nclass == 1` |

---

## 4. Sound Routines

| Routine | Assessment |
|---------|------------|
| `compute_log_ylik` | **Correct** after fix. Accumulates log-probabilities with zero guard. |
| `compute_prior_from_beta` | **Correct.** Softmax with stable `max(0, η_max)` trick; layout matches `beta` spec. |
| `compute_logsumexp` | **Correct.** Standard `max + log(sum(exp(x - max)))` implementation. |
| `m_step_probs` (math) | **Correct** for complete data; indexing into `vecprobs` matches the `r * total_choices + pos + k` layout. |
| `update_beta` | **Correct** after fix. Uses updated beta and returns valid probability matrix. |
| `compute_beta_derivatives` | **Correct** after fix. Returns observed information matrix with fully symmetric blocks. |

---

## 5. Recommendations before wiring `em_engine`

1. ~~Rewrite `compute_log_ylik`~~ ✅ Done.
2. ~~Fix the sign in `compute_beta_derivatives`~~ ✅ Done.
3. ~~Fix block symmetrisation in `compute_beta_derivatives`~~ ✅ Done.
4. ~~Fix `update_beta`~~ ✅ Done.
5. **Add a unit test** that compares `compute_log_ylik` against a hand-rolled Python product-of-probs + log for a tiny dataset (e.g. 2 obs, 2 items, 2 classes).
6. **Add a unit test** for `update_beta`: starting from a known β, verify that the returned prior matrix is a valid stochastic matrix (rows sum to 1) and that β actually changed when the gradient is non-zero.
7. **Clean up hygiene issues** in §3 (remove internal Eigen headers, avoid copies, add `compute_logsumexp` to header, etc.).

The mathematical core of `math_ops` is now solid enough to drive the EM loop in `em_engine.cpp`.
