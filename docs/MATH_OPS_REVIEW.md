# Code Review: `math_ops.cpp` / `math_ops.h`

**Date:** 2026-05-08 (final revision)  
**Scope:** Core math routines that the EM engine will call.  
**Verdict:** **All bugs and hygiene issues are resolved.** The mathematical core is solid and ready for the EM loop.

---

## 1. Critical Bugs — RESOLVED

### 1.1 `compute_log_ylik` — ✅ Fixed
Initializes with `setZero()` and accumulates `std::log(prob_rjk)`, with a guard against `prob_rjk <= 0`.

### 1.2 `update_beta` — ✅ Fixed
Correctly calls `compute_prior_from_beta(x, updated_beta, nclass)` using the updated coefficients.

### 1.3 `compute_beta_derivatives` sign convention — ✅ Fixed
Returns the **observed information matrix** (`I_obs = prior_term - posterior_term`), matching the `poLCA` C convention. The diagonal and off-diagonal formulas read:

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
The `l <= m` guard was removed from the inner loop, so off-diagonal `(r, s)` blocks are written in full. Combined with the `triangularView` mirror, the Hessian is fully symmetric.

---

## 2. Moderate Issues (correctness edge cases) — RESOLVED

### 2.1 Missing data (`y(i,j) == 0`)

- **`compute_log_ylik`:** Missing values are correctly skipped because `k + 1` ranges from `1` to `cat_choices`, so `obs_value == k + 1` never matches `0`.
- **`m_step_probs`:** The denominator is the sum of posteriors over **non-missing** observations for each item (`y(i,j) > 0`), which is the standard LCA handling. Missing observations are naturally excluded from the numerator as well.

### 2.2 `log(prior(i,r))` in `e_step`

If a prior is exactly `0.0`, `log(0)` yields `-inf`. The `logsumexp` trick handles this gracefully (the `-inf` terms drop out), so no epsilon guard is required.

### 2.3 Operator precedence in `beta.size()` checks — ✅ Fixed
Both `compute_prior_from_beta` and `update_beta` previously used `nclass - 1 * M`, which C++ evaluates as `nclass - M`. This has been corrected to `(nclass - 1) * M`.

---

## 3. Minor / Hygiene Issues — RESOLVED

| Issue | Location | Fix |
|-------|----------|-----|
| Includes internal Eigen headers | `math_ops.cpp` | Already clean; only `<Eigen/Dense>` is included. |
| Unnecessary copies of large matrices | `compute_log_ylik` (`vecprobs`), `m_step_probs`, `compute_beta_derivatives`, `update_beta` | All now use `const` references. |
| `compute_logsumexp` not declared in header | `math_ops.cpp` only | Already declared in `math_ops.h` with `const Eigen::VectorXd&`. |
| `compute_logsumexp` passes vector by value | signature | Already uses `const Eigen::VectorXd&`. |
| No bounds check on `beta.size()` | `compute_prior_from_beta`, `update_beta` | Already present and now precedence-correct. |
| `nclass == 1` not handled in `compute_beta_derivatives` | loop `r = 1; r < 1` | Already handled with early return of zero gradient/Hessian. |

---

## 4. Sound Routines

| Routine | Assessment |
|---------|------------|
| `compute_log_ylik` | **Correct.** Accumulates log-probabilities with zero guard. |
| `compute_prior_from_beta` | **Correct.** Softmax with stable `max(0, η_max)` trick; layout matches `beta` spec. |
| `compute_logsumexp` | **Correct.** Standard `max + log(sum(exp(x - max)))` implementation. |
| `m_step_probs` | **Correct.** Missing-data aware; indexing into `vecprobs` matches the layout. |
| `update_beta` | **Correct.** Uses updated beta and returns valid probability matrix. |
| `compute_beta_derivatives` | **Correct.** Returns observed information matrix with fully symmetric blocks. |

---

## 5. Recommendations before wiring `em_engine`

1. ~~Rewrite `compute_log_ylik`~~ ✅ Done.
2. ~~Fix the sign in `compute_beta_derivatives`~~ ✅ Done.
3. ~~Fix block symmetrisation in `compute_beta_derivatives`~~ ✅ Done.
4. ~~Fix `update_beta`~~ ✅ Done.
5. ~~Fix operator precedence in `beta.size()` guards~~ ✅ Done.
6. ~~Clean up hygiene issues~~ ✅ Done.
7. **Add a unit test** that compares `compute_log_ylik` against a hand-rolled Python product-of-probs + log for a tiny dataset (e.g. 2 obs, 2 items, 2 classes).
8. **Add a unit test** for `update_beta`: starting from a known β, verify that the returned prior matrix is a valid stochastic matrix (rows sum to 1) and that β actually changed when the gradient is non-zero.

The mathematical core of `math_ops` is now solid enough to drive the EM loop in `em_engine.cpp`.
