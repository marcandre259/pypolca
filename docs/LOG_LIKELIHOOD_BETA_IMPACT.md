---
date: 2026-05-08
tags: [math, numerics, beta]
---

# Impact of Log-Likelihood Conversion on Beta Gradients/Hessians

**Date:** 2026-05-08  
**Status:** Implemented  
**Conclusion:** No impact on `compute_beta_derivatives` or `update_beta`. Log-likelihood conversion applied; two unrelated beta bugs fixed.

---

## 1. Executive Summary

Converting `compute_ylik` and `e_step` from raw likelihoods (product space) to log-likelihoods (sum space) is the correct fix for numerical underflow/overflow. **This change requires zero modifications to the beta gradient and Hessian code.**

The beta Newton-Raphson functions are fully decoupled from how the posterior is computed. They consume `posterior` and `prior` matrices, which remain probabilities in `[0, 1]` regardless of whether the E-step uses raw likelihoods or the log-sum-exp trick.

Two pre-existing bugs in `update_beta` and `compute_beta_derivatives` were discovered during this investigation. These should be fixed independently of the log-likelihood conversion.

---

## 2. Data Flow Diagram

```
compute_ylik ──► e_step ──► posterior ──► compute_beta_derivatives / update_beta
    │
    │  (log conversion here only)
    ▼
log_ylik ──────► log-sum-exp ──► posterior (identical output)
```

The beta code sits downstream of the E-step. It sees only `posterior` and `prior`.

---

## 3. Beta Gradient/Hessian Formulas

Source: `src/cpp/core/math_ops.cpp` (lines 127-148)

### Gradient
```cpp
gradients(row) += x(i, j) * (posterior(i, r) - prior(i, r));
```

### Hessian diagonal
```cpp
hessians(row, col) += x(i, j) * x(i, k) *
                      (prior(i, r) * (1 - prior(i, r)) -
                       posterior(i, r) * (1 - posterior(i, r)));
```

### Hessian off-diagonal
```cpp
hessians(row, col2) +=
    x(i, j) * x(i, k) *
    (posterior(i, r) * posterior(i, s) - prior(i, r) * prior(i, s));
```

**Dependencies:** `posterior(i, r)`, `prior(i, r)`, `x(i, j)`  
**No dependency on `ylik`**

---

## 4. Why the E-Step Output Is Identical

Mathematically:

```
posterior(i, r) = prior(i, r) * ylik(i, r) / sum_s(prior(i, s) * ylik(i, s))
```

Taking logs:
```
posterior(i, r) = exp[ log(prior(i, r)) + log(ylik(i, r)) - log_sum_exp(...) ]
```

The resulting `posterior` matrix is **bitwise-identical** (up to floating-point rounding) whether computed in raw space or log space. Since the beta code only consumes `posterior`, it cannot tell the difference.

---

## 5. Implementation Steps

### Step 1: Add a stable `log_sum_exp` helper

```cpp
static double log_sum_exp(const Eigen::VectorXd &x) {
  double max_x = x.maxCoeff();
  double sum = 0.0;
  for (int i = 0; i < x.size(); ++i) {
    sum += std::exp(x(i) - max_x);
  }
  return max_x + std::log(sum);
}
```

### Step 2: Rename `compute_ylik` → `compute_log_ylik`

Change the accumulation from products to sums of logs:

```cpp
// Before:
Eigen::MatrixXd ylik(N, R);
ylik.setOnes();
// ...
ylik(i, r) *= prob_rjk;

// After:
Eigen::MatrixXd log_ylik(N, R);
log_ylik.setZero();
// ...
log_ylik(i, r) += std::log(prob_rjk);
```

Return type stays `Eigen::MatrixXd`, but values are now log-probabilities (negative or zero).

### Step 3: Rewrite `e_step` to use log-sum-exp

```cpp
Eigen::MatrixXd log_ylik = compute_log_ylik(data, p, nclass);
Eigen::MatrixXd posterior(N, nclass);

for (int i = 0; i < N; i++) {
  Eigen::VectorXd log_num = prior.row(i).array().log() + log_ylik.row(i).array();
  double log_denom = log_sum_exp(log_num);
  for (int r = 0; r < R; r++) {
    posterior(i, r) = std::exp(log_num(r) - log_denom);
  }
}
```

The output `posterior` is numerically identical to the product-space version, so `compute_beta_derivatives` and `update_beta` require **no changes**.

### Step 4: Add `compute_loglik` for convergence monitoring

```cpp
double compute_loglik(const Data &data, const Params &p,
                      const Eigen::MatrixXd &prior, int nclass) {
  int N = data.n_obs();
  Eigen::MatrixXd log_ylik = compute_log_ylik(data, p, nclass);
  double loglik = 0.0;
  for (int i = 0; i < N; i++) {
    Eigen::VectorXd log_joint = prior.row(i).array().log() + log_ylik.row(i).array();
    loglik += log_sum_exp(log_joint);
  }
  return loglik;
}
```

### Step 5: Update header and bindings

- In `math_ops.h`: rename `compute_ylik` → `compute_log_ylik`, add `compute_loglik`.
- In `bindings.cpp`: expose `compute_log_ylik` and `compute_loglik` to Python.

### What does NOT change

| Function | Why no change |
|----------|---------------|
| `compute_beta_derivatives` | Consumes only `posterior` and `prior`, both still probabilities |
| `update_beta` | Same reason; Newton step formula unchanged |
| `compute_prior_from_beta` | Already uses `max_eta` softmax stabilization |

---

## 6. Known Beta Bugs to Fix Separately

These are **not** caused by the log-likelihood conversion, but will break Newton-Raphson regardless.

### Bug 6a: Double-negation sign error in `update_beta`

`hessians` already accumulates the *negative* Hessian (observed information). Do **not** negate it again:

```cpp
// Wrong:
Eigen::MatrixXd obs_information = -1.0 * hessians;

// Correct:
Eigen::MatrixXd obs_information = hessians;  // hessians is already -∂²ℓ
```

### Bug 6b: Wrong initialization type for `hessians`

```cpp
// Wrong:
Eigen::MatrixXd hessians = Eigen::VectorXd::Zero(rank * rank);

// Correct:
Eigen::MatrixXd hessians = Eigen::MatrixXd::Zero(rank, rank);
```

---

## 7. Verification

Build and tests pass after the changes:

```bash
./rebuild.sh
uv run pytest tests/python -v
```

---

## 8. References

- `src/cpp/core/math_ops.cpp` — gradient/Hessian implementation
- `src/cpp/include/pypolca/math_ops.h` — function signatures
- `docs/LOG_SUM_EXP_AND_LIKELIHOOD_OVERFLOW.md` — existing log-sum-exp documentation
