---
date: 2026-05-11
tags: [review, cpp, em, bug]
---

# Review of `src/cpp/core/em_engine.cpp`

**File:** `src/cpp/core/em_engine.cpp`  
**Date:** 2026-05-11  
**Status:** 🔴 **Critical bugs present — do not use as-is.**

> See also: [[EM_LOOP_LIKELIHOOD_DROP_INVESTIGATION]], [[MATH_OPS_REVIEW]], [[OPTIMIZATION_ALGORITHM]]

---

## 1. Critical Bugs

### 1.1 Loop condition is inverted — EM never runs
```cpp
double dll = std::numeric_limits<double>::infinity();
// ...
while (std::abs(dll) < tol) {   // ❌ abs(inf) < 1e-10  → false
```
Because `dll` is initialized to `infinity`, the condition is **false on the first check** and the while-body never executes. The function immediately returns an uninitialized result.

**Fix:** Continue iterating while the change is *larger* than tolerance.
```cpp
while (std::abs(dll) >= tol && n_iter < maxiter) {
```
(Or, better, use a `for` loop with an explicit break on convergence.)

---

### 1.2 Variable shadowing breaks `dll` computation
```cpp
double log_lik_latest = std::numeric_limits<double>::infinity();  // outer

while (...) {
    double log_lik_prev = log_lik_latest;   // reads outer (inf)
    // ...
    double log_lik_latest = 0.0;            // ❌ inner shadow
    // ...
    dll = log_lik_latest - log_lik_prev;    // uses inner - outer = -inf
}
```
Even if the loop condition were fixed, `log_lik_prev` would still read the **outer** `log_lik_latest` (still `inf`), so `dll` becomes `-inf` and the loop exits after one iteration.

**Fix:** Remove the inner declaration.
```cpp
log_lik_latest = 0.0;   // assign, do not re-declare
```

---

### 1.3 Log-likelihood computation is statistically wrong
```cpp
Eigen::MatrixXd log_lik_mat = compute_log_ylik(data, p, nclass);

for (int i = 0; i < N; i++) {
    for (int r = 0; r < nclass; r++) {
        log_lik_latest += log_lik_mat(i, r);   // ❌ plain sum
    }
}
```
`compute_log_ylik` returns the **class-conditional** log-likelihoods `log P(y_i | c=r)`.  
The observed-data log-likelihood is the *marginal* over classes weighted by priors:

```
log L = Σ_i log( Σ_r prior[i,r] · exp(log_ylik[i,r]) )
```

Summing the raw conditional log-values has no statistical meaning, so:
* The convergence criterion is meaningless.
* `res.loglik` is wrong.
* The `dll < -1e-7` error trap may fire spuriously or miss real problems.

**Fix:** Compute the marginal log-likelihood with a numerically stable log-sum-exp:
```cpp
double log_lik_latest = 0.0;
for (int i = 0; i < N; ++i) {
    double max_ll = log_lik_mat.row(i).maxCoeff();
    double sum = 0.0;
    for (int r = 0; r < nclass; ++r) {
        sum += prior(i, r) * std::exp(log_lik_mat(i, r) - max_ll);
    }
    log_lik_latest += std::log(sum) + max_ll;
}
```
(Consider using the existing `compute_logsumexp` utility if it can accept weighted vectors.)

---

### 1.4 No-covariate (`S==1`) prior update computes wrong dimensions
```cpp
Eigen::VectorXd col_means(S);          // size 1
for (int s = 0; s < S; s++) {
    col_means(s) = posterior.col(s).mean();   // only column 0
}
prior = col_means.transpose().replicate(N, 1);   // N x 1, not N x nclass
```
When there are no covariates, the M-step for class proportions is simply the column-means of the **posterior** matrix over all `nclass` classes. The current code creates an `N × 1` matrix because `S == 1`, but `prior` must remain `N × nclass`.

**Fix:**
```cpp
Eigen::VectorXd col_means(nclass);
for (int r = 0; r < nclass; ++r) {
    col_means(r) = posterior.col(r).mean();
}
prior = col_means.transpose().replicate(N, 1);
```

---

## 2. API / Build Issues

### 2.1 Parameter mismatch between header and implementation
The `.cpp` declares an extra `unsigned int seed` parameter that does **not** appear in `include/pypolca/em_engine.h`:

```cpp
// em_engine.h
Results fit_em(..., const Eigen::VectorXd& beta_start = Eigen::VectorXd());

// em_engine.cpp
Results fit_em(..., const Eigen::VectorXd& beta_start,
               unsigned int seed)   // <- missing from header
```
This will cause a compilation error or linker mismatch once the header is included by another translation unit.

**Fix:** Add `unsigned int seed = 0` (or similar) to the header declaration.

---

### 2.2 Internal Eigen include path
```cpp
#include "Eigen/src/Core/Matrix.h"   // ❌ implementation detail
```
This path is not guaranteed to exist across Eigen versions or installations. Use the public header:
```cpp
#include <Eigen/Core>
```
(Or rely on the transitive includes from `pypolca/types.h` which already pulls `<Eigen/Dense>`.)

---

## 3. Minor Issues

| Issue | Location | Impact | Recommendation |
|---|---|---|---|
| `verbose` is unused | parameter list | No progress logging | Add basic `std::cout` prints or remove the parameter until implemented. |
| `int total` overflow risk | `random_init_probs` | Unlikely in practice, but brittle | Use `Eigen::Index` or `std::ptrdiff_t` for counts that feed into Eigen sizes. |
| Loop iterates once after `maxiter` hit | end of `while` body | One extra iteration possible | Prefer `for (int iter = 0; iter < maxiter; ++iter)` with an early `break` on convergence. |
| `col_means` name is confusing | `S==1` branch | Misleading variable name | Rename to `class_props` or `posterior_means`. |
| Comments out of sync | lines 67–68 | Comment refers to checking `dll`, but code does not match | Update or remove stale comments. |

---

## 4. Suggested Refactored Loop Skeleton

```cpp
double log_lik_prev = -std::numeric_limits<double>::infinity();
double log_lik_latest = -std::numeric_limits<double>::infinity();
int n_iter = 0;
bool converged = false;

for (; n_iter < maxiter; ++n_iter) {
    log_lik_prev = log_lik_latest;

    posterior = e_step(data, p, prior, nclass);
    p.vecprobs  = m_step_probs(data, posterior, data.num_choices, nclass);

    if (S > 1) {
        auto [new_beta, new_prior] = update_beta(data, posterior, prior, p.beta, nclass);
        p.beta = std::move(new_beta);
        prior  = std::move(new_prior);
    } else {
        Eigen::VectorXd class_props(nclass);
        for (int r = 0; r < nclass; ++r) {
            class_props(r) = posterior.col(r).mean();
        }
        prior = class_props.transpose().replicate(N, 1);
    }

    // --- Compute observed-data log-likelihood ---
    Eigen::MatrixXd log_ylik = compute_log_ylik(data, p, nclass);
    log_lik_latest = 0.0;
    for (int i = 0; i < N; ++i) {
        double max_ll = log_ylik.row(i).maxCoeff();
        double sum = 0.0;
        for (int r = 0; r < nclass; ++r) {
            sum += prior(i, r) * std::exp(log_ylik(i, r) - max_ll);
        }
        log_lik_latest += std::log(sum) + max_ll;
    }

    double dll = log_lik_latest - log_lik_prev;

    if (std::abs(dll) < tol) {
        converged = true;
        break;
    }
    if (S > 1 && dll < -1e-7) {
        converged = false;
        break;   // likelihood decreased — reject / restart
    }
}
```

---

## 5. Summary

| Severity | Count | Categories |
|---|---|---|
| 🔴 Critical | 4 | Loop condition, shadowing, wrong log-likelihood, wrong prior dim |
| 🟡 Moderate | 2 | Header mismatch, internal Eigen include |
| 🟢 Minor | 5 | Unused param, naming, loop style, comments, overflow risk |

**Bottom line:** The four critical bugs (§1.1–1.4) mean the current file cannot execute a single valid EM iteration. Fix the loop condition, remove the variable shadow, correct the log-likelihood formula, and fix the no-covariate prior update before running any tests.
