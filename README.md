# pypoLCA

[![CI](https://github.com/marcandre259/pypolca/actions/workflows/ci.yml/badge.svg)](https://github.com/marcandre259/pypolca/actions/workflows/ci.yml)
[![Docs](https://github.com/marcandre259/pypolca/actions/workflows/docs.yml/badge.svg)](https://marcandre259.github.io/pypolca/)
[![codecov](https://codecov.io/gh/marcandre259/pypolca/branch/main/graph/badge.svg)](https://codecov.io/gh/marcandre259/pypolca)

<p align="center">
  <img src="assets/pypolca-logo.png" alt="pypoLCA logo" width="400">
</p>

Polytomous variable latent class analysis (LCA) for Python, powered by a C++17 backend. `pypoLCA` is a translation of R's [poLCA](https://github.com/dlinzer/poLCA) package by Drew Linzer and Jeffrey Lewis.

## What is latent class analysis?

LCA is a statistical method that discovers latent (unobserved) categorical variables from a set of nominal responses. The core assumption is that observed responses are mutually independent conditionally on the latent variable — all dependencies between responses flow through the latent structure.

The model identifies two things:

1. The underlying latent classes (e.g., "high-risk" vs. "low-risk" respondents), and
2. The conditional probabilities of each observed response given each class.

LCA's latent variables are categorical (e.g., *class 1* = "non-cheaters", *class 2* = "chronic cheaters"). This makes LCA the categorical-data analogue of Gaussian Mixture Models (GMM): GMM assumes normally-distributed responses, LCA assumes multinomial responses. Both fit parameters via Expectation-Maximisation (EM). A good tutorial extending EM beyond standard GMM applications is [Gao (2022)](https://teng-gao.github.io/blog/2022/ems/).

### Applications

- **Diagnostic agreement** — LCA estimates rater accuracy without a gold-standard reference. Applied to carcinoma diagnoses by seven pathologists (Uebersax & Grove, 1990; dataset: `carcinoma`).
- **Political typology** — LCA identifies voter segments from candidate trait ratings. Applied to 2000 ANES survey data (dataset: `election`).
- **Academic dishonesty** — Latent classes of cheating behavior among students, regressed on GPA covariates (dataset: `cheating`).
- **Survey attitude clustering** — Uncovering latent opinion groups from social survey responses (McCutcheon, 1987; dataset: `gss82`).

### Latent class regression

LCA can be extended with covariates that predict class membership. `pypoLCA` fits latent class regression using the same hybrid EM / Newton-Raphson algorithm as R's `poLCA`. The EM loop alternates expected-posterior and maximisation steps. Response probabilities have a closed-form M-step, which guarantees the standard EM ascent property. Covariate coefficients lack a closed form and are updated within each M-step via Newton-Raphson (NR). Unlike pure EM, the NR step can overshoot and cause a likelihood drop. The implementation detects this and restarts with perturbed starting values (`max_restarts`). In any case, the algorithm finds only a local maximum, so multiple random starts (`nrep`) are recommended.

Standard errors are provided for all parameter estimates (i.e. conditional response probabilities, prior class probabilities, and (when covariates are present) regression coefficients). SEs are computed from the observed information matrix via the outer product of the individual score contributions, then transformed to probability space via the delta method. This matches the approach used by R's `poLCA`.

## Install

```bash
pip install polca
```

## Quick start

```python
import pypolca as lca

# Load a built-in dataset (a Polars DataFrame)
df = lca.load_dataset("carcinoma")

# Fit a 2-class model — seven pathologists rating 118 slides
result = lca.fit("cbind(A, B, C, D, E, F, G) ~ 1", data=df, nclass=2, nrep=5)

# Inspect results
print(f"Log-likelihood: {result.loglik:.2f}")
print(f"AIC: {result.aic:.2f}")
print(f"Iterations: {result.iterations}")

# Class-conditional probabilities for the first item
print(result.probs[0])       # shape (nclass, n_categories)

# Posterior class membership (N × R)
print(result.posterior[:5])

# Predicted class for each observation (1-based)
print(result.predclass[:5])    # 1-based (matching R poLCA convention)

# Standard errors
print(result.probs_se[0])    # SEs for first item
print(result.P_se)           # SEs for class priors
```

**With covariates (latent class regression):**

```python
df = lca.load_dataset("cheating")

# Cheating behaviours ~ GPA
result = lca.fit(
    "cbind(LIEEXAM, LIEPAPER, FRAUD, COPYEXAM) ~ GPA",
    data=df,
    nclass=2,
    nrep=10,
)

print(result.coeff)          # Regression coefficients (covariates × (classes − 1))
print(result.coeff_se)       # Standard errors
print(result.P)              # Population class shares
```

The formula syntax uses `cbind(var1, var2, ...)` on the left-hand side, or equivalently Python-style `var1 + var2 + ...`. The right-hand side is `~ 1` for intercept-only or `~ cov1 + cov2` for latent class regression.

## Backend

The EM engine and standard error computation are written in C++17 (Eigen for linear algebra), exposed to Python via pybind11. The build system is CMake + scikit-build-core, managed by `uv`. Incremental C++ rebuilds take ~1–2 s with `./rebuild.sh`.

## Benchmarks

### Intercept-only model (synthetic binary data)

Comparison of `pypolca` (C++, with/without SE) vs R's `poLCA` on synthetic 5-item binary data at increasing N, 2 classes. Timings are means over 10 runs (5 runs for N=10,000).

| N      | Items | Classes | R poLCA  | pypolca (with SE) | Speed-up | pypolca (no SE) | Speed-up |
|--------|-------|---------|----------|-------------------|----------|-----------------|----------|
| 319    | 4     | 2       | 0.025 s  | 0.012 s           | 2.0×     | 0.010 s         | 2.5×     |
| 500    | 5     | 2       | 0.013 s  | 0.006 s           | 2.2×     | 0.006 s         | 2.2×     |
| 2,000  | 5     | 2       | 0.041 s  | 0.021 s           | 1.9×     | 0.021 s         | 1.9×     |
| 10,000 | 5     | 2       | 0.193 s  | 0.105 s           | 1.8×     | 0.103 s         | 1.9×     |

### Covariate model (`election` dataset)

Full latent class regression: 12 polytomous items (4-point scale), 5 covariates, 3 classes, N=880 (complete cases). Timings are means over 20 runs.

| Model                        | R poLCA  | pypolca (with SE) | Speed-up | pypolca (no SE) | Speed-up |
|------------------------------|----------|-------------------|----------|-----------------|----------|
| `~ VOTE3 + AGE + EDUC + GENDER + PARTY` | 0.085 s  | 0.058 s           | 1.4×     | 0.055 s         | 1.5×     |

## Datasets

| Dataset    | N     | Manifest items                                       | Covariates             | Source                    |
|------------|-------|------------------------------------------------------|------------------------|---------------------------|
| carcinoma  | 118   | A–G (7 binary: no carcinoma / carcinoma)             | —                      | Agresti (2002)            |
| cheating   | 319   | LIEEXAM, LIEPAPER, FRAUD, COPYEXAM (4 binary)        | GPA                    | R poLCA                   |
| election   | 1,785 | MORALG–INTELB (12 ordinal: 4-point trait ratings)    | VOTE3, AGE, EDUC, etc. | 2000 ANES                 |
| gss82      | 1,202 | PURPOSE, ACCURACY, UNDERSTA, COOPERAT (2–3 categories)| —                      | McCutcheon (1987)         |
| values     | 216   | A–D (4 binary: universalistic / particularistic)     | —                      | R poLCA                   |

## Credits

pypoLCA is a translation of Drew A. Linzer and Jeffrey B. Lewis's R package:

> Linzer, D. A., & Lewis, J. B. (2011). poLCA: An R Package for Polytomous Variable Latent Class Analysis. *Journal of Statistical Software*, 42(10), 1–29. [doi:10.18637/jss.v042.i10](https://doi.org/10.18637/jss.v042.i10)

Built-in datasets and the EM / Newton-Raphson algorithm are taken from the `poLCA` R package, licensed GPL-2.0-or-later.

C++ bindings powered by [pybind11](https://github.com/pybind/pybind11). Linear algebra via [Eigen](https://eigen.tuxfamily.org/).

## Contributing

Contributions are welcome and appreciated. Please keep submissions tight and purposeful. The goal is to keep `pypoLCA` a focused, maintainable package. AI-assisted contributions are fine, but AI use doesn't excuse sloppy or verbose work. Review your output before submitting a PR.

## License

GPL-2.0-or-later
