---
tags: [implementation, initialization, probs]
---

# Random Initialization of Response Probabilities

> Concise version. For the full derivation see [[RANDOM_INIT_PROBABILITIES]].

## Directive

When initializing class-conditional response probabilities (`vecprobs`), draw each
(class, item) block from a **Dirichlet(1, …, 1)** distribution.

Implement this by generating `K` independent `Gamma(1, 1)` (i.e. `Exp(1)`) variates
and normalizing them so the block sums to 1.0. Use a configurable `seed` for
reproducibility.

**Do not** normalize independent `Uniform(0, 1)` draws. That does not produce a
uniform distribution on the probability simplex and introduces a bias toward the
center of the simplex.

## Mathematical Justification

### The Dirichlet(1,…,1) is uniform on the simplex

The probability simplex in \(K\) dimensions is

\[
\Delta_{K-1} = \{(p_1,\dots,p_K) \mid p_k \ge 0,\; \sum_k p_k = 1\}.
\]

A **uniform** distribution on this set has constant density everywhere on
\(\Delta_{K-1}\). The Dirichlet distribution with parameter vector
\(\boldsymbol\alpha = (1,\dots,1)\) has density

\[
f(p_1,\dots,p_K) \propto \prod_{k=1}^K p_k^{\alpha_k - 1}
    = \prod_{k=1}^K p_k^{0}
    = \text{constant},
\]

so \(\operatorname{Dir}(1,\dots,1)\) is exactly the uniform distribution on the
simplex.

### Generation via Gamma variates

If \(X_1,\dots,X_K \stackrel{\text{iid}}{\sim} \operatorname{Gamma}(1,1)\) (i.e.
\(\operatorname{Exp}(1)\)), then

\[
P_k = \frac{X_k}{\sum_{j=1}^K X_j}
\]

is distributed as \(\operatorname{Dir}(1,\dots,1)\).

*Proof sketch:* The joint density of the \(X_k\) is \(\exp(-\sum x_k)\).
Changing variables to the total \(S = \sum X_k\) and proportions
\(P_k = X_k / S\) yields a Jacobian factor \(S^{K-1}\). The joint density factors
into a term depending only on \(S\) and a constant term for the proportions,
proving the proportions are independent of \(S\) and uniform on the simplex.

### Why normalizing Uniform(0,1) is wrong

Drawing \(U_k \stackrel{\text{iid}}{\sim} \operatorname{Uniform}(0,1)\) and setting
\(P_k = U_k / \sum U_j\) does **not** yield a uniform distribution on the simplex.
Geometrically, this intersects the hypercube \([0,1]^K\) with the plane
\(\sum u_k = S\) and projects the result. The density at a point \(p\) is
proportional to the \((K-1)\)-dimensional volume of that slice, which varies with
\(p\).

For the 2-category case one can derive explicitly:

\[
f_P(p) \propto \min\!\left(\frac{1}{p}, \frac{1}{1-p}\right)^2,
\]

which peaks at \(p = 0.5\) and tapers toward the boundaries. For larger \(K\) the
same bias appears: the method systematically oversamples the center of the
simplex and undersamples the corners (where one category has probability near 1).

## References

1. **Devroye, L.** (1986). *Non-Uniform Random Variate Generation*. Springer,
   Chapter V, Section 3 (The Dirichlet distribution). Devroye states explicitly
   that normalizing i.i.d. Gamma variates yields a Dirichlet, and that
   \(\operatorname{Dir}(1,\dots,1)\) is uniform on the simplex.
2. **Wikipedia**, “Dirichlet distribution”, *Random variate generation* and
   *Special cases* sections.
   <https://en.wikipedia.org/wiki/Dirichlet_distribution>
