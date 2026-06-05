---
tags: [implementation, initialization, probs]
---

# Random Initialization of Response Probabilities

> This is the expanded version with full derivations. For the concise version see [[RANDOM_INITIALIZATION]].

## Directives

### 1. Use Dirichlet(1, …, 1) per (class, item) block

When initializing `vecprobs`, draw each (class, item) probability vector from a
**Dirichlet(1, …, 1)** distribution. This is the only method that produces a
**uniform distribution over the probability simplex**.

Implement by drawing `K` independent `Gamma(1, 1)` (i.e. `Exp(1)`) variates and
normalizing them to sum to 1.0.

**Do not** normalize independent `Uniform(0, 1)` draws. That method is biased
toward the center of the simplex.

### 2. Respect the `vecprobs` layout

`vecprobs` is a flattened vector with layout:

```
[class 0, item 0, cat 0] ... [class 0, item 0, cat K0-1]
[class 0, item 1, cat 0] ... [class 0, item 1, cat K1-1]
...
[class R-1, last item, cat 0] ... [class R-1, last item, cat KJ-1]
```

Total length = `nclass * sum_j(num_choices[j])`.

For every `(r, j)` block of length `num_choices[j]`, the values must sum to `1.0`.

### 3. Make the seed configurable and use it

The initialization function must accept an `unsigned int seed` parameter and use
it to seed the random number generator (e.g. `std::mt19937`). This ensures
reproducible results across runs.

### 4. Implementation pattern

```cpp
static Eigen::VectorXd random_init_probs(const std::vector<int>& num_choices,
                                         int nclass,
                                         unsigned int seed) {
    int total = 0;
    for (int k : num_choices) total += k;

    Eigen::VectorXd vecprobs(nclass * total);
    std::mt19937 gen(seed);
    std::gamma_distribution<double> gamma(1.0, 1.0);

    for (int r = 0; r < nclass; ++r) {
        int pos = 0;
        for (size_t j = 0; j < num_choices.size(); ++j) {
            int K = num_choices[j];
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                double draw = gamma(gen);
                vecprobs(r * total + pos + k) = draw;
                sum += draw;
            }
            // Normalize so probabilities for this (class, item) sum to 1
            for (int k = 0; k < K; ++k) {
                vecprobs(r * total + pos + k) /= sum;
            }
            pos += K;
        }
    }
    return vecprobs;
}
```

---

## Mathematical Justification

### 1. Why Dirichlet(1,…,1) is uniform on the simplex

The probability simplex in \(K\) dimensions is

\[
\Delta_{K-1} = \left\{(p_1,\dots,p_K) \;\middle|\; p_k \ge 0,\; \sum_{k=1}^K p_k = 1\right\}.
\]

A **uniform** distribution on this set has constant density everywhere on
\(\Delta_{K-1}\). The Dirichlet distribution with parameter vector
\(\boldsymbol\alpha = (\alpha_1,\dots,\alpha_K)\) has density

\[
f(p_1,\dots,p_K) = \frac{1}{B(\boldsymbol\alpha)} \prod_{k=1}^K p_k^{\alpha_k - 1},
\qquad
B(\boldsymbol\alpha) = \frac{\prod_{k=1}^K \Gamma(\alpha_k)}{\Gamma\left(\sum_{k=1}^K \alpha_k\right)}.
\]

When \(\alpha_k = 1\) for all \(k\), the density becomes

\[
f(p_1,\dots,p_K) = \frac{1}{B(1,\dots,1)} \prod_{k=1}^K p_k^{0}
    = \frac{\Gamma(K)}{\Gamma(1)^K} \cdot 1
    = (K-1)!,
\]

which is constant on \(\Delta_{K-1}\). Therefore \(\operatorname{Dir}(1,\dots,1)\)
is exactly the uniform distribution on the simplex.

### 2. Proof that Gamma normalization produces Dir(1,…,1)

Let \(X_1,\dots,X_K \stackrel{\text{iid}}{\sim} \operatorname{Gamma}(1,1)\), i.e.
the standard exponential distribution with density \(f_X(x) = e^{-x}\) for
\(x > 0\). Define

\[
S = \sum_{j=1}^K X_j, \qquad P_k = \frac{X_k}{S}.
\]

We show that \((P_1,\dots,P_K)\) has the same distribution as
\(\operatorname{Dir}(1,\dots,1)\).

**Change of variables.**
We use the invertible map
\((X_1,\dots,X_K) \mapsto (S, P_1,\dots,P_{K-1})\) defined by

\[
X_k = S \cdot P_k \quad (k = 1,\dots,K-1), \qquad
X_K = S \cdot \left(1 - \sum_{k=1}^{K-1} P_k\right) = S \cdot P_K.
\]

The Jacobian matrix \(\partial(X_1,\dots,X_K) / \partial(S, P_1,\dots,P_{K-1})\)
has first column \((P_1,\dots,P_K)^T\) and for \(j = 1,\dots,K-1\) the
\((j+1)\)-th column has \(S\) in row \(j\), \(-S\) in row \(K\), and \(0\)
elsewhere. Its determinant is

\[
|J| = S^{K-1}.
\]

**Joint density of \((S, P)\).**
Since the \(X_k\) are independent with density \(e^{-x_k}\), the joint density is

\[
f_{X}(x_1,\dots,x_K) = \exp\!\left(-\sum_{k=1}^K x_k\right)
    = e^{-S}.
\]

By the change-of-variables formula,

\[
f_{S,P}(s, p_1,\dots,p_{K-1})
    = f_X(x_1,\dots,x_K) \cdot |J|
    = e^{-s} \cdot s^{K-1}.
\]

**Marginal density of \(P\).**
The joint density factors into a function of \(s\) alone and a constant function
of the \(p_k\):

\[
f_{S,P}(s, p) = \underbrace{s^{K-1} e^{-s}}_{\text{function of } s}
    \times \underbrace{1}_{\text{function of } p}.
\]

Integrating over \(s > 0\) yields the marginal density of \(P\):

\[
f_P(p) = \int_0^{\infty} s^{K-1} e^{-s} \, ds
    = \Gamma(K) = (K-1)!.
\]

This is constant on the simplex, identical to the density of
\(\operatorname{Dir}(1,\dots,1)\).

### 3. Proof that normalizing Uniform(0,1) is NOT uniform

Let \(U_1,\dots,U_K \stackrel{\text{iid}}{\sim} \operatorname{Uniform}(0,1)\)
with joint density \(f_U(u) = 1\) on \([0,1]^K\). Define

\[
S = \sum_{j=1}^K U_j, \qquad P_k = \frac{U_k}{S}.
\]

We derive the density of \(P = (P_1,\dots,P_K)\) on the simplex.

**Change of variables with box constraints.**
Use the same map as above: \(U_k = S \cdot P_k\) for \(k = 1,\dots,K\), with
\(|J| = S^{K-1}\). The constraint \((U_1,\dots,U_K) \in [0,1]^K\) translates to

\[
0 \le S \cdot P_k \le 1 \quad \text{for all } k
\quad\Longleftrightarrow\quad
0 \le S \le \frac{1}{\max_k P_k}.
\]

Because the uniform density is \(1\) on the cube and \(0\) outside, the joint
density of \((S, P)\) is

\[
f_{S,P}(s, p) = s^{K-1} \quad\text{for } 0 < s < \frac{1}{\max_k p_k},
\]
and \(0\) otherwise.

**Marginal density of \(P\).**
Integrate over the allowed range of \(s\):

\[
f_P(p) = \int_0^{1/\max_k p_k} s^{K-1} \, ds
    = \left[ \frac{s^K}{K} \right]_0^{1/\max_k p_k}
    = \frac{1}{K \cdot \bigl(\max_k p_k\bigr)^K}.
\]

**Interpretation.**
The density is **not** constant. It depends on \(\max_k p_k\):

- At the center \(p = (1/K,\dots,1/K)\):
  \[
  f_P(p) = \frac{K^{K-1}}{K} = K^{K-1}.
  \]

- At a corner, e.g. \(p = (1,0,\dots,0)\):
  \[
  f_P(p) = \frac{1}{K \cdot 1^K} = \frac{1}{K}.
  \]

The ratio of density at the center to a corner is \(K^K\). For \(K=2\) the
center is \(4\times\) more likely than the corners; for \(K=3\) it is
\(27\times\) more likely. The distribution is **strongly biased toward the
center** of the simplex.

**Explicit formula for K = 2.**
For a 2-category item with \(P_1 = p\) and \(P_2 = 1-p\), we have
\(\max(p, 1-p)\) and therefore

\[
f_P(p) = \frac{1}{2 \cdot \max(p, 1-p)^2}
    = \begin{cases}
        \dfrac{1}{2(1-p)^2}, & p \le \tfrac{1}{2}, \\[6pt]
        \dfrac{1}{2p^2}, & p > \tfrac{1}{2}.
      \end{cases}
\]

This peaks at \(p = \tfrac{1}{2}\) with value \(2\) and decreases to
\(\tfrac{1}{2}\) at the boundaries \(p = 0\) and \(p = 1\).

---

## References

1. **Devroye, L.** (1986). *Non-Uniform Random Variate Generation*. Springer,
   Chapter V, Section 3. Devroye derives the Dirichlet via Gamma normalization
   and notes that \(\operatorname{Dir}(1,\dots,1)\) is uniform on the simplex.
2. **Wikipedia**, "Dirichlet distribution", *Random variate generation* and
   *Special cases* sections.  
   <https://en.wikipedia.org/wiki/Dirichlet_distribution>
