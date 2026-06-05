import numpy as np

import pypolca as lca

df_carcinoma = lca.load_dataset("carcinoma")

df_carcinoma.head()
result = lca.fit("cbind(A, B, C, D, E, F, G) ~ 1", data=df_carcinoma, nclass=2)

print(result.aic)

print(np.round(result.probs[0], 3))
print(np.round(result.probs[1], 3))

print(np.round(result.posterior[0:5], 3))
df_carcinoma.select("A").unique()
# %%
## Derivate a posterior probability (common mistake is forgetting about overall prob)
obs = df_carcinoma.row(0)
ir = 1
prior_i = result.prior[0]
denom = []
num = 1.0 * prior_i[ir - 1]
R = result.prior.shape[1]

for j, p_cond in enumerate(result.probs):
    idx = obs[j] - 1
    p_cond_irj = p_cond[ir - 1][idx]
    num *= p_cond_irj

denom = 0.0

for r in range(R):
    p = 1.0
    for j, p_cond in enumerate(result.probs):
        idx = obs[j] - 1
        p_cond_irj = p_cond[r][idx]
        p *= p_cond_irj
    denom += p * prior_i[r]

print(f"{(num / denom):.3f}")
