import numpy as np
from pypolca._core import Data, Params, compute_ylik

data = Data()
data.y = np.array([[1], [2]], dtype=np.int32)  # 1-based
data.x = np.ones((2, 1), dtype=np.float64)
data.num_choices = [2]

p = Params()
# 1 class, 2 categories: vecprobs layout = [class0_cat0, class0_cat1]
p.vecprobs = np.array([0.3, 0.7], dtype=np.float64)
p.beta = np.array([], dtype=np.float64)

lik = compute_ylik(data, p, 1)
print(lik)
