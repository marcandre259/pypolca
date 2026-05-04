# Debugging `compute_ylik` in pypoLCA

## Bugs already visible in `src/cpp/core/math_ops.cpp`

### 1. Off-by-one / out-of-bounds indexing
```cpp
int current_choice_pos = 0;
// ...
current_choice_pos++;          // now = 1
int idx = r * sum_choices + current_choice_pos;
```
You increment *before* indexing, so the first element you read for each class is at offset `1`, not `0`. Worse, on the last iteration `current_choice_pos == sum_choices`, giving `idx = r*sum_choices + sum_choices`, which is **one past the end** of that class block (and exactly `vecprobs.size()` for the last class). Eigen doesn’t bounds-check `operator()` by default, so this silently reads garbage.

### 2. 1-based vs 0-based mismatch
`types.h` says `y` stores **1-based** responses (`0 = missing`), but your inner loop uses:
```cpp
if (obs_value == k)   // k is 0-based
```
For a response of `1` (first category), `k` starts at `0`, so this is never true. Every valid response is effectively ignored and `ylik` stays at its initialized value of `1`.

**Fix:**
```cpp
// Use k as 0-based index but compare to 1-based response
if (obs_value == k + 1) {
    ylik(i, r) *= prob_rjk;
}
```

---

## Practical debugging strategies

### 1. Python smoke-test with a hand-computed answer
Since `compute_ylik` is already exposed in `bindings.cpp`, the fastest feedback loop is a small Python script. Create a case so trivial you can compute the answer on paper.

```python
# debug_ylik.py
import numpy as np
from pypolca._core import Data, Params, compute_ylik

data = Data()
data.y = np.array([[1], [2]], dtype=np.int32)   # 1-based
data.x = np.ones((2, 1), dtype=np.float64)
data.num_choices = [2]

p = Params()
# 1 class, 2 categories: vecprobs layout = [class0_cat0, class0_cat1]
p.vecprobs = np.array([0.3, 0.7], dtype=np.float64)
p.beta = np.array([], dtype=np.float64)

lik = compute_ylik(data, p, 1)
print(lik)
# Expected:
# [[0.3],   # obs 1 -> category 1 -> prob 0.3
#  [0.7]]   # obs 2 -> category 2 -> prob 0.7
```

Run it with `uv run python debug_ylik.py`. If the output is all `1.0`s, you know the category matching logic is broken.

### 2. Standalone C++ executable (no Python needed)
The project already builds `examples/tryout.cpp`. Expand it to exercise `compute_ylik` directly and `std::cout` the internals. This is much easier to run under a native debugger (`lldb`/`gdb`) than the Python module.

```cpp
// examples/tryout.cpp
#include <iostream>
#include "pypolca/types.h"
#include "pypolca/math_ops.h"

int main() {
    pypolca::Data data;
    data.y = Eigen::MatrixXi(2, 1);
    data.y << 1, 2;
    data.num_choices = {2};

    pypolca::Params p;
    p.vecprobs = Eigen::VectorXd(2);
    p.vecprobs << 0.3, 0.7;

    auto ylik = pypolca::compute_ylik(data, p, 1);
    std::cout << "ylik:\n" << ylik << std::endl;
    return 0;
}
```

Build and run:
```bash
cmake -B build_debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build_debug --target tryout
./build_debug/tryout
```

### 3. Build a Debug version of the Python extension
Your `AGENTS.md` shows how to compile with debug symbols:
```bash
uv pip install -e . --no-build-isolation -Ccmake.define.CMAKE_BUILD_TYPE=Debug
```
This lets you step through the C++ code in a debugger while calling it from Python (e.g., attach `lldb -p <python_pid>` or use VS Code’s mixed-mode debugging).

### 4. Add temporary `std::cout` tracing
While iterating, add prints inside `compute_ylik` to see exactly which indices are being accessed:
```cpp
std::cout << "i=" << i << " r=" << r << " j=" << j
          << " k=" << k << " obs=" << obs_value
          << " idx=" << idx << " prob=" << prob_rjk << "\n";
```

### 5. Use AddressSanitizer (ASan)
Because of the out-of-bounds bug, building with ASan will make it crash loud and clear instead of returning garbage:
```bash
cmake -B build_san -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build_san --target tryout
./build_san/tryout
```

---

## 🎯 Recommended workflow

1. **Fix the two bugs above** (index off-by-one and 1-based response matching).
2. Run the hand-computed Python test — it should match your expected values exactly.
3. If something is still wrong, build the standalone `tryout` target in **Debug** with `std::cout` traces or run it under `lldb`.
4. Once the math checks out on trivial cases, run the existing pytest suite (`uv run pytest tests/python -v`) to make sure nothing else broke.
