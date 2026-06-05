---
tags: [cpp, debug, build-system]
---

# C++ Development & Debug Guide

This project uses **scikit-build-core** + **CMake** + **pybind11** to compile C++17 code and expose it as a Python extension module (`pypolca._core`).

## Quick Start — Compile & Test

### 1. Editable install (Python entry point)

```bash
uv pip install -e ".[dev]" --no-build-isolation \
    -Ccmake.define.CMAKE_BUILD_TYPE=Debug
```

This builds the C++ extension in-place and links it into the `pypolca/` package. Rebuilds are triggered automatically when C++ sources change and you next `import pypolca`.

### 2. Run Python tests

```bash
# Full test suite
uv run pytest tests/python -v

# Single test
uv run pytest tests/python/test_basic.py::TestFitEM::test_em_runs -v

# Smoke test via example
uv run python examples/basic_fit.py
```

### 3. Interactive sanity check

```python
python -c "
from pypolca._core import Data, Params, compute_ylik
import numpy as np

d = Data()
d.y = np.array([[1,1],[2,2]], dtype=np.int32)
d.num_choices = [2, 2]
p = Params()
p.vecprobs = np.ones(8) * 0.5
print(compute_ylik(d, p, 2))
"
```

---

## “I just want to compile a C++ file and print something”

If you don’t care about Python and just want to run a standalone `.cpp` against the core library:

1. **Create `src/cpp/quick.cpp`** with a `main()`:

```cpp
#include <iostream>
#include "pypolca/types.h"

int main() {
    std::cout << "hello from c++\n";
    return 0;
}
```

2. **Add two lines to `src/cpp/CMakeLists.txt`** (do this once):

```cmake
add_executable(quick quick.cpp)
target_link_libraries(quick PRIVATE pypolca_core)
```

3. **Build and run**:

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --target quick
./build/debug/src/cpp/quick
```

After the first configure, just edit `quick.cpp` and rerun the `cmake --build ... --target quick` line.

---

## Manual CMake Build (for pure C++ debugging)

If you want to step through the C++ math without Python overhead, build a standalone debug executable.

### Add a debug target

In `src/cpp/CMakeLists.txt`, add:

```cmake
add_executable(test_cpp_debug debug_main.cpp)
target_link_libraries(test_cpp_debug PRIVATE pypolca_core)
```

Then create `src/cpp/debug_main.cpp`:

```cpp
#include <iostream>
#include "pypolca/types.h"
#include "pypolca/em_engine.h"

int main() {
    pypolca::Data data;
    data.y = Eigen::MatrixXi(2, 2);
    data.y << 1, 1,
              2, 2;
    data.num_choices = {2, 2};

    pypolca::Params p;
    p.vecprobs = Eigen::VectorXd::Constant(8, 0.5);

    auto ylik = pypolca::compute_ylik(data, p, 2);
    std::cout << "ylik:\n" << ylik << "\n";
    return 0;
}
```

### Build & run

```bash
cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build/debug --target test_cpp_debug
./build/debug/src/cpp/test_cpp_debug
```

### Debug with LLDB / GDB

```bash
lldb ./build/debug/src/cpp/test_cpp_debug
(lldb) breakpoint set --name compute_ylik
(lldb) run
```

Or attach to the Python process when the crash happens through the bindings:

```bash
lldb -- python -c "from pypolca._core import fit_em; ..."
(lldb) breakpoint set --name fit_em
(lldb) run
```

---

## Daily Workflow Cheat Sheet

| Task | Command |
|------|---------|
| **Install / reinstall** | `uv pip install -e ".[dev]" --no-build-isolation -Ccmake.define.CMAKE_BUILD_TYPE=Debug` |
| **Run Python tests** | `uv run pytest tests/python -v` |
| **Run example** | `uv run python examples/basic_fit.py` |
| **Build C++ debug exe** | `cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug && cmake --build build/debug --target test_cpp_debug` |
| **Debug C++ only** | `lldb ./build/debug/src/cpp/test_cpp_debug` |
| **Debug through Python** | `lldb -- python examples/basic_fit.py` |

---

## Project Layout

| Path | Purpose |
|------|---------|
| `src/cpp/core/` | Core EM algorithm and math operations |
| `src/cpp/include/pypolca/` | Public C++ headers (`types.h`, `math_ops.h`, `em_engine.h`) |
| `src/cpp/bindings.cpp` | pybind11 wrapper exposing `pypolca._core` |
| `pypolca/` | Pure Python package (`__init__.py`, `api.py`, `utils.py`) |
| `tests/python/` | pytest suite |
| `CMakeLists.txt` | Top-level CMake configuration |
| `pyproject.toml` | Python packaging (scikit-build-core backend) |
