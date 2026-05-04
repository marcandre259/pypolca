# pypoLCA C++ & Python Bindings — A Beginner's Tutorial

> **Goal:** By the end of this document you will understand (1) how the C++ code in this project gets compiled, and (2) how that C++ code talks to Python. No prior knowledge of CMake, pybind11, or build systems is assumed.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Part 1: Compiling the C++](#part-1-compiling-the-c)
   - [What is CMake and why do we need it?](#what-is-cmake-and-why-do-we-need-it)
   - [The three-layer build](#the-three-layer-build)
   - [Step-by-step: what happens when you install](#step-by-step-what-happens-when-you-install)
   - [Where did my compiled files go?](#where-did-my-compiled-files-go)
   - [Compiling `tryout` by hand](#compiling-tryout-by-hand)
   - [The fast rebuild loop](#the-fast-rebuild-loop)
3. [Part 2: How the Python Bindings Work](#part-2-how-the-python-bindings-work)
   - [What are bindings?](#what-are-bindings)
   - [pybind11 in 30 seconds](#pybind11-in-30-seconds)
   - [Reading `bindings.cpp` line by line](#reading-bindingscpp-line-by-line)
   - [How numpy arrays become Eigen matrices](#how-numpy-arrays-become-eigen-matrices)
   - [The Python wrapper (`pypolca/api.py`)](#the-python-wrapper-pypolcaapipy)
   - [The full data journey](#the-full-data-journey)
4. [Cheat Sheet](#cheat-sheet)

---

## The Big Picture

This project is a **hybrid**: the heavy math lives in C++ (for speed), but the user-friendly API lives in Python (for convenience).  Think of the C++ code as the engine of a car and the Python code as the steering wheel and dashboard.

```
┌─────────────────────────────────────────────────────────────┐
│  Python world                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │  your script │──▶│ pypolca.api │──▶│ pypolca._core   │  │
│  └─────────────┘   └─────────────┘   │  (Python module)│  │
│                                      └────────┬────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────┐
│  C++ world                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐  │
│  │  Data struct │   │  EM engine  │   │  Math ops       │  │
│  │  (types.h)   │   │  (em_engine)│   │  (math_ops.cpp) │  │
│  └─────────────┘   └─────────────┘   └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Your Python script calls `pypolca.fit(...)`.  That function builds a C++ `Data` object, passes it to the C++ `fit_em(...)` function, and converts the C++ `Results` back into a Python object you can inspect.

---

## Part 1: Compiling the C++

### What is CMake and why do we need it?

C++ code must be **compiled** before it can run.  Unlike Python, you cannot just `import` a `.cpp` file.

Compiling by hand looks like this:

```bash
c++ my_file.cpp -o my_program
```

That works for one file, but this project has:
- Multiple `.cpp` files
- External dependencies (Eigen for matrix math, pybind11 for Python bindings)
- Different build modes (Debug vs. Release)
- Different platforms (macOS, Linux, Windows)

**CMake** is a "meta-build-system."  You write a human-readable recipe (`CMakeLists.txt`) that describes *what* you want to build, and CMake generates the actual build files for your platform.  On macOS/Linux it generates files for **Ninja** (a fast build tool).

### The three-layer build

This project actually has *three* build systems stacked together:

| Layer | Tool | What it does |
|-------|------|-------------|
| 1 | **CMake** | Reads `CMakeLists.txt` and generates build instructions. |
| 2 | **Ninja** | Reads CMake's output and actually compiles the C++ files. |
| 3 | **scikit-build-core** | Tells Python's installer (`uv pip install`) how to trigger CMake+Ninja automatically. |

You almost never talk to CMake or Ninja directly.  You just run:

```bash
uv pip install -e . --no-build-isolation
```

and the three layers do their dance.

### Step-by-step: what happens when you install

Let's trace exactly what happens when you run the install command.

**Step 1 — `uv` reads `pyproject.toml`**

This file tells `uv`:
- "This is a Python package named `pypolca`"
- "Use `scikit-build-core` as the build backend"
- "Pass `-GNinja` to CMake"
- "Put build artifacts in `build/{wheel_tag}`"

**Step 2 — `scikit-build-core` runs CMake**

CMake reads the top-level `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(pypolca LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Download Eigen3 (matrix library) and pybind11
FetchContent_Declare(Eigen ...)
FetchContent_Declare(pybind11 ...)
FetchContent_MakeAvailable(Eigen pybind11)

# Build everything in src/cpp/
add_subdirectory(src/cpp)

# Also build the tryout executable
add_executable(tryout ${CMAKE_SOURCE_DIR}/examples/tryout.cpp)
target_link_libraries(tryout PRIVATE pypolca_core)
```

CMake downloads Eigen and pybind11 into `build/_deps/`, then creates a `build.ninja` file.

**Step 3 — Ninja compiles everything**

Ninja reads `build.ninja` and performs the actual compilation.  It produces:

| Output | Description |
|--------|-------------|
| `_core*.so` | The Python extension module (a shared library). |
| `libpypolca_core.a` | The static library containing the core math code. |
| `tryout` | A standalone C++ executable for quick testing. |

The `.so` file is what Python imports when you write `from pypolca._core import fit_em`.

### Where did my compiled files go?

After installation, look inside:

```bash
ls build/cp313-cp313-macosx_26_0_arm64/
```

You will see:
- `_core.cpython-313-darwin.so` — the Python extension
- `tryout` — the test executable
- `src/cpp/libpypolca_core.a` — the compiled core library

The `.so` is also **copied** into your Python package so Python can find it:

```bash
ls pypolca/_core*.so
```

This is called an **editable install**: the `.so` lives inside `pypolca/` but was built from the C++ source files in `src/cpp/`.

### Compiling `tryout` by hand

`tryout` is a small C++ program that lets you test the core math without touching Python.  You can compile it without any build system:

```bash
c++ -std=c++17 -O2 -DNDEBUG \
    -I src/cpp/include \
    -I build/_deps/eigen-src \
    examples/tryout.cpp \
    src/cpp/core/math_ops.cpp \
    src/cpp/core/em_engine.cpp \
    -o tryout
```

What each piece means:

| Piece | Meaning |
|-------|---------|
| `c++` | The compiler (Clang on Mac, GCC on Linux). |
| `-std=c++17` | Use the C++17 language version. |
| `-O2` | Optimize for speed. |
| `-DNDEBUG` | Disable Eigen's internal safety assertions (matches Release mode). |
| `-I src/cpp/include` | "Look for `#include` files here." |
| `-I build/_deps/eigen-src` | "Eigen's headers are here." |
| `examples/tryout.cpp` | The test program. |
| `src/cpp/core/*.cpp` | The math implementation files. |
| `-o tryout` | Name the output executable `tryout`. |

Run it:

```bash
./tryout
```

### The fast rebuild loop

When you edit a C++ file, you don't need to reinstall the whole Python package.  You have two options:

**Option A — Rebuild only the C++ extension (fastest):**

```bash
./rebuild.sh
```

This runs `ninja` incrementally (only changed files are recompiled) and copies the new `.so` into `pypolca/`.

**Option B — Full clean rebuild (if something is broken):**

```bash
./rebuild.sh --clean
```

This deletes the build directory and runs the full `uv pip install -e . --no-build-isolation` again.

> **Rule of thumb:** Use `./rebuild.sh` for 99 % of edits.  Use `--clean` only if you get weird CMake errors.

---

## Part 2: How the Python Bindings Work

### What are bindings?

Python cannot directly call C++ functions.  They are two different languages with different memory layouts, type systems, and object models.

**Bindings** are a translation layer.  They let you write C++ code that Python can import and use as if it were a normal Python module.

This project uses **pybind11** to create the bindings.

### pybind11 in 30 seconds

pybind11 is a C++ library that makes binding code look almost like Python.  The basic idea:

```cpp
#include <pybind11/pybind11.h>
namespace py = pybind11;

int add(int a, int b) { return a + b; }

PYBIND11_MODULE(my_module, m) {
    m.def("add", &add, "Add two numbers");
}
```

When compiled into a shared library (`.so`), this creates a Python module you can use like:

```python
import my_module
print(my_module.add(2, 3))  # 5
```

The magic macro `PYBIND11_MODULE` generates all the boilerplate that Python's C API requires.

### Reading `bindings.cpp` line by line

Open `src/cpp/bindings.cpp`.  Here is what each section does.

**Top — includes:**

```cpp
#include <pybind11/pybind11.h>     // Core pybind11
#include <pybind11/eigen.h>         // Automatic numpy <-> Eigen conversion
#include <pybind11/stl.h>           // Automatic std::vector <-> list conversion
```

The `eigen.h` header is crucial: it teaches pybind11 how to convert a NumPy `ndarray` into an Eigen `MatrixXd` automatically, and vice versa.

**The module definition:**

```cpp
PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core for pypoLCA ...";
```

`_core` is the name of the Python module.  After compilation it becomes `pypolca/_core.cpython-...so`, which Python imports as `pypolca._core`.

**Exposing a C++ struct:**

```cpp
py::class_<pypolca::Data>(m, "Data")
    .def(py::init<>())                                    // Constructor
    .def_readwrite("y", &pypolca::Data::y, "N x J ...") // Field
    .def_readwrite("x", &pypolca::Data::x)
    .def_readwrite("num_choices", &pypolca::Data::num_choices)
    .def("n_obs", &pypolca::Data::n_obs);                // Method
```

This creates a Python class `Data` that wraps the C++ struct.  When you do:

```python
from pypolca._core import Data
d = Data()
d.y = np.array([[1, 2], [1, 3]], dtype=np.int32)
```

pybind11 allocates a C++ `pypolca::Data` object, stores it in Python, and converts the numpy array into an `Eigen::MatrixXi`.

**Exposing a function:**

```cpp
m.def("fit_em",
      &pypolca::fit_em,
      py::arg("data"),
      py::arg("nclass"),
      py::arg("maxiter") = 1000,
      py::arg("tol") = 1e-10,
      ...);
```

This creates the Python function `fit_em`.  Default arguments are preserved.  The docstring is written inside `R"pbdoc(...)pbdoc"`.

### How numpy arrays become Eigen matrices

This is the most important binding magic.  In `pypolca/api.py` you see:

```python
cpp_data = Data()
cpp_data.y = y_int          # y_int is a numpy ndarray
cpp_data.x = x_mat          # x_mat is a numpy ndarray
```

Behind the scenes:

1. pybind11 checks that `y_int` is a 2-D array of `int32`.
2. It creates an `Eigen::Map<Eigen::MatrixXi>` that points to the same memory block as the numpy array (no copy!).
3. The C++ code sees a normal Eigen matrix and does math on it.
4. When the C++ function returns an `Eigen::MatrixXd`, pybind11 wraps it in a numpy array and hands it back to Python.

> **Key insight:** The data is shared, not copied.  Modifying the numpy array on the Python side *after* passing it to C++ can affect the C++ object (and vice versa).  In this project we treat the data as read-only after hand-off.

### The Python wrapper (`pypolca/api.py`)

The C++ module (`pypolca._core`) is **low-level**.  It expects C++ structs and flat vectors.  The Python wrapper makes it friendly.

Look at `pypolca/api.py`:

```python
def fit(formula: str, data: pl.DataFrame, nclass: int = 2, ...) -> LCAResult:
    # 1. Parse a formula string like "Y1 + Y2 ~ 1"
    y_mat, x_mat, num_choices = build_design_matrix(formula, data)

    # 2. Convert Python types to C++ types
    cpp_data = Data()
    cpp_data.y = y_mat.astype(np.int32)
    cpp_data.x = x_mat.astype(np.float64)
    cpp_data.num_choices = num_choices

    # 3. Call the C++ engine
    raw = fit_em(cpp_data, nclass=nclass, ...)

    # 4. Wrap the result in a Python-friendly object
    return LCAResult(raw, formula=formula, data=data)
```

`LCAResult` is a pure Python class that exposes the C++ `Results` fields as properties:

```python
@property
def posterior(self) -> np.ndarray:
    return np.array(self._raw.posterior)
```

Here `self._raw.posterior` is an Eigen matrix returned by C++.  `np.array(...)` makes a copy so you get a normal numpy array.

### The full data journey

Here is the complete round-trip when you run:

```python
from pypolca.api import fit
result = fit("Y1 + Y2 ~ 1", df, nclass=2)
print(result.posterior)
```

```
Python script
     │
     ▼
pypolca.api.fit()
     │
     ├──▶ build_design_matrix()  ──▶  Polars DataFrame → numpy arrays
     │
     ├──▶ Data()                 ──▶  pybind11 creates C++ struct
     │      cpp_data.y = ...     ──▶  numpy → Eigen::MatrixXi
     │
     ├──▶ fit_em()               ──▶  C++ EM algorithm runs
     │         │
     │         └──▶ compute_ylik(), e_step(), m_step_probs() ...
     │
     └──▶ LCAResult(raw)         ──▶  C++ Results → Python object
              │
              └──▶ result.posterior  ──▶  Eigen::MatrixXd → numpy ndarray
```

---

## Cheat Sheet

| I want to... | Command |
|-------------|---------|
| Install the package (first time) | `uv pip install -e . --no-build-isolation` |
| Rebuild C++ after editing `.cpp` | `./rebuild.sh` |
| Full clean rebuild | `./rebuild.sh --clean` |
| Compile `tryout` without build system | `c++ -std=c++17 -O2 -DNDEBUG -I src/cpp/include -I build/_deps/eigen-src examples/tryout.cpp src/cpp/core/math_ops.cpp src/cpp/core/em_engine.cpp -o tryout` |
| Run tests | `uv run pytest tests/python -v` |
| Run the basic example | `uv run python examples/basic_fit.py` |
| Find the built `.so` | `ls build/cp313*/_core*.so` |

---

## Glossary

| Term | Meaning |
|------|---------|
| **CMake** | A tool that generates platform-specific build files from a high-level recipe. |
| **Ninja** | A fast build tool that compiles C++ code. |
| **pybind11** | A C++ library that creates Python modules from C++ code. |
| **scikit-build-core** | A Python build backend that integrates CMake into `pip install`. |
| **Eigen** | A C++ header-only library for matrix math. |
| **Shared library (`.so`)** | A compiled file that can be loaded into a running program (like Python). |
| **Static library (`.a`)** | A compiled file that is copied into another program at link time. |
| **Editable install** | Installing a package so that changes to source files are reflected immediately without re-installing. |
