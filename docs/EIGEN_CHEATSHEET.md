# Eigen Cheatsheet

Quick reference for common Eigen C++ operations (dynamic dense types).

---

## Initialization

### `VectorXd` (dynamic column vector)

```cpp
#include <Eigen/Dense>

// 1. Size then comma initializer
Eigen::VectorXd v(3);
v << 1, 2, 3;

// 2. Fixed-size then cast (compile-time size)
Eigen::Vector3d v3(1, 2, 3);
Eigen::VectorXd v = v3;

// 3. Map from raw array / std::vector
double data[] = {1, 2, 3};
Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(data, 3);

// 4. Zero / Ones / Constant
Eigen::VectorXd z = Eigen::VectorXd::Zero(5);
Eigen::VectorXd o = Eigen::VectorXd::Ones(5);
Eigen::VectorXd c = Eigen::VectorXd::Constant(5, 3.14);
```

### `MatrixXd` (dynamic matrix)

```cpp
// 1. Rows/cols then comma initializer (column-major by default)
Eigen::MatrixXd m(2, 3);
m << 1, 2, 3,
     4, 5, 6;
// Layout: each row separated by comma, columns separated by <<
// Actually: first row elements, then second row...
// m(0,0)=1, m(0,1)=2, m(0,2)=3, m(1,0)=4, ...

// 2. Fixed-size then cast
Eigen::Matrix2d m2;
m2 << 1, 2,
      3, 4;
Eigen::MatrixXd m = m2;

// 3. Map from raw array (column-major storage)
double data[] = {1, 4, 2, 5, 3, 6};  // col-major: col0, col1, col2
Eigen::MatrixXd m = Eigen::Map<Eigen::MatrixXd>(data, 2, 3);

// 4. Convenience factories
Eigen::MatrixXd z = Eigen::MatrixXd::Zero(3, 3);
Eigen::MatrixXd i = Eigen::MatrixXd::Identity(3, 3);
Eigen::MatrixXd o = Eigen::MatrixXd::Ones(3, 3);
Eigen::MatrixXd c = Eigen::MatrixXd::Constant(3, 3, 7.0);
```

---

## Assignment

### Single element

```cpp
Eigen::VectorXd v(3);
v(0) = 1.0;
v[1] = 2.0;

Eigen::MatrixXd m(2, 2);
m(0, 0) = 1.0;
m(1, 1) = 4.0;
```

### Whole object (comma initializer)

```cpp
Eigen::VectorXd v(3);
v << 1, 2, 3;

Eigen::MatrixXd m(2, 3);
m << 1, 2, 3,
     4, 5, 6;
```

### From another matrix/vector (copy)

```cpp
Eigen::MatrixXd a = Eigen::MatrixXd::Random(3, 3);
Eigen::MatrixXd b(3, 3);
b = a;                    // deep copy
b = a.transpose();        // assigns transposed copy
b = a.block(0, 0, 2, 2);  // assigns 2×2 block (resizes b)
```

### Block / segment assignment

```cpp
Eigen::VectorXd v(5);
v.setZero();
v.segment(1, 3) << 10, 20, 30;   // mutates elements 1..3

Eigen::MatrixXd m(3, 3);
m.setZero();
m.block(0, 0, 2, 2) << 1, 2,
                       3, 4;
m.row(1).setOnes();
m.col(2).setConstant(7.0);
```

### Set methods (in-place)

```cpp
Eigen::MatrixXd m(3, 3);
m.setZero();
m.setOnes();
m.setConstant(3.14);
m.setRandom();            // uniform in [-1, 1]
m.setIdentity();

// Same methods exist for VectorXd
Eigen::VectorXd v(5);
v.setZero();
v.setOnes();
v.setConstant(42.0);
v.setRandom();
```

### Scalar assignment (fills all entries)

```cpp
Eigen::MatrixXd m(2, 2);
m = 5.0;                  // every entry becomes 5.0

Eigen::VectorXd v(3);
v = 1.0;                  // every entry becomes 1.0
```

---

## Slicing / Indexing

### `VectorXd`

```cpp
Eigen::VectorXd v(5);
v << 1, 2, 3, 4, 5;

v(2);                // element access: 3
v[2];                // also works

v.segment(1, 3);     // block: start, size  -> [2,3,4]
v.head(2);           // first 2             -> [1,2]
v.tail(2);           // last 2              -> [4,5]

// Assign to slice (mutates in place)
v.segment(1, 2) << 10, 20;
```

### `MatrixXd`

```cpp
Eigen::MatrixXd m(3, 4);
m << 1, 2, 3, 4,
     5, 6, 7, 8,
     9, 10, 11, 12;

// Element
m(1, 2);             // row 1, col 2  -> 7

// Block (sub-matrix)
m.block(1, 2, 2, 2); // startRow, startCol, blockRows, blockCols
                     // -> [[7, 8],
                     //     [11,12]]

// Row / column vectors
m.row(1);            // [5, 6, 7, 8]  (returns RowVectorXd)
m.col(2);            // [3, 7, 11]    (returns VectorXd)

// Top/bottom/left/right blocks
m.topRows(2);        // first 2 rows
m.bottomRows(1);     // last row
m.leftCols(2);       // first 2 columns
m.rightCols(2);      // last 2 columns
m.middleRows(1, 2);  // startRow, numRows
m.middleCols(1, 2);  // startCol, numCols

// Corner blocks
m.topLeftCorner(2, 2);
m.bottomRightCorner(2, 2);

// Assign to block (mutates in place)
m.block(0, 0, 2, 2) << 0, 0, 0, 0;
m.row(1).setZero();
```

---

## Type Casting

### Whole matrix/vector

```cpp
Eigen::VectorXd  vd = Eigen::VectorXd::Random(5);
Eigen::MatrixXd  md = Eigen::MatrixXd::Random(3, 3);

// Cast to float
Eigen::VectorXf vf = vd.cast<float>();
Eigen::MatrixXf mf = md.cast<float>();

// Cast to int
Eigen::VectorXi vi = vd.cast<int>();      // truncates toward zero
Eigen::MatrixXi mi = md.cast<int>();
```

### Single element

```cpp
float  f = static_cast<float>(v(0));
int    i = static_cast<int>(v(0));       // truncates toward zero
```

---

## Linear Algebra

### Dot product

```cpp
Eigen::VectorXd a(3);
a << 1, 2, 3;
Eigen::VectorXd b(3);
b << 4, 5, 6;

// Member function (most common)
double result = a.dot(b);   // 1*4 + 2*5 + 3*6 = 32

// Also works with expressions
double r = a.head(2).dot(b.tail(2));
```

For complex vectors, `.dot()` conjugates the **left** operand. Use `.cwiseProduct().sum()` to skip conjugation:

```cpp
Eigen::VectorXcd u = /* ... */;
Eigen::VectorXcd v = /* ... */;
std::complex<double> r1 = u.dot(v);               // conj(u) · v
std::complex<double> r2 = u.cwiseProduct(v).sum(); // u · v (no conj)
```

### Matrix-vector / matrix-matrix multiplication

```cpp
Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
Eigen::VectorXd x = Eigen::VectorXd::Random(3);

Eigen::VectorXd y = A * x;        // matrix-vector
Eigen::MatrixXd C = A * A;        // matrix-matrix
Eigen::MatrixXd D = A.transpose() * A;  // AᵀA
```

## Reshaping & Maps

```cpp
Eigen::VectorXd v(6);
v << 1, 2, 3, 4, 5, 6;

// Vector -> Matrix (column-major by default)
Eigen::MatrixXd m = v.reshaped(2, 3);
// [[1, 3, 5],
//  [2, 4, 6]]

// Matrix -> Vector (by columns)
Eigen::VectorXd v2 = m.reshaped();

// Map with row-major storage
double data[] = {1, 2, 3, 4, 5, 6};
Eigen::MatrixXd m = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, 2, 3);
```

---

## Common Gotchas

| Gotcha | Fix |
|--------|-----|
| `v << 1, 2` on a `VectorXd` without setting size first | `VectorXd v(2);` before `<<` |
| Matrix `<<` layout confusion | Eigen is **column-major** by default; `m << a,b,c,d` for 2×2 fills `m(0,0)=a, m(1,0)=b, m(0,1)=c, m(1,1)=d` |
| Slicing returns expressions, not copies | Assign to `auto` or explicit type if you need a copy |
| `.cast<int>()` truncates, does not round | Use `(x + 0.5).cast<int>()` for rounding if positive |

---

## Quick Reference Table

| Operation | `VectorXd` | `MatrixXd` |
|-----------|------------|------------|
| Size | `.size()` | `.rows()`, `.cols()` |
| Element | `v(i)` | `m(r, c)` |
| Block | `.segment(i, n)` | `.block(r, c, rows, cols)` |
| Row | — | `.row(r)` |
| Column | — | `.col(c)` |
| Head | `.head(n)` | `.topRows(n)` |
| Tail | `.tail(n)` | `.bottomRows(n)` |
| Cast | `.cast<T>()` | `.cast<T>()` |
| Dot product | `.dot(v)` | — |
| Multiply | — | `A * B` |
