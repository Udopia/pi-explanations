# Solbert

[![Build-Test](https://github.com/Udopia/solbert/actions/workflows/build_test.yml/badge.svg?branch=main)](https://github.com/Udopia/solbert/actions/workflows/build_test.yml)
[![PyPI](https://img.shields.io/pypi/v/solbert.svg)](https://pypi.org/project/solbert/)
[![Python](https://img.shields.io/pypi/pyversions/solbert.svg)](https://pypi.org/project/solbert/)
[![License](https://img.shields.io/github/license/Udopia/solbert.svg)](https://github.com/Udopia/solbert/blob/main/LICENSE)

Solbert computes **prime-implicant explanations** for scikit-learn decision trees and random forests.
These explanations are also known as *abductive explanations* or *sufficient reasons*: inclusion-minimal sets of feature conditions that are enough to force a classification result.

Solbert translates a trained classifier into a propositional formula, uses the incremental SAT solver [CaDiCaL](https://github.com/arminbiere/cadical) through
its IPASIR interface, and decodes the resulting prime implicants back into feature-space case distinctions.

## Installation

Install a prebuilt wheel from PyPI:

```bash
python -m pip install solbert
```

Wheels are built for supported CPython versions on Linux (x86-64 and AArch64) and macOS (Apple Silicon).

## Package Structure

- `solbert` provides native prime-implicant and projected-model operations.
- `solbert.tree` provides decision-tree wrappers, encoders, and explainers.
- `solbert.forest` provides random-forest wrappers, encoders, and explainers.
- `solbert._native` is the private C++ extension backed by CaDiCaL.

The Python encoders represent split thresholds, tree nodes, and classification outcomes as a CNF resembling a monotonic circuit.
Prime implicants of that encoding are minimized incrementally and decoded into minimal case distinctions over the original features.

## Development

Install directly from a checkout:

```bash
python -m pip install ./solbert
```

The source build requires a C++17 compiler, CMake, Git, and Make.
It downloads a pinned CaDiCaL release automatically.
Local CaDiCaL builds use one job by default; a stronger machine can opt into parallel compilation:

```bash
CMAKE_ARGS="-DSOLBERT_CADICAL_BUILD_JOBS=4" python -m pip install ./solbert
```

To reuse an existing position-independent CaDiCaL archive:

```bash
CMAKE_ARGS="-DSOLBERT_CADICAL_LIBRARY=/path/to/libcadical.a" \
  python -m pip install ./solbert
```

To build and run directly from the checkout without installing:

```bash
cmake -S solbert -B solbert/build
cmake --build solbert/build --target _native
python solbert/eval.py 1
```

CMake builds copy the native extension into the source package. The generated
extension is ignored by Git.

Run the package tests with:

```bash
python solbert/test/test_solbert.py
```

## Research

Solbert implements the method described in *Calculating Sufficient Reasons for Random Forest Classifiers* by Ashlin Iser.
The [paper source](https://github.com/Udopia/solbert/tree/main/paper2) contains the formal encodings, prime-implicant algorithm, and evaluation on SAT benchmark classification and solver selection.

If you use Solbert in research, please cite that paper and link to this repository.
