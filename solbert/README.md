# Solbert

Solbert is a CPython extension for SAT-based computation of prime implicants
and projected models, with Python encoders and explainers for scikit-learn
decision trees and random forests. It uses the IPASIR interface provided by
CaDiCaL.

The public API is organized into three namespaces:

```python
import solbert
from solbert.tree import DecisionTreeEncoder, DecisionTreeExplainer, DecisionTreeWrapper
from solbert.forest import RandomForestEncoder, RandomForestExplainer, RandomForestWrapper
```

The native SAT API remains available directly from `solbert`; its compiled
implementation lives in the private `solbert._native` module.

## Installation

Install a published wheel with:

```bash
python3 -m pip install solbert
```

To build from source, a C++17 compiler, CMake, Git, and Make are required.
The build downloads and compiles the pinned CaDiCaL release automatically:

```bash
python3 -m pip install ./solbert
```

## Development

Configure, build, and test the extension directly with CMake:

```bash
cmake -S solbert -B solbert/build -DCMAKE_BUILD_TYPE=Debug
cmake --build solbert/build --parallel 1
ctest --test-dir solbert/build --output-on-failure
```

An existing CaDiCaL archive can be reused without a download:

```bash
cmake -S solbert -B solbert/build \
  -DSOLBERT_CADICAL_LIBRARY=/path/to/libcadical.a
```

## Publishing

Publishing a GitHub release builds and uploads the supported wheels and source
distribution. Configure a PyPI Trusted Publisher for this repository and the
`build_wheels.yml` workflow, without an environment restriction.
