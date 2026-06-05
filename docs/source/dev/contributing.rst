Contributing
============

Development Setup
-----------------

.. code-block:: bash

    git clone https://github.com/marcandre259/pypolca
    cd pypoLCA
    uv pip install -e . --no-build-isolation

After editing C++ code, rebuild with:

.. code-block:: bash

    ./rebuild.sh

Project Structure
-----------------

- ``src/cpp/`` — C++17 EM engine (Eigen + pybind11)
- ``pypolca/`` — Python package (high-level API, formula parsing, datasets)
- ``tests/`` — Python test suite
- ``docs/`` — Sphinx documentation (this site)

Building Docs
-------------

.. code-block:: bash

    uv pip install -e ".[docs]"
    cd docs
    make html
    open build/html/index.html
