C++ API
=======

The C++ backend implements the EM algorithm and all mathematical operations.
It is exposed to Python via pybind11 but can also be used directly.

Core Types
----------

.. doxygenstruct:: pypolca::Data
    :members:
    :undoc-members:

.. doxygenstruct:: pypolca::Params
    :members:
    :undoc-members:

.. doxygenstruct:: pypolca::Results
    :members:
    :undoc-members:

.. doxygenstruct:: pypolca::SEs
    :members:
    :undoc-members:

EM Engine
---------

.. doxygenfunction:: pypolca::fit_em

Math Operations
---------------

.. doxygenfunction:: pypolca::compute_log_ylik

.. doxygenfunction:: pypolca::e_step

.. doxygenfunction:: pypolca::m_step_probs

.. doxygenfunction:: pypolca::compute_prior_from_beta

.. doxygenfunction:: pypolca::compute_beta_derivatives

.. doxygenfunction:: pypolca::update_beta

.. doxygenfunction:: pypolca::compute_standard_errors

.. doxygenfunction:: pypolca::compute_logsumexp
