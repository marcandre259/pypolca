Quickstart
==========

Installation
------------

.. code-block:: bash

    pip install polca

Basic Usage
-----------

The simplest way to fit a latent class model is with the :func:`pypolca.fit`
function and a formula string:

.. code-block:: python

    from pypolca import fit, load_dataset

    data = load_dataset("CHEATING")
    result = fit("cbind(GPA, LIEEXAM, LIEFOOL, LIEMARCH) ~ 1", data, nclass=2)

    print(result)
    # Model converged in 28 iterations
    # log-likelihood: -435.99
    # AIC: 891.98, BIC: 916.06
    # Class shares: 0.77, 0.23

Formula Syntax
--------------

Outcome variables are wrapped in ``cbind(...)``. Covariates go after ``~``:

- ``cbind(Y1, Y2) ~ 1`` — no covariates (intercept only)
- ``cbind(Y1, Y2, Y3) ~ COV1 + COV2`` — two covariates
- ``cbind(Y1) ~ SEX + AGE`` — single outcome with covariates

Results Object
--------------

The :class:`pypolca.LCAResult` object provides everything you need:

.. code-block:: python

    result.probs          # class-conditional response probabilities
    result.posterior      # posterior class membership (N × nclass)
    result.predclass      # modal class assignment
    result.loglik         # log-likelihood
    result.aic, result.bic  # information criteria
    result.coeff           # covariate coefficients (beta)

Low-Level API
-------------

For advanced use, you can call the C++ functions directly:

.. code-block:: python

    from pypolca import Data, compute_ylik, fit_em

    data = Data(y, x, num_choices)
    results = fit_em(data, nclass=3, maxiter=1000, tol=1e-8)
