Examples
========

Built-in Datasets
-----------------

pypolca ships with five classic latent class analysis datasets:

.. code-block:: python

    from pypolca import load_dataset, get_dataset_info

    get_dataset_info("ELECTION")
    data = load_dataset("ELECTION")

Full Example: 2-Class Model
---------------------------

.. code-block:: python

    from pypolca import fit
    from pypolca.data import load_dataset

    # Load the Cheating dataset
    cheating = load_dataset("CHEATING")

    # Fit 2-class model
    result = fit(
        "cbind(GPA, LIEEXAM, LIEFOOL, LIEMARCH) ~ 1",
        cheating,
        nclass=2,
        nrep=5,
    )

    # Inspect results
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"log-likelihood: {result.loglik:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")

    # Class-conditional probabilities
    print(result.probs)

    # Posterior class membership
    print(result.posterior)

    # Predict posterior for new data
    posterior_new = result.predict_posterior(cheating)

With Covariates
---------------

.. code-block:: python

    from pypolca import fit, load_dataset

    gss = load_dataset("GSS82")
    result = fit(
        "cbind(PURPOSE, ACCURACY, UNDERSTA, COOPERAT) ~ AGE + ETH",
        gss,
        nclass=3,
        nrep=10,
    )

    print(result.coeff)    # covariate coefficients
    print(result.coeff_se) # standard errors
