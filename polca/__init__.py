"""pypoLCA: Polytomous Variable Latent Class Analysis in C++ with Python bindings."""

from polca._core import Data, Results, fit_em, compute_ylik, compute_prior_from_beta
from polca.api import fit

__all__ = [
    "Data",
    "Results",
    "fit_em",
    "compute_ylik",
    "compute_prior_from_beta",
    "fit",
]

__version__ = "0.1.0"
