"""pypoLCA: Polytomous Variable Latent Class Analysis in C++ with Python bindings."""

from pypolca._core import Data, Results, fit_em, compute_ylik, compute_prior_from_beta
from pypolca.api import fit
from pypolca.data import Dataset, load_dataset, get_dataset_info

__all__ = [
    "Data",
    "Results",
    "fit_em",
    "compute_ylik",
    "compute_prior_from_beta",
    "fit",
    "Dataset",
    "load_dataset",
    "get_dataset_info",
]

__version__ = "0.4.0"
