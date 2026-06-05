"""Built-in datasets from R's poLCA package.

All datasets are re-exported from R's poLCA (GPL-2.0-or-later, compatible with
this package) as CSV files. Use `load_dataset()` with a `Dataset` enum member
to load one as a Polars DataFrame.

Usage::

    from pypolca.data import load_dataset, Dataset

    df = load_dataset(Dataset.CARCINOMA)
    # or by name:
    df = load_dataset("carcinoma")

    from pypolca import fit
    result = fit("cbind(A,B,C,D,E,F,G) ~ 1", df, nclass=2)
"""

from enum import StrEnum
from importlib.resources import files as _files
from typing import Any

import polars as pl

_DATA_DIR = _files("pypolca.data")


class Dataset(StrEnum):
    """Built-in datasets available for loading.

    Attributes
    ----------
    CARCINOMA : str
        Dichotomous ratings by seven pathologists of 118 slides for the
        presence or absence of carcinoma in the uterine cervix. Columns:
        A–G (1=no, 2=yes). Source: Agresti (2002), Table 13.1.
    CHEATING : str
        319 undergraduate students surveyed on chronic cheating behavior.
        Columns: LIEEXAM, LIEPAPER, FRAUD, COPYEXAM (1=no, 2=yes), GPA (1–5).
    ELECTION : str
        2000 American National Election Study survey, 1,785 respondents.
        12 trait ratings (MORALG–INTELB, 1–4) for Gore and Bush, plus
        VOTE3, AGE, EDUC, GENDER, PARTY covariates.
    GSS82 : str
        1,202 white respondents to the 1982 General Social Survey.
        Columns: PURPOSE (1–3), ACCURACY (1–2), UNDERSTA (1–3),
        COOPERAT (1–3). Source: McCutcheon (1987), Table 3.1.
    VALUES : str
        216 respondents on four dichotomous items measuring universalistic
        vs. particularistic values. Columns: A–D (1=universalistic,
        2=particularistic).
    """

    CARCINOMA = "carcinoma"
    CHEATING = "cheating"
    ELECTION = "election"
    GSS82 = "gss82"
    VALUES = "values"


# Column descriptions for use in docstrings / help
_DATASET_META: dict[Dataset, dict[str, Any]] = {
    Dataset.CARCINOMA: {
        "description": "Diagnoses of carcinoma by 7 pathologists, 118 slides.",
        "columns": {
            "A": "Pathologist A (1=no, 2=yes)",
            "B": "Pathologist B (1=no, 2=yes)",
            "C": "Pathologist C (1=no, 2=yes)",
            "D": "Pathologist D (1=no, 2=yes)",
            "E": "Pathologist E (1=no, 2=yes)",
            "F": "Pathologist F (1=no, 2=yes)",
            "G": "Pathologist G (1=no, 2=yes)",
        },
        "source": "Agresti, A. (2002). Categorical Data Analysis, 2nd ed., Table 13.1.",
        "example_formula": "cbind(A,B,C,D,E,F,G) ~ 1",
        "nclass_example": 2,
    },
    Dataset.CHEATING: {
        "description": "GPA and chronic cheating survey, 319 students.",
        "columns": {
            "LIEEXAM": "Lied to avoid an exam (1=no, 2=yes)",
            "LIEPAPER": "Lied to avoid handing in a paper (1=no, 2=yes)",
            "FRAUD": "Witnessed/engaged in fraud (1=no, 2=yes)",
            "COPYEXAM": "Copied from another student (1=no, 2=yes)",
            "GPA": "Grade point average group (1–5)",
        },
        "source": "R poLCA built-in dataset.",
        "example_formula": "cbind(LIEEXAM,LIEPAPER,FRAUD,COPYEXAM) ~ GPA",
        "nclass_example": 2,
    },
    Dataset.ELECTION: {
        "description": "2000 NES survey, 1,785 respondents, candidate trait + covariate data.",
        "columns": {
            "MORALG": "Gore: moral (1=Extremely well – 4=Not well)",
            "CARESG": "Gore: caring",
            "KNOWG": "Gore: knowledgeable",
            "LEADG": "Gore: good leader",
            "DISHONG": "Gore: dishonest",
            "INTELG": "Gore: intelligent",
            "MORALB": "Bush: moral",
            "CARESB": "Bush: caring",
            "KNOWB": "Bush: knowledgeable",
            "LEADB": "Bush: good leader",
            "DISHONB": "Bush: dishonest",
            "INTELB": "Bush: intelligent",
            "VOTE3": "Vote choice (1=Gore, 2=Bush, 3=Other)",
            "AGE": "Age in years",
            "EDUC": "Education (1=≤8 grades – 7=Advanced degree)",
            "GENDER": "Gender (1=Male, 2=Female)",
            "PARTY": "Party ID (1=Strong Democrat – 7=Strong Republican)",
        },
        "source": "The National Election Studies (https://electionstudies.org/).",
        "example_formula": "cbind(MORALG,CARESG,KNOWG,LEADG,DISHONG,INTELG,MORALB,CARESB,KNOWB,LEADB,DISHONB,INTELB) ~ 1",
        "nclass_example": 3,
    },
    Dataset.GSS82: {
        "description": "1982 General Social Survey, 1,202 white respondents, survey attitudes.",
        "columns": {
            "PURPOSE": "Purpose of surveys (1=good, 2=depends, 3=waste)",
            "ACCURACY": "Survey accuracy (1=mostly true, 2=not true)",
            "UNDERSTA": "Understanding questions (1=good, 2=fair, 3=poor)",
            "COOPERAT": "Cooperation with interviewer (1=interested, 2=cooperative, 3=impatient/hostile)",
        },
        "source": "McCutcheon, A.L. (1987). Latent Class Analysis, Table 3.1.",
        "example_formula": "cbind(PURPOSE,ACCURACY,UNDERSTA,COOPERAT) ~ 1",
        "nclass_example": 2,
    },
    Dataset.VALUES: {
        "description": "Universalistic vs. particularistic values, 216 respondents.",
        "columns": {
            "A": "Item A (1=universalistic, 2=particularistic)",
            "B": "Item B (1=universalistic, 2=particularistic)",
            "C": "Item C (1=universalistic, 2=particularistic)",
            "D": "Item D (1=universalistic, 2=particularistic)",
        },
        "source": "R poLCA built-in dataset.",
        "example_formula": "cbind(A,B,C,D) ~ 1",
        "nclass_example": 2,
    },
}


def load_dataset(name: Dataset | str) -> pl.DataFrame:
    """Load a built-in dataset as a Polars DataFrame.

    Parameters
    ----------
    name : Dataset or str
        Dataset to load, e.g. ``Dataset.CARCINOMA`` or ``"carcinoma"``.

    Returns
    -------
    pl.DataFrame

    Raises
    ------
    ValueError
        If *name* is not a valid dataset.

    Examples
    --------
    >>> from pypolca.data import load_dataset, Dataset
    >>> df = load_dataset(Dataset.CARCINOMA)
    >>> df.shape
    (118, 7)
    """
    if isinstance(name, Dataset):
        ds = name
    else:
        try:
            ds = Dataset(name.lower())
        except ValueError:
            valid = ", ".join(Dataset)
            raise ValueError(f"Unknown dataset {name!r}. Valid options: {valid}") from None

    return pl.read_csv(str(_DATA_DIR / f"{ds.value}.csv"), null_values="NA")


def get_dataset_info(name: Dataset | str) -> dict[str, Any]:
    """Return metadata for a dataset (description, columns, source, example).

    Parameters
    ----------
    name : Dataset or str
        Dataset name.

    Returns
    -------
    dict
        Keys: ``description`` (str), ``columns`` (dict), ``source`` (str),
        ``example_formula`` (str), ``nclass_example`` (int).
    """
    if isinstance(name, str):
        name = Dataset(name.lower())
    return dict(_DATASET_META[name])
