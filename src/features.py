from __future__ import annotations

import matminer.featurizers.composition
import numpy as np
import pandas as pd
from matminer import featurizers
from matminer.featurizers.conversions import StrToComposition


def compute_composition_features(df: pd.DataFrame) -> tuple(pd.DataFrame, pd.DataFrame):
    """ run standard magpie, but drop space group number... """
    features = [
        "Number",
        "MendeleevNumber",
        "AtomicWeight",
        "MeltingT",
        "Column",
        "Row",
        "CovalentRadius",
        "Electronegativity",
        "NsValence",
        "NpValence",
        "NdValence",
        "NfValence",
        "NValence",
        "NsUnfilled",
        "NpUnfilled",
        "NdUnfilled",
        "NfUnfilled",
        "NUnfilled",
        "GSvolume_pa",
        "GSbandgap",
        "GSmagmom",
    ]
    stats = ["mean", "avg_dev", "minimum", "maximum", "range"]
    magpie = featurizers.composition.ElementProperty("magpie", features, stats)

    # parse compositions from formula strings
    comp = StrToComposition()
    df = comp.featurize_dataframe(df, "Formula")

    # get the chemical system (i.e. a tuple of constituent species in alphabetical order)
    df["system"] = df["composition"].apply(lambda x: tuple(sorted(x.as_dict().keys())))

    # compute magpie featurization
    # assign to a throwaway dataframe, and slice out the feature columns
    _df = magpie.featurize_dataframe(df, "composition")
    _df.head()
    features = _df.iloc[:, df.shape[1] :]

    return df, features


def randcat(X: np.array, n_random_features: int = 1):
    """ append n_random_features columns sampled from independent standard normal distributions """
    N, D = X.shape
    return np.hstack((X, np.random.normal(size=(N, n_random_features))))
