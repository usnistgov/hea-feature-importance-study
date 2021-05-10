from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import model_selection

import src.features

datafile = "data/HEA_database_additions.csv"


def load_hea_dataset(subset="train", progress=True):
    """deciding what property to target, opening the datafile containing the training data, and pulling out the holdout set"""

    df = pd.read_csv(datafile)

    df, X = src.features.compute_composition_features(df, progress=progress)

    # split data by source
    id_acta = df["REFERENCE: doi"] == "10.1016/j.actamat.2019.06.032"

    if subset == "train":
        selection = ~id_acta.values
    elif subset == "test":
        selection = id_acta.values

    X = X.iloc[selection]

    return df.iloc[selection], X


def load_citrine_dataset(
    datafile="data/Citrine_MPEA_dataset.csv", subset="train", progress=True
):
    df = pd.read_csv(datafile)
    # remove NaNs and Other from Property: Microstructure
    df = df.dropna(
        subset=["PROPERTY: Microstructure", "PROPERTY: Processing method"]
    )
    df = df[~df["PROPERTY: Microstructure"].str.contains("Other")]
    df = df[df["PROPERTY: Processing method"].str.contains("CAST")]
    df = df.drop_duplicates(subset=["Formula"])

    df, X = src.features.compute_composition_features(df, progress=progress)

    # split data by source
    id_acta = df["REFERENCE: doi"] == "10.1016/j.actamat.2019.06.032"

    if subset == "train":
        selection = ~id_acta.values
    elif subset == "test":
        selection = id_acta.values

    X = X.iloc[selection]

    return df.iloc[selection], X


def split_chemical_systems(
    df: pd.DataFrame, test_size: float = 0.2, verbose: bool = True
) -> tuple(np.array, np.array):
    """a group shuffle split cv iterator!

    group rows that share a chemical system together across the train/val split
    """

    cv = model_selection.GroupShuffleSplit(n_splits=1, test_size=test_size)
    for train, val in cv.split(df, groups=df["system"]):
        if verbose:
            print(f"{train.size} training data")
            print(f"{val.size} validation data")

    return train, val
