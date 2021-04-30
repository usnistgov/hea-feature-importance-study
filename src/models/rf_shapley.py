from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
from sklearn import ensemble, pipeline, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from src.data import load_hea_dataset, split_chemical_systems
from src.features import randcat


class Target(str, Enum):
    multiphase = "Multiphase"
    intermetallic = "IM"


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray


@dataclass
class Result:
    auc_train: float
    auc_val: float
    auc_test: float
    impurity_importance: np.ndarray
    shapley_importance: np.ndarray


def fit_replicate(
    model,
    df: pd.DataFrame,
    data: Dataset,
    testdata: Dataset,
    n_random_features: int = 1,
    verbose: bool = False,
):

    # random feature augmentation
    data.X = randcat(data.X, n_random_features=n_random_features)
    testdata.X = randcat(testdata.X, n_random_features=n_random_features)

    # train/val split
    train_ids, val_ids = split_chemical_systems(df, verbose=verbose)
    train = Dataset(data.X[train_ids], data.y[train_ids])
    val = Dataset(data.X[val_ids], data.y[val_ids])

    # standardization not needed for models without PCA...
    # # feature standardization
    # if np.std(train.X, axis=0).min() == 0:
    #     print("bad standardization!")
    # std = preprocessing.StandardScaler()

    # std.fit(train.X)

    # train.X = std.transform(train.X)
    # val.X = std.transform(val.X)
    # testdata.X = std.transform(testdata.X)

    model.fit(train.X, train.y)
    feature_importances = model.feature_importances_

    def compute_roc(model, X, y):
        scores = model.predict_proba(X)[:, 1]
        return roc_auc_score(y, scores)

    r = {
        "auc_train": compute_roc(model, train.X, train.y),
        "auc_val": compute_roc(model, val.X, val.y),
        "auc_test": compute_roc(model, testdata.X, testdata.y),
        "impurity_importance": feature_importances,
    }

    impurity_rank = np.argsort(feature_importances)[::-1]
    if verbose:
        print("random feature impurity rank: ", impurity_rank[-1])

    # aggregated shapley importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(train.X)
    shap_values = shap_values[..., 1]  # just values for positive class
    aggregated_shap = np.abs(shap_values.data).mean(axis=0)
    shap_rank = np.argsort(aggregated_shap)[::-1]
    if verbose:
        print("random feature shap rank: ", shap_rank[-1], aggregated_shap[-1])

    r["shapley_importance"] = aggregated_shap

    return Result(**r)


def main(
    target: Target = Target.multiphase,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    replicates: int = 1,
    n_random_features: int = 1,
):
    target_key = f"PROPERTY: {target.value}"
    print(target_key)

    df, X = load_hea_dataset(subset="train")
    y = df[target_key].values
    data = Dataset(X, y)

    df_acta, X_acta = load_hea_dataset(subset="test")
    y_acta = df_acta[target_key].values
    testdata = Dataset(X_acta, y_acta)

    model = ensemble.RandomForestClassifier(
        n_estimators=144,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=4,
        class_weight="balanced",
    )

    results = []
    for replicate in tqdm(range(replicates)):
        results.append(fit_replicate(model, df, data, testdata))

    print(pd.DataFrame(results))


if __name__ == "__main__":
    typer.run(main)
