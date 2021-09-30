import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
from scipy import stats
from sklearn import ensemble, pipeline, preprocessing
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.data import (
    load_citrine_dataset,
    load_hea_dataset,
    split_chemical_systems,
)
from src.features import randcat

app = typer.Typer(help="Random forest classification script.")


class Target(str, Enum):
    multiphase = "Multiphase"
    intermetallic = "IM"


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray


@dataclass
class Result:
    train_AUC: float
    train_precision: float
    train_recall: float
    train_mAP: float
    val_AUC: float
    val_precision: float
    val_recall: float
    val_mAP: float
    test_AUC: float
    test_precision: float
    test_recall: float
    test_mAP: float
    impurity_importance: np.ndarray
    shapley_train_importance: np.ndarray
    shapley_val_importance: np.ndarray


def fit_replicate(
    model,
    df: pd.DataFrame,
    data: Dataset,
    testdata: Dataset,
    n_random_features: int = 1,
    verbose: bool = False,
):

    # random feature augmentation
    _data = Dataset(
        randcat(data.X.values, n_random_features=n_random_features), y=data.y
    )
    test = Dataset(
        randcat(testdata.X, n_random_features=n_random_features),
        y=testdata.y,
    )

    # train/val split
    train_ids, val_ids = split_chemical_systems(df, verbose=verbose)
    train = Dataset(_data.X[train_ids], _data.y[train_ids])
    val = Dataset(_data.X[val_ids], _data.y[val_ids])

    # print(data.X.shape)
    # print(train.X.shape)
    # print(val.X.shape)

    # standardization not needed for models without PCA...
    # # feature standardization
    # if np.std(train.X, axis=0).min() == 0:
    #     print("bad standardization!")
    std = preprocessing.StandardScaler()

    std.fit(train.X)

    train.X = std.transform(train.X)
    val.X = std.transform(val.X)
    test.X = std.transform(test.X)

    model.fit(train.X, train.y)
    feature_importances = model.feature_importances_

    def compute_metrics(model, dataset, split="train"):
        scores = model.predict_proba(dataset.X)[:, 1]
        predictions = model.predict(dataset.X)
        results = {
            f"{split}_AUC": roc_auc_score(dataset.y, scores),
            f"{split}_recall": recall_score(dataset.y, predictions),
            f"{split}_precision": precision_score(
                dataset.y, predictions, zero_division=0
            ),
            f"{split}_mAP": average_precision_score(dataset.y, scores),
        }
        return results

    r = compute_metrics(model, train, split="train")
    r.update(compute_metrics(model, val, split="val"))
    r.update(compute_metrics(model, test, split="test"))
    r["impurity_importance"] = feature_importances

    impurity_rank = np.argsort(feature_importances)[::-1]
    if verbose:
        print("random feature impurity rank: ", impurity_rank[-1])

    # aggregated shapley importance
    def shapley_importance(explainer, X):
        shap_values = explainer(X)
        shap_values = shap_values[..., 1]  # just values for positive class
        return np.abs(shap_values.data).mean(axis=0)

    explainer = shap.TreeExplainer(model)
    shap_train = shapley_importance(explainer, train.X)
    r["shapley_train_importance"] = shap_train
    shap_val = shapley_importance(explainer, val.X)
    r["shapley_val_importance"] = shap_val

    shap_rank_train = stats.rankdata(-shap_train)
    shap_rank_val = stats.rankdata(-shap_val)
    if verbose:
        print(
            "random feature shap rank (train): ",
            shap_rank_train[-1],
            shap_train[-1],
        )
        print(
            "random feature shap rank (val): ", shap_rank_val[-1], shap_val[-1]
        )

    return Result(**r)


@app.command()
def cv(
    target: Target = Target.multiphase,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 1,
    max_features: Optional[int] = 10,
    replicates: int = 1,
    n_random_features: int = 1,
    progress: bool = True,
    results_dir: Path = Path("data/shapley_results"),
):
    os.makedirs(results_dir, exist_ok=True)

    target_key = f"PROPERTY: {target.value}"
    if progress:
        print(target_key)

    # df, X = load_hea_dataset(subset="train", progress=progress)
    df, X = load_citrine_dataset(subset="none", progress=progress)
    y = df[target_key].values
    data = Dataset(X, y)

    df_acta, X_acta = load_hea_dataset(subset="test", progress=progress)
    # df_acta, X_acta = load_citrine_dataset(subset="test", progress=progress)
    y_acta = df_acta[target_key].values
    testdata = Dataset(X_acta, y_acta)

    run_id = f"{target.value}_{max_depth=}_{min_samples_leaf=}_{max_features=}"
    if progress:
        print(run_id)

    results = []
    for replicate in tqdm(range(replicates), disable=(not progress)):
        model = ensemble.RandomForestClassifier(
            n_estimators=144,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=4,
            class_weight="balanced",
        )

        results.append(fit_replicate(model, df, data, testdata))

    results = pd.DataFrame(results)
    results.to_pickle(results_dir / f"{run_id}.pkl")


@app.command()
def grid(target: Target = Target.multiphase, replicates: int = 50):
    settings = pd.read_csv("data/CV_gridsearch_2021-01-29.csv", index_col=0)
    print(settings.head())
    settings = settings.loc[
        :, ("max_depth", "min_samples_leaf", "max_features")
    ]
    settings = settings.astype(int)

    for idx, row in tqdm(settings.iterrows()):
        args = row.to_dict()
        cv(target, replicates=replicates, progress=False, **args)


if __name__ == "__main__":
    app()
