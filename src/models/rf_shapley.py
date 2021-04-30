import sys
from enum import Enum
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer
from sklearn import ensemble, pipeline, preprocessing
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import load_hea_dataset, split_chemical_systems
from src.features import randcat


class Target(str, Enum):
    multiphase = "Multiphase"
    intermetallic = "IM"


def main(
    target: Target = Target.multiphase,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    n_random_features: int = 1,
):
    target_key = f"PROPERTY: {target.value}"
    print(target_key)

    df, X = load_hea_dataset(subset="train")
    y = df[target_key].values

    df_acta, X_test = load_hea_dataset(subset="test")
    y_test = df_acta[target_key].values

    train, val = split_chemical_systems(df)
    Xrand = randcat(X, n_random_features=n_random_features)

    # feature standardization
    std = preprocessing.StandardScaler()
    std.fit(Xrand[train])
    Xrand = std.transform(Xrand)

    X_test = std.transform(randcat(X_test, n_random_features=n_random_features))

    model = ensemble.RandomForestClassifier(
        n_estimators=144,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=4,
        class_weight="balanced",
    )

    model.fit(Xrand[train], y[train])
    feature_importances = model.feature_importances_

    train_scores = model.predict_proba(Xrand[train])[:, 1]
    train_auc = roc_auc_score(y[train], train_scores)

    val_scores = model.predict_proba(Xrand[val])[:, 1]
    val_auc = roc_auc_score(y[val], val_scores)

    test_scores = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_scores)

    print(f"{train_auc=}")
    print(f"{val_auc=}")
    print(f"{test_auc=}")

    impurity_rank = np.argsort(feature_importances)[::-1]
    print("random feature impurity rank: ", impurity_rank[-1])

    # aggregated shapley importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(Xrand)
    shap_values = shap_values[..., 1]  # just values for positive class
    aggregated_shap = np.abs(shap_values.data[train]).mean(axis=0)
    shap_rank = np.argsort(aggregated_shap)[::-1]
    print("random feature shap rank: ", shap_rank[-1], aggregated_shap[-1])


if __name__ == "__main__":
    typer.run(main)
