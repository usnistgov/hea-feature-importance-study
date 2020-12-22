""" This runs the code to generate the feature importances for the PCA version of study """

from __future__ import annotations

import pickle

import pandas as pd
from sklearn import decomposition, ensemble, pipeline, preprocessing
from sklearn.compose import ColumnTransformer

from src.data import load_hea_dataset, split_chemical_systems
from src.features import randcat

target = "Multiphase"
target_key = f"PROPERTY: {target}"


def setup_pipeline(predictor, n_components=5, n_random_features=1):
    """partial PCA pipeline with random features concatenated to the projected data

    intented for use with src.features.randcat, as in
    ```
    n_components, n_random_features = 5, 5
    Xrand = randcat(X, n_random_features=n_random_features)
    model = setup_pipeline(predictor, n_components=n_components, n_random_features=n_random_features)
    ```
    """

    # exclude the last n_random_features from the PCA
    # use negative indices to avoid explicitly needing to know the input dimension
    rand_indices = list(range(-n_random_features, 0, 1))

    ct = ColumnTransformer(
        [
            (
                "pca",
                decomposition.PCA(n_components=n_components),
                slice(0, -n_random_features),
            ),
            ("pass", "passthrough", rand_indices),
        ]
    )

    model = pipeline.Pipeline(
        [
            ("standardize", preprocessing.StandardScaler()),
            ("partial_pca", ct),
            ("predictor", predictor),
        ]
    )

    return model


if __name__ == "__main__":
    df, X = load_hea_dataset(subset="train")
    y = df[target_key].values

    # dict with keys == number of components, dataframe of shape (reps, components)
    importances_dict = {}

    feature_names = {}
    comp_range = range(5, 100, 5)
    rep_range = range(0, 50)

    train_auc = pd.DataFrame(
        index=list(rep_range), columns=list(comp_range)
    )  # shape (reps, tested num of components)
    val_auc = pd.DataFrame(index=list(rep_range), columns=list(comp_range))
    for n_components in comp_range:
        importances_df = pd.DataFrame()
        training_stats = pd.DataFrame()
        all_AUC_Scores = pd.DataFrame()
        feature_importances = pd.DataFrame()
        for rep in rep_range:

            n_random_features = 1

            predictor = ensemble.RandomForestClassifier(
                n_estimators=144,
                max_depth=30,
                min_samples_leaf=1,
                n_jobs=4,
                class_weight="balanced",
            )

            model = setup_pipeline(
                predictor,
                n_components=n_components,
                n_random_features=n_random_features,
            )

            train, val = split_chemical_systems(df)
            Xrand = randcat(X, n_random_features=n_random_features)
            model.fit(Xrand[train], y[train])

            rf = model.named_steps["predictor"]
            feature_names = [f"PC{n}" for n in range(1, n_components + 1)] + [
                f"Random{n}" for n in range(1, n_random_features + 1)
            ]
            feature_importances = pd.DataFrame(
                rf.feature_importances_.reshape(-1, 1).T,
                index=[rep],
                columns=feature_names,
            )
            importances_df = pd.concat([importances_df, feature_importances], axis=0)

            # train_scores = model.predict(Xrand[train])
            # train_auc.loc[rep,n_components] = roc_auc_score( y_dib[train], train_scores)

            # val_scores=model.predict(Xrand[val])
            # val_auc.loc[rep,n_components] = roc_auc_score(y_dib[val],val_scores)
            importances_dict[n_components] = importances_df
            # The next two lines of code do not function properly
            # holdout_scores = model.predict(randcat(X[id_acta]))
            # holdout_AUC=roc_auc_score(y_acta, holdout_scores)

    with open("data/processed/pca_importances_dict.pkl", "wb") as f:
        pickle.dump(importances_dict, f)
