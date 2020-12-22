""" Random Forest feature importances (this runs without PCA) """
import math

import numpy as np
import pandas as pd
from sklearn import ensemble, pipeline, preprocessing
from sklearn.metrics import roc_auc_score

from src.data import load_hea_dataset, split_chemical_systems
from src.features import randcat

target = "Multiphase"
target_key = f"PROPERTY: {target}"


if __name__ == "__main__":
    df, X = load_hea_dataset(subset="train")
    y = df[target_key].values

    df_acta, X_acta = load_hea_dataset(subset="train")
    y_acta = df_acta[target_key].values

    # dict with keys == number of components, dataframe of shape (reps, components)
    importances_dict = {}

    for j in range(0, 1):
        Importances_df_NPCA = pd.DataFrame()
        train_NPCAing_stats_NPCA = pd.DataFrame()
        all_AUC_Scores_NPCA = pd.DataFrame()

        for i in range(0, 50):
            n_components = (j + 1) * 5
            n_random_features = 1

            train_NPCA, val_NPCA = split_chemical_systems(df)
            Xrand_NPCA = randcat(X, n_random_features=n_random_features)

            predictor = ensemble.RandomForestClassifier(
                n_estimators=144,
                max_depth=30,
                min_samples_leaf=1,
                n_jobs=4,
                class_weight="balanced",
            )

            model_NPCA = pipeline.Pipeline(
                [
                    ("standardize", preprocessing.StandardScaler()),
                    ("predictor", predictor),
                ]
            )

            model_NPCA.fit(Xrand_NPCA[train_NPCA], y[train_NPCA])

            rf = model_NPCA.named_steps["predictor"]
            feature_names = X.columns

            feature_importances = pd.DataFrame(
                rf.feature_importances_.reshape(-1, 1).T, index=[i]
            )
            feature_names = [f"Random{n}" for n in range(1, n_random_features + 1)]

            Importances_df_NPCA = pd.concat(
                [Importances_df_NPCA, feature_importances], axis=0
            )

            train_NPCA_scores = model_NPCA.predict(Xrand_NPCA[train_NPCA])
            train_NPCA_auc = roc_auc_score(y[train_NPCA], train_NPCA_scores)

            val_NPCA_scores = model_NPCA.predict(Xrand_NPCA[val_NPCA])
            val_NPCA_auc = roc_auc_score(y[val_NPCA], val_NPCA_scores)

            # The next two lines of code do not function properly for BCC
            holdout_scores = model_NPCA.predict(
                randcat(X_acta, n_random_features=n_random_features)
            )
            holdout_AUC = roc_auc_score(y_acta, holdout_scores)

            AUC1 = pd.DataFrame(np.array([train_NPCA_auc, val_NPCA_auc, holdout_AUC]))
            all_AUC_Scores_NPCA = pd.concat([all_AUC_Scores_NPCA, AUC1], axis=1)

        name2 = f"data/Feature_{target}_importances_no_PCA_with_{n_components}_components.csv"
        name3 = (
            f"data/AUC_{target}_importances_no_PCA_with_{n_components}_components.csv"
        )

        """prepare dataframes for a quick statistical breakdown"""
        ci95_hi_NPCA = []
        ci95_lo_NPCA = []

        Transpose_Importances_df_NPCA = Importances_df_NPCA
        Transpose_all_AUC_Scores_NPCA = all_AUC_Scores_NPCA.T
        """take stats and then append to original dataframes"""

        Importances_stats = Transpose_Importances_df_NPCA.describe()
        Transpose_Importances_df_NPCA = pd.concat(
            [Transpose_Importances_df_NPCA, Importances_stats]
        )
        Importances_df_NPCA = Transpose_Importances_df_NPCA.T

        for i in Importances_stats.T.index:
            c, m, s, k, l, n, a, e = Importances_stats.T.loc[i]
            ci95_hi_NPCA.append(m + 1.96 * s / math.sqrt(c))
            ci95_lo_NPCA.append(m - 1.96 * s / math.sqrt(c))
        Importances_df_NPCA["ci95_hi_NPCA"] = ci95_hi_NPCA
        Importances_df_NPCA["ci95_lo_NPCA"] = ci95_lo_NPCA
        """take stats and then append to original dataframes"""
        AUC_Stats = Transpose_all_AUC_Scores_NPCA.describe()
        Transpose_all_AUC_Scores_NPCA = pd.concat(
            [Transpose_all_AUC_Scores_NPCA, AUC_Stats]
        )

        all_AUC_Scores_NPCA = Transpose_all_AUC_Scores_NPCA.T

        Importances_df_NPCA.to_csv(name2)
        all_AUC_Scores_NPCA.to_csv(name3)

        ci95_hi_NPCA = []
        ci95_lo_NPCA = []

        for i in AUC_Stats.T.index:
            c, m, s, k, l, n, a, e = AUC_Stats.T.loc[i]
            ci95_hi_NPCA.append(m + 1.96 * s / math.sqrt(c))
            ci95_lo_NPCA.append(m - 1.96 * s / math.sqrt(c))
        all_AUC_Scores_NPCA["ci95_hi_NPCA"] = ci95_hi_NPCA
        all_AUC_Scores_NPCA["ci95_lo_NPCA"] = ci95_lo_NPCA

        Importances_df_NPCA.to_csv(name2)
        all_AUC_Scores_NPCA.to_csv(name3)
