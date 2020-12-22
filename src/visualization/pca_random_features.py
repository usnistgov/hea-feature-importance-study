"""
    1.) Performs PCA on the standardized matminer features
    2.) identifies the number of PCA features needed to explain some proportion of the variance
    3.) Plots that number as a vertical line in graphs of Random Feature importance ranked against number of PCAs used in RF model
"""
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition, pipeline, preprocessing

from src.data import load_hea_dataset

if __name__ == "__main__":
    # load DIB data
    target = "Multiphase"
    target_key = f"PROPERTY: {target}"
    df, X = load_hea_dataset(subset="train")
    y = df[target_key].values

    # load precomputed feature importances
    with open("data/processed/pca_importances_dict.pkl", "rb") as f:
        importances_dict = pickle.load(f)

    model = pipeline.Pipeline(
        [
            ("standardize", preprocessing.StandardScaler()),
            ("PCA", decomposition.PCA(0.90, svd_solver="full")),
        ]
    )

    model.fit(X)
    pca_model = model.named_steps["PCA"]
    Num_comp_var = pca_model.n_components_
    explained_variance = pca_model.explained_variance_

    fig, ax = plt.subplots(2, 1, figsize=(14, 16), tight_layout=True)

    plt.sca(ax[0])

    rank_df1 = pd.DataFrame()
    for n_comp, df in importances_dict.items():
        rank_df = df.rank(axis=1, ascending=False).agg(["mean", "std"])
        plt.errorbar(
            int(n_comp),
            rank_df.loc["mean", "Random1"],
            rank_df.loc["std", "Random1"],
            fmt="ob",
        )
        rank_df1 = pd.concat([rank_df1, rank_df])
    plt.axvline(x=Num_comp_var, ymin=0, ymax=1)
    plt.plot([0, int(n_comp)], [0, int(n_comp)], "--k", alpha=0.5, label="Worst")
    plt.ylabel("Random feature absolute rank (lower is better)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.sca(ax[1])
    for n_comp, df in importances_dict.items():
        rank_df = df.rank(axis=1, pct=True, ascending=False).agg(["mean", "std"])
        plt.errorbar(
            int(n_comp),
            rank_df.loc["mean", "Random1"] * 100,
            rank_df.loc["std", "Random1"] * 100,
            fmt="ob",
        )
    plt.axvline(x=Num_comp_var, ymin=0, ymax=1)
    plt.xlabel("Number components")
    plt.ylabel("Random feature percentile rank (100 = worst)")
    plt.ylim([0, 110])
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"reports/figures/{target}_PCA_random_features_4.png", dpi=300)
