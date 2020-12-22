"""
This code will plot the ranked feature importance for the HEA data set with 90% CFI.
It uses the Importances_df_NPCA to grab the average, Upper Confidence Bound
and Lower Confidence Bound of every Magpie/Matminer feature. It then reorders the dataframe and plots
the feature importances based on their rank order.  The Magpie identity of the features is not
carried over here, feature 105 is the Random Feature.
"""

import matplotlib.pyplot as plt
import pandas as pd

n_components = 5
target = "Multiphase"

Importances_df_NPCA = pd.read_csv(
    f"data/processed/Feature_{target}_importances_no_PCA_with_{n_components}_components.csv"
)

Importances_df_NPCA = Importances_df_NPCA.sort_values(by="mean", ascending=False)
Importances_df_NPCA = Importances_df_NPCA.reset_index()
Importances_df_NPCA["Rank Order"] = Importances_df_NPCA.index
Random_Feature_Row = Importances_df_NPCA[Importances_df_NPCA["index"] == 105].index[0]

plt.plot(Importances_df_NPCA["Rank Order"], Importances_df_NPCA["mean"])
plt.fill_between(
    Importances_df_NPCA["Rank Order"],
    (Importances_df_NPCA["ci95_lo_NPCA"]),
    (Importances_df_NPCA["ci95_hi_NPCA"]),
    alpha=0.1,
)
plt.vlines(Random_Feature_Row, ymin=0, ymax=0.08)
plt.ylabel("RF Feature Importance")
plt.xlabel("Rank Order of Importance")

plt.savefig(f"reports/figures/{target}_Non_PCA_random_features_4.png", dpi=300)
