import streamlit as st

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

replicates = 25
results_dir = Path("data/permutation_importance")
results = list(filter(lambda s: "AUC" in str(s), results_dir.glob("*.csv")))

# st.write(list(map(str, results.glob("*.csv"))))


"""
Drop_Replicates_Permutation_Importances_AUCMultiphase_1_max_depth_1_min_leaf_sample_1max_features.csv
each column is a replicate
rows are: [train_NPCA_auc, val_NPCA_auc, validation_precision, validation_recall, train_precision, train_recall]

Drop_Replicates_Permutation_Importances_FeatureMultiphase_1_max_depth_1_min_leaf_sample_1max_features.csv
each row is a the feature (105 is the random feature), each column is a replicate
"""

prefix = "Drop_Replicates_Permutation_Importances_"


def parse_filename(path: Path, prefix=prefix):
    """extract task and hyperparameters from filename.

    hyperparameter values are integers that precede their names
    regex match multiple integers to get values
    split on same regex to get names
    """
    stem = str(path.stem).strip(prefix)
    keys = re.split(r"\d+", stem)
    keys = list(map(lambda s: s.strip("_"), keys))
    prefix, keys = keys[0], keys[1:]
    target = prefix.split("AUC")[-1]
    values = map(int, re.findall(r"\d+", stem))
    settings = dict(zip(keys, values))
    settings["target"] = target
    return settings


st.header("run summary:")
df = pd.DataFrame(map(parse_filename, results))
df


names = [
    "train_NPCA_auc",
    "val_NPCA_auc",
    "validation_precision",
    "validation_recall",
    "train_precision",
    "train_recall",
]


def load_results(path: Path):

    r = pd.read_csv(path, index_col=0)
    r = r.T
    r.columns = names

    stats = r.iloc[replicates:]
    runs = r.iloc[:replicates].reset_index(drop=True)

    rank_file = str(results[0]).replace("AUC", "Feature")
    r = pd.read_csv(rank_file, index_col=0)
    r = r.T
    importance_stats = r.iloc[replicates:]
    importance = r.iloc[:replicates].reset_index(drop=True)

    return runs, importance


st.header("perf for single setting:")
runs, importance = load_results(results[0])
df.iloc[0]
runs

st.header("permutation importance results")
importance
st.header("ok, now what.")

fig, ax = plt.subplots()
for r in results[:10]:
    runs, importance = load_results(r)
    ax.scatter(runs.validation_precision, runs.validation_recall)
st.pyplot(fig)


perf = []
for (idx, row), r_path in zip(df.iterrows(), results):
    runs, importance = load_results(r_path)
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    perf.append(mean.to_dict())
perf = pd.DataFrame(perf)

fig, ax = plt.subplots()
ax.scatter(df.max_depth, df.min_leaf_sample, c=perf.train_precision)
st.pyplot(fig)

# perf.shape
