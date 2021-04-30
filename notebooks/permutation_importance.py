import streamlit as st

import re
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
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

    rank_file = str(path).replace("AUC", "Feature")
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
st.header("train/val performance vs random feature rank")

clip = st.sidebar.selectbox("clip?", ["no", "yes"])
stat = st.sidebar.selectbox("statistic", ["mean", "std", "min", "max"])

perf = []
rf_rank = []
for (idx, row), r_path in zip(df.iterrows(), results):
    runs, importance = load_results(r_path)
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    perf.append(mean.to_dict())

    if clip:
        importance = np.clip(importance, 0, np.inf)

    # get feature rankings by sorting negative values
    # to rank highest feature first
    rank = np.argsort(-importance, axis=1)
    rf_rank.append(rank.iloc[:, -1].values)

perf = pd.DataFrame(perf)
rf_rank = pd.DataFrame(rf_rank)

fig, ax = plt.subplots()
c = {
    "min": rf_rank.min(axis=1),
    "max": rf_rank.max(axis=1),
    "mean": rf_rank.mean(axis=1),
    "std": rf_rank.std(axis=1),
}[stat]

p = ax.scatter(perf.train_NPCA_auc, perf.val_NPCA_auc, c=c)
plt.colorbar(p)
plt.xlabel("train AUC")
plt.ylabel("val AUC")
plt.tight_layout()

# ax.scatter(df.max_depth, df.min_leaf_sample, c=perf.train_precision)
st.pyplot(fig)

# perf.shape

fig, ax = plt.subplots()
y = rf_rank.mean(axis=1)
yerr = (y - rf_rank.min(axis=1), rf_rank.max(axis=1) - y)
p = ax.scatter(perf.train_NPCA_auc, c)
p = ax.scatter(perf.train_NPCA_auc, y)
ax.errorbar(
    perf.train_NPCA_auc,
    y,
    yerr=yerr,
    color="k",
    linestyle="none",
)
plt.xlabel("train AUC")
plt.ylabel("RF rank")
plt.tight_layout()

# ax.scatter(df.max_depth, df.min_leaf_sample, c=perf.train_precision)
st.pyplot(fig)

# box plot of random feature ranks
# groups/runs sorted by increasing train AUC
order = np.argsort(perf.train_NPCA_auc)
_rank = rf_rank.iloc[order]
_rank = _rank.reset_index(drop=True).T

# rearrange to make one seaborn boxplot per column
# see https://stackoverflow.com/a/46134162
_rankdata = _rank.melt(var_name="run", value_name="RF Rank")

fig, ax = plt.subplots(figsize=(16, 4))
sns.boxplot(x="run", y="RF Rank", data=_rankdata)
plt.plot(_rank.mean(axis=0), color="k")
st.pyplot(fig)

score = _rank / 105
# for s in score:
#     stats.beta

st.write(score.shape)

s = score.iloc[:, -5]
pars = stats.beta.fit(s, floc=0, fscale=1)
a, b, loc, scal = pars
st.write(pars)
rv = stats.beta(a, b)

st.write(score.shape)
fig, ax = plt.subplots(figsize=(16, 4))
x = np.arange(0.001, 0.999, 0.001)
# ax.plot(x, rv.pdf(x), "k-", lw=2, label="frozen pdf")
ax.plot(x, stats.beta.pdf(x, a, b))
st.write(rv.pdf(x))
ax.hist(s)
st.pyplot(fig)
