import parse
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt

import pymoo
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort


def parse_filename(path: Path):
    """Load model hyperparameters from metrics filename.

    e.g. "Multiphase_max_depth=3_min_samples_leaf=1_max_features=2.pkl"
    """
    settings = parse.parse(
        "{}_max_depth={:d}_min_samples_leaf={:d}_max_features={:d}.pkl",
        path.name,
    )
    keys = "target", "max_depth", "min_samples_leaf", "max_features"
    return dict(zip(keys, settings))


# dataset = "multiphase"
# results_dir = Path("data/shapley_results")

dataset = "mg"
results_dir = Path("data/glass_shapley_results")

figure_key = f"{dataset}_shapley_importance"


figure_dir = Path("reports/ranking-figures")
# results_dir = Path("data/shapley_results_v3")
results = results_dir.glob("*.pkl")

r = st.sidebar.selectbox("run", list(results))
# r = list(results)[0]

st.write(parse_filename(r))

st.write(f"results for run {r}")
r = pd.read_pickle(r)
r

s = r.pop("shapley_val_importance")
replicate_id = st.sidebar.slider("replicate_id", 0, len(s))
run = s[replicate_id]
# run = s[3]
fig, ax = plt.subplots()
plt.plot(np.sort(run)[::-1])
plt.axhline(run[-1])
st.pyplot(fig)

# s = r.pop("impurity_importance")
importance = np.vstack([row[:106] for row in s])
st.write(f"importance {importance.shape}")
st.write(importance)

shap_rank = stats.rankdata(-importance, axis=1, method="ordinal")
st.write("rank")
st.write(shap_rank)


fig, ax = plt.subplots()
plt.scatter(
    r.val_precision,
    r.val_recall,
    c=stats.rankdata(-importance, axis=-1)[:, -1],
)
plt.colorbar()
st.pyplot(fig)

st.header("train/val performance vs random feature rank")

stat = st.sidebar.selectbox("statistic", ["mean", "std", "min", "max"])
importance = st.sidebar.selectbox("importance", ["Shapley", "impurity"])

perf = []
rf_rank = {"impurity": [], "shapley": [], "avgimpurity": []}

pars = []
for r in results_dir.glob("*.pkl"):
    pars.append(parse_filename(r))
    runs = pd.read_pickle(r)
    impurity_importance = runs.pop("impurity_importance")
    # shapley_importance = np.vstack(runs.pop("shapley_train_importance"))
    shapley_importance = np.vstack(runs.pop("shapley_val_importance"))

    mean, std = runs.mean(axis=0), runs.std(axis=0)
    perf.append(mean.to_dict())

    # get feature rankings
    # to rank highest feature first
    # s_rank = np.argsort(np.vstack(shapley_importance.values), axis=0)[
    #     :, ::-1
    # ].argsort(axis=0)

    # s_rank = s_rank.max(axis=1)[:, np.newaxis] - s_rank
    s_rank = stats.rankdata(-shapley_importance, axis=1, method="ordinal")
    rf_rank["shapley"].append(s_rank[:, -1])

    # i_rank = np.argsort(np.vstack(impurity_importance.values), axis=0)[
    #     :, ::-1
    # ].argsort(axis=0)
    i_rank = stats.rankdata(-np.vstack(impurity_importance.values), axis=1)
    rf_rank["impurity"].append(i_rank[:, -1])

    # average over runs
    i_avg = np.vstack(impurity_importance.values).mean(axis=0)
    i_rank_avg = stats.rankdata(-i_avg, method="ordinal")
    rf_rank["avgimpurity"].append(i_rank_avg[-1])

st.write("shapley", shapley_importance)
st.write("rank shape", s_rank.shape)
st.write("rank", s_rank)
st.write("rf rank", s_rank[:, -1])

df = pd.DataFrame(pars)
perf = pd.DataFrame(perf)
shapley_rank = pd.DataFrame(rf_rank["shapley"])
impurity_rank = pd.DataFrame(rf_rank["impurity"])
impurity_avg = rf_rank["avgimpurity"]
st.write("rf_rank", shapley_rank.shape)

shapley_rank

if importance == "Shapley":
    rf_rank = shapley_rank
elif importance == "impurity":
    rf_rank = impurity_rank

figure_key = f"{dataset}_{importance}_importance"

fig, ax = plt.subplots()
c = {
    "min": rf_rank.min(axis=1),
    "max": rf_rank.max(axis=1),
    "mean": rf_rank.mean(axis=1),
    "std": rf_rank.std(axis=1),
}[stat]

p = ax.scatter(perf.train_AUC, perf.val_AUC, c=c)
lims = ax.get_ylim()
plt.plot([0.65, 1.0], [0.65, 1.0])
plt.ylim(*lims)
plt.colorbar(p)
plt.xlabel(r"$AUC_{train}$")
plt.ylabel(r"$AUC_{val}$")
plt.tight_layout()
plt.savefig(
    figure_dir / f"{figure_key}_CV_AUC_scatter.tiff",
    bbox_inches="tight",
    dpi=600,
)
plt.tight_layout()

# ax.scatter(df.max_depth, df.min_leaf_sample, c=perf.train_precision)
st.pyplot(fig)


fig, ax = plt.subplots()
y = rf_rank.mean(axis=1)
yerr = (y - rf_rank.min(axis=1), rf_rank.max(axis=1) - y)
p = ax.scatter(perf.train_AUC, c)
p = ax.scatter(perf.train_AUC, y)
ax.errorbar(
    perf.train_AUC,
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
sort_by = st.sidebar.selectbox("sort", ["train_AUC", "val_AUC"])

# box plot of random feature ranks
# groups/runs sorted by increasing train AUC
order = np.argsort(perf[sort_by])
_rank = rf_rank.iloc[order]
_rank = _rank.reset_index(drop=True).T

# rearrange to make one seaborn boxplot per column
# see https://stackoverflow.com/a/46134162
_rankdata = _rank.melt(var_name="run", value_name="RF Rank")

fig, ax = plt.subplots(figsize=(16, 4))
sns.boxplot(x="run", y="RF Rank", data=_rankdata)
plt.plot(_rank.mean(axis=0), color="k")
plt.tight_layout()
plt.savefig(
    figure_dir / f"{figure_key}_boxplot.tiff",
    bbox_inches="tight",
    dpi=600,
)
st.pyplot(fig)

st.write("rank(average)")
fig, ax = plt.subplots()
p = ax.scatter(perf.train_AUC, impurity_avg)
plt.xlabel("train AUC")
plt.ylabel("RF rank")
plt.tight_layout()
st.pyplot(fig)


fig, ax = plt.subplots()
p = ax.scatter(perf.train_AUC, perf.val_AUC, c=impurity_avg)
plt.colorbar(p, label="random feature\nimpurity rank")
plt.xlabel("train AUC")
plt.ylabel("val AUC")
plt.tight_layout()
st.pyplot(fig)


score = _rank / 105
# for s in score:
#     stats.beta

st.write(score.shape)

s = score.iloc[:, -5]
pars = stats.beta.fit(s, floc=0, fscale=1)
a, b, loc, scale = pars
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

st.markdown("## impurity scatter")


AUC_gap = perf.train_AUC - perf.val_AUC


def perfplot(x, y):
    # X = np.c_[AUC_gap.values, -perf.val_AUC.values]
    X = np.c_[x, y]
    V = (X - X.mean(0)) / (X.max(0) - X.min(0))

    nds = NonDominatedSorting().do(
        V, epsilon=0.0, only_non_dominated_front=False
    )
    sel = nds[0]

    pf = pd.DataFrame(X[nds[0]])
    pf = pf.sort_values(by=0)
    plt.plot(pf[0], -pf[1], color="r", linestyle="--")


fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(16, 5))
plt.sca(axes[0])
perfplot(perf.train_AUC.values, -perf.val_AUC.values)
p = plt.scatter(perf.train_AUC, perf.val_AUC, c=impurity_avg)
plt.colorbar(p, label="random feature\nimpurity rank")
plt.xlabel(r"$AUC_{train}$")
plt.ylabel(r"$AUC_{val}$")
xlims, ylims = axes[0].get_xlim(), axes[0].get_ylim()
plt.plot([0, 1], [0, 1], "k--")
axes[0].set_xlim(*xlims)
axes[0].set_ylim(*ylims)

plt.sca(axes[1])
perfplot(AUC_gap.values, -perf.val_AUC.values)
p = plt.scatter(perf.train_AUC - perf.val_AUC, perf.val_AUC, c=impurity_avg)
plt.colorbar(p, label="random feature\nimpurity rank")
plt.xlabel(r"$AUC_{train} - AUC_{val}$")
plt.sca(axes[2])
p = plt.scatter(impurity_avg, perf.val_AUC, c=perf.train_AUC - perf.val_AUC)
plt.colorbar(p, label=r"$AUC_{train} - AUC_{val}$")
plt.xlabel("random feature\nimpurity rank")
plt.tight_layout()
plt.savefig(
    figure_dir / f"{dataset}_impurity_importance_comparison.tiff",
    bbox_inches="tight",
    dpi=600,
)
st.pyplot(fig)


st.markdown("## non-aggregated")
rf_rank = {"impurity": [], "shapley": [], "avgimpurity": []}

pars = []
perf_all = []
for r in results_dir.glob("*.pkl"):
    pars.append(parse_filename(r))
    runs = pd.read_pickle(r)

    impurity_importance = runs.pop("impurity_importance")
    # shapley_importance = np.vstack(runs.pop("shapley_train_importance"))
    shapley_importance = np.vstack(runs.pop("shapley_val_importance"))

    perf_all.append(runs)

    s_rank = stats.rankdata(-shapley_importance, axis=1, method="ordinal")
    rf_rank["shapley"].append(s_rank[:, -1])

    # i_rank = np.argsort(np.vstack(impurity_importance.values), axis=0)[
    #     :, ::-1
    # ].argsort(axis=0)
    i_rank = stats.rankdata(-np.vstack(impurity_importance.values), axis=1)
    rf_rank["impurity"].append(i_rank[:, -1])

    # average over runs
    i_avg = np.vstack(impurity_importance.values).mean(axis=0)
    i_rank_avg = stats.rankdata(-i_avg, method="ordinal")
    rf_rank["avgimpurity"].append(i_rank_avg[-1])

# df = pd.DataFrame(pars)
perf = pd.concat(perf_all)
shapley_rank = pd.DataFrame(rf_rank["shapley"])
impurity_rank = pd.DataFrame(rf_rank["impurity"])
impurity_avg = rf_rank["avgimpurity"]

st.write(perf.shape)
st.write(shapley_rank.shape)

fig, axes = plt.subplots()
plt.scatter(
    perf.train_AUC, perf.val_AUC, c=impurity_rank.values.flatten(), alpha=0.5
)
plt.colorbar(label="random feature\nimpurity rank")
plt.xlabel(r"$AUC_{train}$")
plt.ylabel(r"$AUC_{val}$")
plt.tight_layout()
plt.savefig(
    figure_dir / f"{dataset}_impurity_importance_CV_AUC_all_scatter.tiff",
    bbox_inches="tight",
    dpi=600,
)
st.pyplot(fig)

fig, axes = plt.subplots()
plt.scatter(
    perf.train_AUC, perf.val_AUC, c=shapley_rank.values.flatten(), alpha=0.5
)
plt.colorbar(label="random feature\nShapley rank")
plt.xlabel(r"$AUC_{train}$")
plt.ylabel(r"$AUC_{val}$")
plt.tight_layout()
plt.savefig(
    figure_dir / f"{dataset}_shapley_importance_CV_AUC_all_scatter.tiff",
    bbox_inches="tight",
    dpi=600,
)
st.pyplot(fig)


fig, axes = plt.subplots()
plt.scatter(perf.train_AUC, impurity_rank.values.flatten())
st.pyplot(fig)


fig, axes = plt.subplots()
plt.scatter(perf.val_AUC, impurity_rank.values.flatten())
st.pyplot(fig)


fig, axes = plt.subplots()
plt.scatter(perf.train_AUC - perf.val_AUC, impurity_rank.values.flatten())
st.pyplot(fig)
