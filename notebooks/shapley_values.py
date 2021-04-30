import shap

import streamlit as st

import math

import numpy as np
import pandas as pd
from sklearn import ensemble, pipeline, preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

import sys

sys.path.append(".")
from src.data import load_hea_dataset, split_chemical_systems
from src.features import randcat

target = "Multiphase"
target_key = f"PROPERTY: {target}"
n_random_features = 1

df, X = load_hea_dataset(subset="train")
y = df[target_key].values

df_acta, X_acta = load_hea_dataset(subset="test")
y_acta = df_acta[target_key].values

train, val = split_chemical_systems(df)
Xrand = randcat(X, n_random_features=n_random_features)

predictor = ensemble.RandomForestClassifier(
    n_estimators=144,
    max_depth=30,
    min_samples_leaf=1,
    n_jobs=4,
    class_weight="balanced",
)

model = pipeline.Pipeline(
    [
        ("standardize", preprocessing.StandardScaler()),
        ("predictor", predictor),
    ]
)

model.fit(Xrand[train], y[train])

rf = model.named_steps["predictor"]
feature_names = X.columns

feature_importances = rf.feature_importances_
feature_names = [f"Random{n}" for n in range(1, n_random_features + 1)]

train_scores = model.predict_proba(Xrand[train])[:, 1]
train_auc = roc_auc_score(y[train], train_scores)

val_scores = model.predict_proba(Xrand[val])[:, 1]
val_auc = roc_auc_score(y[val], val_scores)

# The next two lines of code do not function properly for BCC
holdout_scores = model.predict(randcat(X_acta, n_random_features=n_random_features))
holdout_AUC = roc_auc_score(y_acta, holdout_scores)

# ROC
fig, ax = plt.subplots(figsize=(4, 4))

fpr, tpr, thresholds = roc_curve(y[train], train_scores)
ax.plot(fpr, tpr)
fpr, tpr, thresholds = roc_curve(y[val], val_scores)
ax.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.tight_layout()
st.pyplot(fig)


## SHAP
# explainer = shap.Explainer(model.named_steps["predictor"])
explainer = shap.TreeExplainer(model.named_steps["predictor"])
# f = lambda x: model.named_steps["predictor"].predict_proba(x)[:, 1]
XX = model.named_steps["standardize"].transform(Xrand)
# explainer = shap.Explainer(f, XX)
shap_values = explainer(XX)
# just values for positive class
shap_values = shap_values[..., 1]
# visualize the first prediction's explanation
fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[val][0])
st.pyplot(fig)

fig, ax = plt.subplots()
# shap.plots.bar(shap_values[val])
shap.summary_plot(shap_values[val], XX[val])

st.pyplot(fig)


# feature importance ranking
fig, axes = plt.subplots(nrows=2, sharex=True)
ax = plt.sca(axes[0])
plt.plot(np.sort(feature_importances)[::-1])

# show the importance of the random feature
plt.axhline(feature_importances[-1], linestyle="--", color="k")
plt.ylabel("impurity\nimportance")

ax = plt.sca(axes[1])
aggregated_shap = np.abs(shap_values.data[val]).mean(axis=0)
plt.plot(np.sort(aggregated_shap)[::-1])
plt.axhline(aggregated_shap[-1], linestyle="--", color="k")

aggregated_shap = np.abs(shap_values.data[train]).mean(axis=0)
plt.plot(np.sort(aggregated_shap)[::-1])
plt.axhline(aggregated_shap[-1], linestyle="--", color="r")

plt.xlabel("feature rank")
plt.ylabel("avg |SHAP|")
plt.tight_layout()
st.pyplot(fig)
