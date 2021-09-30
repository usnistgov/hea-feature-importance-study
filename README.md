# Aggressively optimizing validation statistics can degrade interpretability of data-driven materials models
J. Chem. Phys. 155, 054105 (2021) 10.1063/5.0050885[https://dx.doi.org/10.1063/5.0050885]

Katherine Lei,  Howie Joress, Nils Persson,  Jason R. Hattrick-Simpers, and  Brian DeCost

The principal dataset is https://github.com/CitrineInformatics/MPEA_dataset, which is included [here](data/Citrine_MPEA_dataset.csv).

We also analyzed the metallic glass data reported in [10.1038/npjcompumats.2016.28](https://doi.org/10.1038/npjcompumats.2016.28), which is included [here](data/glassdata.csv), and is also available via [matminer as glass_ternary_landolt](https://hackingmaterials.lbl.gov/matminer/dataset_summary.html#glass-ternary-landolt).

The cross-validation experiments can be reproduced as follows:

```python
ipython src/models/rf_pca_feature_importances.py
ipython src/models/rf_feature_importances.py

ipython src/visualization/pca_random_features.py
ipython src/visualization/ranked_feature_importances.py
```
