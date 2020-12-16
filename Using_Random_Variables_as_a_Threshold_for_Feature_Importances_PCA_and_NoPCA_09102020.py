# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:39:24 2020

This is a mixture of Brian's code and my own.  Just not doing this in
Jupyter because it is more stupider.

Oddly, it seems as though the number of principal components doesn't
have a great impact on the validation/holdout predictions/stats.

I don't like that the AUC for the training stats is consistently 1.
That feels wrong somehow...

@author: jrh8
"""
#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import matminer
import matminer.datasets
from matminer import featurizers
import matminer.featurizers.composition
from matminer.featurizers.conversions import StrToComposition

from pymatgen import Composition

from sklearn import metrics
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import model_selection
from sklearn.decomposition.pca import PCA

from sklearn.compose import ColumnTransformer

#%%
'''setting up some basic composition handling and featurization functions to be used later'''
def featurize(df):
    """ run standard magpie, but drop space group number... """
    features = [
        "Number", "MendeleevNumber", "AtomicWeight", "MeltingT",
        "Column", "Row", "CovalentRadius", "Electronegativity",
        "NsValence", "NpValence", "NdValence", "NfValence", "NValence",
        "NsUnfilled", "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled",
        "GSvolume_pa", "GSbandgap", "GSmagmom"
    ]
    stats = ["mean", "avg_dev", "minimum", "maximum", "range"]
    magpie = featurizers.composition.ElementProperty('magpie', features, stats)
    
    # parse compositions from formula strings
    comp = StrToComposition()
    df = comp.featurize_dataframe(df, "Formula")
    
    # get the chemical system (i.e. a tuple of constituent species in alphabetical order)
    df['system'] = df['composition'].apply(lambda x: tuple(sorted(x.as_dict().keys())))
    
    # compute magpie featurization
    # assign to a throwaway dataframe, and slice out the feature columns
    _df = magpie.featurize_dataframe(df, 'composition')
    _df.head()
    features = _df.iloc[:,df.shape[1]:]
    
    return df, features
#%%
'''deciding what property to target, opening the datafile containing the training data, and pulling out the holdout set'''
target = 'PROPERTY: Multiphase'
target1 = target.split(" ")[1]
datafile = 'C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//HEA_database_additions.csv'
df = pd.read_csv(datafile)

df, X = featurize(df)
#%%
# split data by source
id_acta = df['REFERENCE: doi'] == '10.1016/j.actamat.2019.06.032'
X_dib = X.iloc[~id_acta.values]

y_dib = df.iloc[~id_acta.values][target].values
y_acta=df.iloc[id_acta.values][target].values

#%%
'''This defines the functions for random feature generation, the PCA pipeline, and the pipeline for doing feature importance
and RF model building on the original feature set'''

def randcat(X, n_random_features=1):
    """ append n_random_features columns sampled from independent standard normal distributions """
    N, D = X.shape
    return np.hstack((X, np.random.normal(size=(N,n_random_features))))

def split_chemical_systems(df, test_size=0.2, verbose=True):
    """ a group shuffle split cv iterator! 
    
    group rows that share a chemical system together across the train/val split
    """
    
    cv = model_selection.GroupShuffleSplit(n_splits=1, test_size=test_size)
    for train, val in cv.split(df[~id_acta], groups=df[~id_acta]['system']):
        if verbose:
            print(f'{train.size} training data')
            print(f'{val.size} validation data')
        
    return train, val

def setup_pipeline(predictor, n_components=5, n_random_features=1):

    # exclude the last n_random_features from the PCA
    # use negative indices to avoid explicitly needing to know the input dimension
    rand_indices = list(range(-n_random_features,0,1))
    
    ct = ColumnTransformer(
        [
            ("pca", decomposition.PCA(n_components=n_components), slice(0,-n_random_features)),
            ("pass", "passthrough", rand_indices)
        ]
    )

    model = pipeline.Pipeline(
        [
            ('standardize', preprocessing.StandardScaler()),
            ('partial_pca', ct),
            ('predictor', predictor)
        ]
    )
    
    return model

def setup_pipeline1(predictor, n_components=5, n_random_features=1):

    # exclude the last n_random_features from the PCA
    # use negative indices to avoid explicitly needing to know the input dimension
    rand_indices = list(range(-n_random_features,0,1))
    
    

    model = pipeline.Pipeline(
        [
            ('standardize', preprocessing.StandardScaler()),
            ('predictor', predictor)
        ]
    )
    
    return model

 





from sklearn.metrics import roc_auc_score
#%%
'''
This runs the code to generate the feature importances for the PCA version of study
'''
importances_dict = {} # dict with keys == number of components, dataframe of shape (reps, components)
feature_names = {}
comp_range = range(5,100,5)
rep_range = range(0,50)
train_auc = pd.DataFrame(index=list(rep_range),
                         columns=list(comp_range)) # shape (reps, tested num of components)
val_auc = pd.DataFrame(index=list(rep_range),
                         columns=list(comp_range))
for n_components in comp_range:
    Importances_df = pd.DataFrame()
    training_stats=pd.DataFrame()
    all_AUC_Scores=pd.DataFrame()
    feature_importances=pd.DataFrame()
    for rep in rep_range:
        ci95_hi = []
        ci95_lo = []
        
        n_random_features = 1
        
        train, val = split_chemical_systems(df)
        Xrand = randcat(X_dib, n_random_features=n_random_features)
        
        #need to use the same training stuff
        
        predictor = ensemble.RandomForestClassifier(n_estimators=144, max_depth=30, min_samples_leaf=1, n_jobs=4, class_weight='balanced')
        #predictor=ensemble.RandomForestClassifier(n_estimators=200, max_features='log2')
        model = setup_pipeline(predictor, n_components=n_components, n_random_features=n_random_features)
        model.fit(Xrand[train], y_dib[train]); 
        
        
        rf=model.named_steps['predictor']
        feature_names = [f"PC{n}" for n in range(1,n_components+1)] \
            +[f"Random{n}" for n in range(1,n_random_features+1)]
        feature_importances = pd.DataFrame(rf.feature_importances_.reshape(-1,1).T,
                                           index=[rep],
                                           columns=feature_names)
        Importances_df=pd.concat([Importances_df, feature_importances], axis=0)
        
        
        
        #train_scores = model.predict(Xrand[train])
        #train_auc.loc[rep,n_components] = roc_auc_score( y_dib[train], train_scores)
        
        #val_scores=model.predict(Xrand[val])
        #val_auc.loc[rep,n_components] = roc_auc_score(y_dib[val],val_scores)
        importances_dict[str(n_components)] = Importances_df
        #The next two lines of code do not function properly
        #holdout_scores = model.predict(randcat(X[id_acta]))
        #holdout_AUC=roc_auc_score(y_acta, holdout_scores)
#%%

'''this runs without PCA '''
importances_dict = {} # dict with keys == number of components, dataframe of shape (reps, components)
for j in range(0,1):
    Importances_df_NPCA = pd.DataFrame()
    train_NPCAing_stats_NPCA=pd.DataFrame()
    all_AUC_Scores_NPCA=pd.DataFrame()
 
    for i in range(0,50):
        ci95_hi_NPCA = []
        ci95_lo_NPCA = []
        n_components = (j+1)*5
        n_random_features = 1
        
        train_NPCA, val_NPCA = split_chemical_systems(df)
        Xrand_NPCA = randcat(X_dib, n_random_features=n_random_features)
          
        predictor = ensemble.RandomForestClassifier(n_estimators=144, max_depth=30, min_samples_leaf=1, n_jobs=4, class_weight='balanced')
        #predictor_NPCA=ensemble.RandomForestClassifier(n_estimators=200, max_features='log2')
        model_NPCA = setup_pipeline1(predictor, n_components=n_components, n_random_features=n_random_features)
        model_NPCA.fit(Xrand_NPCA[train_NPCA], y_dib[train_NPCA]); 
                
        rf=model_NPCA.named_steps['predictor']
        feature_names = X.columns

        feature_importances = pd.DataFrame(rf.feature_importances_.reshape(-1,1).T,
                                           index=[i])
        feature_names = [f"Random{n}" for n in range(1,n_random_features+1)]

        Importances_df_NPCA=pd.concat([Importances_df_NPCA, feature_importances], axis=0)
        
        train_NPCA_scores = model_NPCA.predict(Xrand_NPCA[train_NPCA])
        train_NPCA_auc = roc_auc_score( y_dib[train_NPCA], train_NPCA_scores)
        
        val_NPCA_scores=model_NPCA.predict(Xrand_NPCA[val_NPCA])
        val_NPCA_auc=roc_auc_score(y_dib[val_NPCA],val_NPCA_scores)
        
        #The next two lines of code do not function properly for BCC
        holdout_scores = model_NPCA.predict(randcat(X[id_acta]))
        holdout_AUC=roc_auc_score(y_acta, holdout_scores)
        
        AUC1 = pd.DataFrame(np.array([train_NPCA_auc, val_NPCA_auc, holdout_AUC]))
        all_AUC_Scores_NPCA = pd.concat([all_AUC_Scores_NPCA, AUC1], axis=1)
        
        
    name2 = "C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//Feature" + target1 + "Importances no PCA with " + str((j+1)*5)+" Components.csv"
    name3 = "C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//AUC" + target1 +" scores no PCA with " +str((j+1)*5)+" Components.csv"
    
    

    
    '''prepare dataframes for a quick statistical breakdown'''
    Transpose_Importances_df_NPCA = Importances_df_NPCA
    Transpose_all_AUC_Scores_NPCA = all_AUC_Scores_NPCA.T
    '''take stats and then append to original dataframes'''
    
    Importances_stats = Transpose_Importances_df_NPCA.describe()
    Transpose_Importances_df_NPCA = pd.concat([Transpose_Importances_df_NPCA, Importances_stats])
    Importances_df_NPCA=Transpose_Importances_df_NPCA.T
    
    for i in Importances_stats.T.index:
        c,m,s,k,l,n,a,e = Importances_stats.T.loc[i]
        ci95_hi_NPCA.append(m + 1.96*s/math.sqrt(c))
        ci95_lo_NPCA.append(m - 1.96*s/math.sqrt(c))
    Importances_df_NPCA['ci95_hi_NPCA']=ci95_hi_NPCA
    Importances_df_NPCA['ci95_lo_NPCA']=ci95_lo_NPCA
    '''take stats and then append to original dataframes'''
    AUC_Stats = Transpose_all_AUC_Scores_NPCA.describe()
    Transpose_all_AUC_Scores_NPCA = pd.concat([Transpose_all_AUC_Scores_NPCA, AUC_Stats])
    
    all_AUC_Scores_NPCA=Transpose_all_AUC_Scores_NPCA.T
        
    Importances_df_NPCA.to_csv(name2)
    all_AUC_Scores_NPCA.to_csv(name3)
    
    ci95_hi_NPCA = []
    ci95_lo_NPCA = []
    
    for i in AUC_Stats.T.index:
        c,m,s,k,l,n,a,e = AUC_Stats.T.loc[i]
        ci95_hi_NPCA.append(m + 1.96*s/math.sqrt(c))
        ci95_lo_NPCA.append(m - 1.96*s/math.sqrt(c))
    all_AUC_Scores_NPCA['ci95_hi_NPCA']=ci95_hi_NPCA
    all_AUC_Scores_NPCA['ci95_lo_NPCA']=ci95_lo_NPCA     
    
    Importances_df_NPCA.to_csv(name2)
    all_AUC_Scores_NPCA.to_csv(name3)
    importances_dict[str(n_components)] = Importances_df


    
#%%

'''
This block of code:
    1.) Performs PCA on the standardized matminer features
    2.) identifies the number of PCA features needed to explain some proportion of the variance 
    3.) Plots that number as a vertical line in graphs of Random Feature importance ranked against number of PCAs used in RF model 
'''
rank_df1=pd.DataFrame()
scaler = preprocessing.StandardScaler()
scaler.fit(X_dib)
X_stand = scaler.transform(X_dib)
my_model = PCA(.90, svd_solver='full')
my_model.fit_transform(X_stand)
Num_comp_var = my_model.n_components_
explained_variance = my_model.explained_variance_


fig, ax = plt.subplots(2,1, figsize=(7,8), tight_layout=True)

plt.sca(ax[0])
for n_comp, df in importances_dict.items():
    rank_df = df.rank(axis=1,
                      ascending=False).agg(['mean','std'])
    plt.errorbar(int(n_comp),
                 rank_df.loc['mean','Random1'],
                 rank_df.loc['std','Random1'],
                 fmt='ob')
    rank_df1=pd.concat([rank_df1,rank_df])
plt.axvline(x=Num_comp_var, ymin=0, ymax=1)
plt.plot([0,int(n_comp)],[0,int(n_comp)],'--k',alpha=0.5, label='Worst')
plt.ylabel('Random feature absolute rank (lower is better)')
plt.grid(alpha=0.3);
plt.legend();

plt.sca(ax[1])
for n_comp, df in importances_dict.items():
    rank_df = df.rank(axis=1,
                      pct=True,
                      ascending=False).agg(['mean','std'])
    plt.errorbar(int(n_comp),
                 rank_df.loc['mean','Random1']*100,
                 rank_df.loc['std','Random1']*100,
                 fmt='ob')
plt.axvline(x=Num_comp_var, ymin=0, ymax=1)   
plt.xlabel('Number components'); plt.ylabel('Random feature percentile rank (100 = worst)');
plt.ylim([0,110])
plt.grid(alpha=0.3);

plt.savefig('C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//'+ target1 +'_PCA_random_features_4.png', dpi=300)    
 #%%
'''
This code will plot the ranked feature importance for the HEA data set with 90% CFI.
It uses the Importances_df_NPCA to grab the average, Upper Confidence Bound 
and Lower Confidence Bound of every Magpie/Matminer feature. It then reorders the dataframe and plots
the feature importances based on their rank order.  The Magpie identity of the features is not
carried over here, feature 105 is the Random Feature. 
'''

Importances_df_NPCA1 = Importances_df_NPCA.sort_values(by = 'mean', ascending = False)
Importances_df_NPCA1 = Importances_df_NPCA1.reset_index()
Importances_df_NPCA1["Rank Order"] = Importances_df_NPCA1.index 
Random_Feature_Row = Importances_df_NPCA1[Importances_df_NPCA1['index']==105].index[0]

plt.plot(Importances_df_NPCA1["Rank Order"], Importances_df_NPCA1['mean'])
plt.fill_between(Importances_df_NPCA1["Rank Order"], (Importances_df_NPCA1["ci95_lo_NPCA"]), (Importances_df_NPCA1["ci95_hi_NPCA"]), alpha =.1)
plt.vlines(Random_Feature_Row, ymin = 0, ymax=0.08)
plt.ylabel("RF Feature Importance")
plt.xlabel("Rank Order of Importance")
plt.savefig('C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//'+ target1 +'_Non_PCA_random_features_4.png', dpi=300)
 #%%
'''take saved files and pull out the relevant statistics, starting with the AUC stats'''

All_AUC_stat=pd.DataFrame()

for j in range(0,15):
    name = "C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//Brian Feature Intermetallic Importances in PCA with " + str((j+1)*5)+" Components.csv"
    name1 = "C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//Brian FCC Intermetallic scores in PCA with " +str((j+1)*5)+" Components.csv"
    
    FI_stat = pd.read_csv(name, header=0)
    AUC_stat = pd.read_csv(name1)
    
    All_AUC_stat = pd.concat([All_AUC_stat, pd.concat([AUC_stat['mean'],AUC_stat['std']], axis=1)],axis=0)
name1 = "C://Users//jrh8//Documents//Documents//Manuscripts//2020//Trust in AI//All Brian Intermetallic AUC scores.csv"
All_AUC_stat.to_csv(name1)