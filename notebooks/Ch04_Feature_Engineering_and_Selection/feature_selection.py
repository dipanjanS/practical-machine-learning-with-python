
# coding: utf-8
"""
Created on Mon May 17 00:00:00 2017

@author: DIP
"""

# # Import necessary dependencies and settings

# In[1]:

import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)
pt = np.get_printoptions()['threshold']


# # Threshold based methods

# ## Limiting features in bag of word based models

# In[2]:

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0.1, max_df=0.85, max_features=2000)
cv


# ## Variance based thresholding

# In[3]:

df = pd.read_csv('datasets/Pokemon.csv')
poke_gen = pd.get_dummies(df['Generation'])
poke_gen.head()


# In[4]:

from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=.15)
vt.fit(poke_gen)


# In[5]:

pd.DataFrame({'variance': vt.variances_,
              'select_feature': vt.get_support()},
            index=poke_gen.columns).T


# In[6]:

poke_gen_subset = poke_gen.iloc[:,vt.get_support()].head()
poke_gen_subset


# # Statistical Methods

# In[7]:

from sklearn.datasets import load_breast_cancer

bc_data = load_breast_cancer()
bc_features = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_classes = pd.DataFrame(bc_data.target, columns=['IsMalignant'])

# build featureset and response class labels 
bc_X = np.array(bc_features)
bc_y = np.array(bc_classes).T[0]
print('Feature set shape:', bc_X.shape)
print('Response class shape:', bc_y.shape)


# In[8]:

np.set_printoptions(threshold=30)
print('Feature set data [shape: '+str(bc_X.shape)+']')
print(np.round(bc_X, 2), '\n')
print('Feature names:')
print(np.array(bc_features.columns), '\n')
print('Predictor Class label data [shape: '+str(bc_y.shape)+']')
print(bc_y, '\n')
print('Predictor name:', np.array(bc_classes.columns))
np.set_printoptions(threshold=pt)


# In[9]:

from sklearn.feature_selection import chi2, SelectKBest

skb = SelectKBest(score_func=chi2, k=15)
skb.fit(bc_X, bc_y)


# In[10]:

feature_scores = [(item, score) for item, score in zip(bc_data.feature_names, skb.scores_)]
sorted(feature_scores, key=lambda x: -x[1])[:10]


# In[11]:

select_features_kbest = skb.get_support()
feature_names_kbest = bc_data.feature_names[select_features_kbest]
feature_subset_df = bc_features[feature_names_kbest]
bc_SX = np.array(feature_subset_df)
print(bc_SX.shape)
print(feature_names_kbest)


# In[12]:

np.round(feature_subset_df.iloc[20:25], 2)


# In[13]:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# build logistic regression model
lr = LogisticRegression()

# evaluating accuracy for model built on full featureset
full_feat_acc = np.average(cross_val_score(lr, bc_X, bc_y, scoring='accuracy', cv=5))
# evaluating accuracy for model built on selected featureset
sel_feat_acc = np.average(cross_val_score(lr, bc_SX, bc_y, scoring='accuracy', cv=5))

print('Model accuracy statistics with 5-fold cross validation')
print('Model accuracy with complete feature set', bc_X.shape, ':', full_feat_acc)
print('Model accuracy with selected feature set', bc_SX.shape, ':', sel_feat_acc)


# # Recursive Feature Elimination

# In[14]:

from sklearn.feature_selection import RFE

lr = LogisticRegression()
rfe = RFE(estimator=lr, n_features_to_select=15, step=1)
rfe.fit(bc_X, bc_y)


# In[15]:

select_features_rfe = rfe.get_support()
feature_names_rfe = bc_data.feature_names[select_features_rfe]
print(feature_names_rfe)


# In[16]:

set(feature_names_kbest) & set(feature_names_rfe)


# # Model based selection

# In[17]:

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(bc_X, bc_y)


# In[18]:

importance_scores = rfc.feature_importances_
feature_importances = [(feature, score) for feature, score in zip(bc_data.feature_names, importance_scores)]
sorted(feature_importances, key=lambda x: -x[1])[:10]


# # Feature extraction using dimensionality reduction

# In[19]:

# center the feature set
bc_XC = bc_X - bc_X.mean(axis=0)

# decompose using SVD
U, S, VT = np.linalg.svd(bc_XC)

# get principal components
PC = VT.T

# get first 3 principal components
PC3 = PC[:, 0:3]
PC3.shape


# In[20]:

# reduce feature set dimensionality 
np.round(bc_XC.dot(PC3), 2)


# In[21]:

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(bc_X)


# In[22]:

pca.explained_variance_ratio_


# In[23]:

bc_pca = pca.transform(bc_X)
np.round(bc_pca, 2)


# In[24]:

np.average(cross_val_score(lr, bc_pca, bc_y, scoring='accuracy', cv=5))

