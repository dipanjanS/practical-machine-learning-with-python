
# coding: utf-8

# # Import necessary dependencies

# In[9]:

import pandas as pd
import numpy as np
import text_normalizer as tn
import warnings

warnings.filterwarnings("ignore")


# # Load and normalize data

# In[2]:

dataset = pd.read_csv(r'movie_reviews.csv')

# take a peek at the data
print(dataset.head())
reviews = np.array(dataset['review'])
sentiments = np.array(dataset['sentiment'])

# build train and test datasets
train_reviews = reviews[:35000]
train_sentiments = sentiments[:35000]
test_reviews = reviews[35000:]
test_sentiments = sentiments[35000:]

# normalize datasets
norm_train_reviews = tn.normalize_corpus(train_reviews)
norm_test_reviews = tn.normalize_corpus(test_reviews)


# # Extract features from positive and negative reviews

# In[3]:

from sklearn.feature_extraction.text import TfidfVectorizer

# consolidate all normalized reviews
norm_reviews = norm_train_reviews+norm_test_reviews
# get tf-idf features for only positive reviews
positive_reviews = [review for review, sentiment in zip(norm_reviews, sentiments) if sentiment == 'positive']
ptvf = TfidfVectorizer(use_idf=True, min_df=0.05, max_df=0.95, ngram_range=(1,1), sublinear_tf=True)
ptvf_features = ptvf.fit_transform(positive_reviews)
# get tf-idf features for only negative reviews
negative_reviews = [review for review, sentiment in zip(norm_reviews, sentiments) if sentiment == 'negative']
ntvf = TfidfVectorizer(use_idf=True, min_df=0.05, max_df=0.95, ngram_range=(1,1), sublinear_tf=True)
ntvf_features = ntvf.fit_transform(negative_reviews)
# view feature set dimensions
print(ptvf_features.shape, ntvf_features.shape)


# # Topic Modeling on Reviews

# In[4]:

import pyLDAvis
import pyLDAvis.sklearn
from sklearn.decomposition import NMF
import topic_model_utils as tmu

pyLDAvis.enable_notebook()
total_topics = 10


# ## Display and visualize topics for positive reviews

# In[5]:

# build topic model on positive sentiment review features
pos_nmf = NMF(n_components=total_topics, 
          random_state=42, alpha=0.1, l1_ratio=0.2)
pos_nmf.fit(ptvf_features)      
# extract features and component weights
pos_feature_names = ptvf.get_feature_names()
pos_weights = pos_nmf.components_
# extract and display topics and their components
pos_topics = tmu.get_topics_terms_weights(pos_weights, pos_feature_names)
tmu.print_topics_udf(topics=pos_topics,
                 total_topics=total_topics,
                 num_terms=15,
                 display_weights=False)


# In[10]:

pyLDAvis.sklearn.prepare(pos_nmf, ptvf_features, ptvf, R=15)


# ## Display and visualize topics for negative reviews

# In[7]:

# build topic model on negative sentiment review features
neg_nmf = NMF(n_components=10, 
          random_state=42, alpha=0.1, l1_ratio=0.2)
neg_nmf.fit(ntvf_features)      
# extract features and component weights
neg_feature_names = ntvf.get_feature_names()
neg_weights = neg_nmf.components_
# extract and display topics and their components
neg_topics = tmu.get_topics_terms_weights(neg_weights, neg_feature_names)
tmu.print_topics_udf(topics=neg_topics,
                 total_topics=total_topics,
                 num_terms=15,
                 display_weights=False) 


# In[11]:

pyLDAvis.sklearn.prepare(neg_nmf, ntvf_features, ntvf, R=15)

