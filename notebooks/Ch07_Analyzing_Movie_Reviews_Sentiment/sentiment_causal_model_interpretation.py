
# coding: utf-8

# # Import necessary dependencies

# In[1]:

import pandas as pd
import numpy as np
import text_normalizer as tn


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


# # Build Text Classification Pipeline with The Best Model

# In[3]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")

# build BOW features on train reviews
cv = CountVectorizer(binary=False, min_df=0.0, max_df=1.0, ngram_range=(1,2))
cv_train_features = cv.fit_transform(norm_train_reviews)
# build Logistic Regression model
lr = LogisticRegression()
lr.fit(cv_train_features, train_sentiments)

# Build Text Classification Pipeline
lr_pipeline = make_pipeline(cv, lr)

# save the list of prediction classes (positive, negative)
classes = list(lr_pipeline.classes_)


# # Analyze Model Prediction Probabilities

# In[4]:

lr_pipeline.predict(['the lord of the rings is an excellent movie', 
                     'i hated the recent movie on tv, it was so bad'])


# In[5]:

pd.DataFrame(lr_pipeline.predict_proba(['the lord of the rings is an excellent movie', 
                     'i hated the recent movie on tv, it was so bad']), columns=classes)


# # Interpreting Model Decisions

# In[6]:

from skater.core.local_interpretation.lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=classes)
def interpret_classification_model_prediction(doc_index, norm_corpus, corpus, 
                                              prediction_labels, explainer_obj):
    # display model prediction and actual sentiments
    print("Test document index: {index}\nActual sentiment: {actual}\nPredicted sentiment: {predicted}"
      .format(index=doc_index, actual=prediction_labels[doc_index],
              predicted=lr_pipeline.predict([norm_corpus[doc_index]])))
    # display actual review content
    print("\nReview:", corpus[doc_index])
    # display prediction probabilities
    print("\nModel Prediction Probabilities:")
    for probs in zip(classes, lr_pipeline.predict_proba([norm_corpus[doc_index]])[0]):
        print(probs)
    # display model prediction interpretation
    exp = explainer.explain_instance(norm_corpus[doc_index], 
                                     lr_pipeline.predict_proba, num_features=10, 
                                     labels=[1])
    exp.show_in_notebook()


# In[7]:

doc_index = 100 
interpret_classification_model_prediction(doc_index=doc_index, norm_corpus=norm_test_reviews,
                                         corpus=test_reviews, prediction_labels=test_sentiments,
                                         explainer_obj=explainer)


# In[8]:

doc_index = 2000
interpret_classification_model_prediction(doc_index=doc_index, norm_corpus=norm_test_reviews,
                                         corpus=test_reviews, prediction_labels=test_sentiments,
                                         explainer_obj=explainer)


# In[9]:

doc_index = 347 
interpret_classification_model_prediction(doc_index=doc_index, norm_corpus=norm_test_reviews,
                                         corpus=test_reviews, prediction_labels=test_sentiments,
                                         explainer_obj=explainer)

