
# coding: utf-8
"""
Created on Mon May 17 00:00:00 2017

@author: DIP
"""

# # Import necessary dependencies and settings

# In[1]:

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)


# # Load sample data of video views

# In[2]:

views = pd.DataFrame([1295., 25., 19000., 5., 1., 300.], columns=['views'])
views


# # Standard Scaler $\frac{x_i - \mu}{\sigma}$

# In[3]:

ss = StandardScaler()
views['zscore'] = ss.fit_transform(views[['views']])
views


# In[4]:

vw = np.array(views['views'])
(vw[0] - np.mean(vw)) / np.std(vw)


# # Min-Max Scaler $\frac{x_i - min(x)}{max(x) - min(x)}$

# In[5]:

mms = MinMaxScaler()
views['minmax'] = mms.fit_transform(views[['views']])
views


# In[6]:

(vw[0] - np.min(vw)) / (np.max(vw) - np.min(vw))


# # Robust Scaler $\frac{x_i - median(x)}{IQR_{(1,3)}(x)}$

# In[7]:

rs = RobustScaler()
views['robust'] = rs.fit_transform(views[['views']])
views


# In[8]:

quartiles = np.percentile(vw, (25., 75.))
iqr = quartiles[1] - quartiles[0]
(vw[0] - np.median(vw)) / iqr

