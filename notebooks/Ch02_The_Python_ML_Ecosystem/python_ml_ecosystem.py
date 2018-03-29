
# coding: utf-8

# # Numpy Introduction
# ## numpy arrays

# In[91]:

import numpy as np
arr = np.array([1,3,4,5,6])
arr


# In[8]:

arr.shape


# In[9]:

arr.dtype


# In[10]:

arr = np.array([1,'st','er',3])
arr.dtype


# In[5]:

np.sum(arr)


# ### Creating arrays

# In[11]:

arr = np.array([[1,2,3],[2,4,6],[8,8,8]])
arr.shape


# In[12]:

arr


# In[13]:

arr = np.zeros((2,4))
arr


# In[14]:

arr = np.ones((2,4))
arr


# In[15]:

arr = np.identity(3)
arr


# In[16]:

arr = np.random.randn(3,4)
arr


# In[17]:

from io import BytesIO
b = BytesIO(b"2,23,33\n32,42,63.4\n35,77,12")
arr = np.genfromtxt(b, delimiter=",")
arr


# ### Accessing array elements
# #### Simple indexing

# In[18]:

arr[1]


# In[19]:

arr = np.arange(12).reshape(2,2,3)
arr


# In[20]:

arr[0]


# In[21]:

arr = np.arange(10)
arr[5:]


# In[22]:

arr[5:8]


# In[23]:

arr[:-5]


# In[24]:

arr = np.arange(12).reshape(2,2,3)
arr


# In[25]:

arr[1:2]


# In[26]:

arr = np.arange(27).reshape(3,3,3)
arr


# In[27]:

arr[:,:,2]


# In[28]:

arr[...,2]


# #### Advanced Indexing

# In[29]:

arr = np.arange(9).reshape(3,3)
arr


# In[30]:

arr[[0,1,2],[1,0,0]]


# ##### Boolean Indexing

# In[31]:

cities = np.array(["delhi","banglaore","mumbai","chennai","bhopal"])
city_data = np.random.randn(5,3)
city_data


# In[32]:

city_data[cities =="delhi"]


# In[33]:

city_data[city_data >0]


# In[34]:

city_data[city_data >0] = 0
city_data


# #### Operations on arrays

# In[35]:

arr = np.arange(15).reshape(3,5)
arr


# In[36]:

arr + 5


# In[37]:

arr * 2


# In[38]:

arr1 = np.arange(15).reshape(5,3)
arr2 = np.arange(5).reshape(5,1)
arr2 + arr1


# In[39]:

arr1


# In[40]:

arr2


# In[41]:

arr1 = np.random.randn(5,3)
arr1


# In[42]:

np.modf(arr1)


# #### Linear algebra using numpy

# In[43]:

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[9,8,7],[6,5,4],[1,2,3]])
A.dot(B)


# In[44]:

A = np.arange(15).reshape(3,5)
A.T


# In[45]:

np.linalg.svd(A)


# In[46]:

a = np.array([[7,5,-3], [3,-5,2],[5,3,-7]])
b = np.array([16,-8,0])
x = np.linalg.solve(a, b)
x


# In[47]:

np.allclose(np.dot(a, x), b)


# # Pandas
# ## Data frames

# In[48]:

import pandas as pd
d =  [{'city':'Delhi',"data":1000},
      {'city':'Banglaore',"data":2000},
      {'city':'Mumbai',"data":1000}]
pd.DataFrame(d)


# In[49]:

df = pd.DataFrame(d)


# ### Reading in data

# In[92]:

city_data = pd.read_csv(filepath_or_buffer='simplemaps-worldcities-basic.csv')


# In[93]:

city_data.head(n=10)


# In[55]:

city_data.tail()


# In[56]:

series_es = city_data.lat


# In[57]:

type(series_es)


# In[58]:

series_es[1:10:2]


# In[59]:

series_es[:7]


# In[60]:

series_es[:-7315]


# In[61]:

city_data[:7]


# In[62]:

city_data.iloc[:5,:4]


# In[63]:

city_data[city_data['pop'] > 10000000][city_data.columns[pd.Series(city_data.columns).str.startswith('l')]]


# In[64]:

city_greater_10mil = city_data[city_data['pop'] > 10000000]
city_greater_10mil.rename(columns={'pop':'population'}, inplace=True)
city_greater_10mil.where(city_greater_10mil.population > 15000000)


# In[65]:

df = pd.DataFrame(np.random.randn(8, 3),
columns=['A', 'B', 'C'])


# ### Operations on dataframes

# In[66]:

nparray = df.values
type(nparray)


# In[67]:

from numpy import nan
df.iloc[4,2] = nan


# In[68]:

df


# In[69]:

df.fillna(0)


# In[70]:

columns_numeric = ['lat','lng','pop']


# In[71]:

city_data[columns_numeric].mean()


# In[72]:

city_data[columns_numeric].sum()


# In[73]:

city_data[columns_numeric].count()


# In[74]:

city_data[columns_numeric].median()


# In[75]:

city_data[columns_numeric].quantile(0.8)


# In[76]:

city_data[columns_numeric].sum(axis = 1).head()


# In[77]:

city_data[columns_numeric].describe()


# In[78]:

city_data1 = city_data.sample(3)


# ### Concatanating data frames

# In[79]:

city_data2 = city_data.sample(3)
city_data_combine = pd.concat([city_data1,city_data2])
city_data_combine


# In[80]:

df1 = pd.DataFrame({'col1': ['col10', 'col11', 'col12', 'col13'],
                    'col2': ['col20', 'col21', 'col22', 'col23'],
                    'col3': ['col30', 'col31', 'col32', 'col33'],
                    'col4': ['col40', 'col41', 'col42', 'col43']},
                   index=[0, 1, 2, 3])


# In[81]:

df1


# In[82]:

df4 = pd.DataFrame({'col2': ['col22', 'col23', 'col26', 'col27'],
                    'Col4': ['Col42', 'Col43', 'Col46', 'Col47'],
                    'col6': ['col62', 'col63', 'col66', 'col67']},
                   index=[2, 3, 6, 7])

pd.concat([df1,df4], axis=1)


# In[83]:

country_data = city_data[['iso3','country']].drop_duplicates()


# In[84]:

country_data.shape


# In[85]:

country_data.head()


# In[86]:

del(city_data['country'])


# In[87]:

city_data.merge(country_data, 'inner').head()


# # Scikit-learn

# In[94]:

from sklearn import datasets
diabetes = datasets.load_diabetes()
X = diabetes.data[:10]
y = diabetes.target


# In[95]:

X[:5]


# In[96]:

y[:10]


# In[97]:

feature_names=['age', 'sex', 'bmi', 'bp',
               's1', 's2', 's3', 's4', 's5', 's6']


# ## Scikit example regression

# In[98]:

from sklearn import datasets
from sklearn.linear_model import Lasso

from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

diabetes = datasets.load_diabetes()
X_train = diabetes.data[:310]
y_train = diabetes.target[:310]

X_test = diabetes.data[310:]
y_test = diabetes.target[310:]

lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

scores = list()
scores_std = list()

estimator = GridSearchCV(lasso,
                         param_grid = dict(alpha=alphas))

estimator.fit(X_train, y_train)


# In[99]:

estimator.best_score_


# In[100]:

estimator.best_estimator_


# In[101]:

estimator.predict(X_test)


# ## Deep Learning Frameworks

# ### Theano example 

# In[1]:

import numpy
import theano.tensor as T
from theano import function
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y


# In[2]:

f = function([x, y], z)
f(8, 2)


# ### Tensorflow example

# In[102]:

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


# ### Building a neural network model with Keras

# In[103]:

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train = cancer.data[:340]
y_train = cancer.target[:340]

X_test = cancer.data[340:]
y_test = cancer.target[340:]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[150]:

model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[151]:

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# In[152]:

model.fit(X_train, y_train,
          epochs=20,
          batch_size=50)


# In[153]:

predictions = model.predict_classes(X_test)


# In[154]:

from sklearn import metrics

print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))


# ### The power of deep learning models

# In[155]:

model = Sequential()
model.add(Dense(15, input_dim=30, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=20,
          batch_size=50)


# In[156]:

predictions = model.predict_classes(X_test)


# In[157]:

print('Accuracy:', metrics.accuracy_score(y_true=y_test, y_pred=predictions))
print(metrics.classification_report(y_true=y_test, y_pred=predictions))

