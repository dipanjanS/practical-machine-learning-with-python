
# coding: utf-8

# # Bike Sharing Dataset Exploratory Analysis
# 
# + Based on Bike Sharing dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# + This notebook is based upon the hourly data file, i.e. hour.csv
# 
# ---
# Reference:
# Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg,

# ## Import required packages

# In[1]:


# data manipulation 
import numpy as np
import pandas as pd

# plotting
import seaborn as sn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# setting params
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

sn.set_style('whitegrid')
sn.set_context('talk')

plt.rcParams.update(params)
pd.options.display.max_colwidth = 600

# pandas display data frames as tables
from IPython.display import display, HTML


# ## Load Dataset

# In[2]:


hour_df = pd.read_csv('hour.csv')
print("Shape of dataset::{}".format(hour_df.shape))


# In[3]:


display(hour_df.head())


# ## Data Types and Summary Stats

# In[4]:


# data types of attributes
hour_df.dtypes


# In[5]:


# dataset summary stats
hour_df.describe()


# The dataset has:
# + 17 attributes in total and 17k+ records
# + Except dtedat, rest all are numeric(int or float)
# + As stated on the UCI dataset page, the following attributes have been normalized (same is confirmed above):
#     + temp, atemp
#     + humidity
#     + windspeed
# + Dataset has many categorical variables like season, yr, holiday, weathersit and so on. These will need to handled with care

# ## Standardize Attribute Names

# In[6]:


hour_df.rename(columns={'instant':'rec_id',
                      'dteday':'datetime',
                      'holiday':'is_holiday',
                      'workingday':'is_workingday',
                      'weathersit':'weather_condition',
                      'hum':'humidity',
                      'mnth':'month',
                      'cnt':'total_count',
                      'hr':'hour',
                      'yr':'year'},inplace=True)


# ## Typecast Attributes 

# In[7]:


# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')


# ## Visualize Attributes, Trends and Relationships

# ### Hourly distribution of Total Counts
# + Seasons are encoded as 1:spring, 2:summer, 3:fall, 4:winter
# + Exercise: Convert season names to readable strings and visualize data again

# In[8]:


fig,ax = plt.subplots()
sn.pointplot(data=hour_df[['hour',
                           'total_count',
                           'season']],
             x='hour',y='total_count',
             hue='season',ax=ax)
ax.set(title="Season wise hourly distribution of counts")


# + The above plot shows peaks around 8am and 5pm (office hours)
# + Overall higher usage in the second half of the day

# In[9]:


fig,ax = plt.subplots()
sn.pointplot(data=hour_df[['hour','total_count','weekday']],x='hour',y='total_count',hue='weekday',ax=ax)
ax.set(title="Weekday wise hourly distribution of counts")


# + Weekends (0 and 6) and Weekdays (1-5) show different usage trends with weekend's peak usage in during afternoon hours
# + Weekdays follow the overall trend, similar to one visualized in the previous plot
# + Weekdays have higher usage as compared to weekends
# + It would be interesting to see the trends for casual and registered users separately

# In[10]:


fig,ax = plt.subplots()
sn.boxplot(data=hour_df[['hour','total_count']],x="hour",y="total_count",ax=ax)
ax.set(title="Box Pot for hourly distribution of counts")


# + Early hours (0-4) and late nights (21-23) have low counts but significant outliers
# + Afternoon hours also have outliers
# + Peak hours have higher medians and overall counts with virtually no outliers

# ### Monthly distribution of Total Counts

# In[11]:


fig,ax = plt.subplots()
sn.barplot(data=hour_df[['month',
                         'total_count']],
           x="month",y="total_count")
ax.set(title="Monthly distribution of counts")


# + Months June-Oct have highest counts. Fall seems to be favorite time of the year to use cycles

# In[12]:


df_col_list = ['month','weekday','total_count']
plot_col_list= ['month','total_count']
spring_df = hour_df[hour_df.season==1][df_col_list]
summer_df = hour_df[hour_df.season==2][df_col_list]
fall_df = hour_df[hour_df.season==3][df_col_list]
winter_df = hour_df[hour_df.season==4][df_col_list]

fig,ax= plt.subplots(nrows=2,ncols=2)
sn.barplot(data=spring_df[plot_col_list],x="month",y="total_count",ax=ax[0][0],)
ax[0][0].set(title="Spring")

sn.barplot(data=summer_df[plot_col_list],x="month",y="total_count",ax=ax[0][1])
ax[0][1].set(title="Summer")

sn.barplot(data=fall_df[plot_col_list],x="month",y="total_count",ax=ax[1][0])
ax[1][0].set(title="Fall")

sn.barplot(data=winter_df[plot_col_list],x="month",y="total_count",ax=ax[1][1])  
ax[1][1].set(title="Winter")


# ### Year Wise Count Distributions

# In[13]:


sn.violinplot(data=hour_df[['year',
                            'total_count']],
              x="year",y="total_count")


# + Both years have multimodal distributions
# + 2011 has lower counts overall with a lower median
# + 2012 has a higher max count though the peaks are around 100 and 300 which is then tapering off

# ### Working Day Vs Holiday Distribution

# In[14]:


fig,(ax1,ax2) = plt.subplots(ncols=2)
sn.barplot(data=hour_df,x='is_holiday',y='total_count',hue='season',ax=ax1)
sn.barplot(data=hour_df,x='is_workingday',y='total_count',hue='season',ax=ax2)


# ### Outliers

# In[15]:


fig,(ax1,ax2)= plt.subplots(ncols=2)
sn.boxplot(data=hour_df[['total_count',
                         'casual','registered']],ax=ax1)
sn.boxplot(data=hour_df[['temp','windspeed']],ax=ax2)


# ### Correlations

# In[16]:


corrMatt = hour_df[["temp","atemp",
                    "humidity","windspeed",
                    "casual","registered",
                    "total_count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
sn.heatmap(corrMatt, mask=mask,
           vmax=.8, square=True,annot=True)


# + Correlation between temp and atemp is very high (as expected)
# + Same is te case with registered-total_count and casual-total_count
# + Windspeed to humidity has negative correlation
# + Overall correlational statistics are not very high.
