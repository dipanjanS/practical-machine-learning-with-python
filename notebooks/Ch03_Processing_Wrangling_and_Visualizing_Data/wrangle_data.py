# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:51:47 2017

@author: Raghav Bali
"""

"""

This script showcases data wrangling tasks 

``Execute``
        $ python wrangle_data.py

"""

import random
import datetime 
import numpy as np
import pandas as pd
from random import randrange
from sklearn import preprocessing



def _random_date(start,date_count):
    """This function generates a random date based on params
    Args:
        start (date object): the base date
        date_count (int): number of dates to be generated
    Returns:
        list of random dates

    """
    current = start
    while date_count > 0:
        curr = current + datetime.timedelta(days=randrange(42))
        yield curr
        date_count-=1

  
def generate_sample_data(row_count=100):
    """This function generates a random transaction dataset
    Args:
        row_count (int): number of rows for the dataframe
    Returns:
        a pandas dataframe

    """
    
    # sentinels
    startDate = datetime.datetime(2016, 1, 1,13)
    serial_number_sentinel = 1000
    user_id_sentinel = 5001
    product_id_sentinel = 101
    price_sentinel = 2000
    
    
    # base list of attributes
    data_dict = {
    'Serial No': np.arange(row_count)+serial_number_sentinel,
    'Date': np.random.permutation(pd.to_datetime([x.strftime("%d-%m-%Y") 
                                                    for x in _random_date(startDate,
                                                                          row_count)]).date
                                  ),
    'User ID': np.random.permutation(np.random.randint(0,
                                                       row_count,
                                                       size=int(row_count/10)) + user_id_sentinel).tolist()*10,
    'Product ID': np.random.permutation(np.random.randint(0,
                                                          row_count,
                                                          size=int(row_count/10))+ product_id_sentinel).tolist()*10 ,
    'Quantity Purchased': np.random.permutation(np.random.randint(1,
                                                                  42,
                                                                  size=row_count)),
    'Price': np.round(np.abs(np.random.randn(row_count)+1)*price_sentinel,
                      decimals=2),
    'User Type':np.random.permutation([chr(random.randrange(97, 97 + 3 + 1)) 
                                            for i in range(row_count)])
    }
    
    # introduce missing values
    for index in range(int(np.sqrt(row_count))): 
        data_dict['Price'][np.argmax(data_dict['Price'] == random.choice(data_dict['Price']))] = np.nan
        data_dict['User Type'][np.argmax(data_dict['User Type'] == random.choice(data_dict['User Type']))] = np.nan
        data_dict['Date'][np.argmax(data_dict['Date'] == random.choice(data_dict['Date']))] = np.nan
        data_dict['Product ID'][np.argmax(data_dict['Product ID'] == random.choice(data_dict['Product ID']))] = 0
        data_dict['Serial No'][np.argmax(data_dict['Serial No'] == random.choice(data_dict['Serial No']))] = -1
        data_dict['User ID'][np.argmax(data_dict['User ID'] == random.choice(data_dict['User ID']))] = -101
        
    
    # create data frame
    df = pd.DataFrame(data_dict)
    
    return df
    

def describe_dataframe(df=pd.DataFrame()):
    """This function generates descriptive stats of a dataframe
    Args:
        df (dataframe): the dataframe to be analyzed
    Returns:
        None

    """
    print("\n\n")
    print("*"*30)
    print("About the Data")
    print("*"*30)
    
    print("Number of rows::",df.shape[0])
    print("Number of columns::",df.shape[1])
    print("\n")
    
    print("Column Names::",df.columns.values.tolist())
    print("\n")
    
    print("Column Data Types::\n",df.dtypes)
    print("\n")
    
    print("Columns with Missing Values::",df.columns[df.isnull().any()].tolist())
    print("\n")
    
    print("Number of rows with Missing Values::",len(pd.isnull(df).any(1).nonzero()[0].tolist()))
    print("\n")
    
    print("Sample Indices with missing data::",pd.isnull(df).any(1).nonzero()[0].tolist()[0:5])
    print("\n")
    
    print("General Stats::")
    print(df.info())
    print("\n")
    
    print("Summary Stats::")
    print(df.describe())
    print("\n")
    
    print("Dataframe Sample Rows::")
    print(df.head(5))
    
def cleanup_column_names(df,rename_dict={},do_inplace=True):
    """This function renames columns of a pandas dataframe
       It converts column names to snake case if rename_dict is not passed. 
    Args:
        rename_dict (dict): keys represent old column names and values point to 
                            newer ones
        do_inplace (bool): flag to update existing dataframe or return a new one
    Returns:
        pandas dataframe if do_inplace is set to False, None otherwise

    """
    if not rename_dict:
        return df.rename(columns={col: col.lower().replace(' ','_') 
                    for col in df.columns.values.tolist()}, 
                  inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict,inplace=do_inplace)

def expand_user_type(u_type):
    """This function maps user types to user classes
    Args:
        u_type (str): user type value
    Returns:
        (str) user_class value

    """
    if u_type in ['a','b']:
        return 'new'
    elif u_type == 'c':
        return 'existing'
    elif u_type == 'd':
        return 'loyal_existing'
    else:
        return 'error'
            
            
if __name__=='__main__':
    df = generate_sample_data(row_count=1000)
    describe_dataframe(df)
    
    print("\n\n")
    print("*"*30)
    print("Rename Columns")
    print("*"*30)
    cleanup_column_names(df)
    print(df.head())
    print("\n\n")
    
    
    print("*"*30)
    print("Sorting Rows")
    print("*"*30)
    print(df.sort_values(['serial_no', 'price'], 
                         ascending=[True, False]).head())
    print("\n\n")
    
    
    print("*"*30)
    print("Rearranging Columns")
    print("*"*30)
    print(df[['serial_no','date','user_id','user_type',
              'product_id','quantity_purchased','price']].head())
    print("\n\n")
    
    
    print("*"*30)
    print("Filtering Columns")
    print("*"*30)
    print("Using Column Index::")
    print(df.iloc[:,3].values)
    # or df[[3]].values.tolist()
    print("\n")
    
    print("Using Column Name::")
    print(df.quantity_purchased.values)
    # or df['quantity_purchased'].values.tolist()
    print("\n")
    
    print("Using Column Data Type::")
    print(df.select_dtypes(include=['float64']).values[:,0])
    # or use exclude to remove certain datatype columns 
    print("\n\n")
    
    
    print("*"*30)
    print("Subsetting Rows")
    print("*"*30)
    print("Select Specific row indices::")
    print(df.iloc[[10,501,20]])
    print("\n")
    
    print("Excluding Specific Row indices::")
    print(df.drop([0,24,51], axis=0).head())
    print("\n")
    
    print("Subsetting based on logical condition(s)::")
    print(df[df.quantity_purchased>25].head())
    print("\n")
    
    print("Subsetting based on offset from top (bottom)::")
    print(df[100:].head()) 
    print("\n")
    
    
    print("*"*30)
    print("TypeCasting/Data Type Conversion")
    print("*"*30)
    df['date'] = pd.to_datetime(df.date)
    print(df.dtypes)
    
    print("*"*30)
    print("Apply / Map")
    print("*"*30)
    
    
    df['user_class'] = df['user_type'].map(expand_user_type)
    # or applymap
    print("Create a new user_class attribute::")
    print(df.head())
    print("\n")
    
    print("Get Attribute ranges::")
    print(df.select_dtypes(include=[np.number]).apply(lambda x: 
                                                        x.max()- x.min()))
    
    print("Get Week from date::")
    df['purchase_week'] = df[['date']].applymap(lambda dt:dt.week 
                                                if not pd.isnull(dt.week) 
                                                else 0)
    print(df.head())
    print("\n")
    
    
    print("*"*30)
    print("Handling Missing Values")
    print("*"*30)
    
    print("Drop Rows with missing dates::")
    df_dropped = df.dropna(subset=['date'])
    print(df_dropped.head())
    print("\n")
    
    print("Fill Missing Price values with mean price::")
    df_dropped['price'].fillna(value=np.round(df.price.mean(),decimals=2),
                                inplace=True)
    
    print("Fill Missing user_type values with value from \
             previous row (forward fill) ::")
    df_dropped['user_type'].fillna(method='ffill',inplace=True)
    
    print("Fill Missing user_type values with value from \
            next row (backward fill) ::")
    df_dropped['user_type'].fillna(method='bfill',inplace=True)

    print("\n")
    
    print("Drop Duplicate serial_no rows::\n")
    print("Duplicate sample::")
    print(df_dropped[df_dropped.duplicated(subset=['serial_no'])].head())
    df_dropped.drop_duplicates(subset=['serial_no'],inplace=True)
    print("After cleanup::")
    print(df_dropped.head())
    print("\n")
    
    print("Remove rows which have less than 3 attributes with non-missing data::")
    print(df.dropna(thresh=3).head())
    print("\n")
    
    print("*"*30)
    print("Encoding Categorical Variables")
    print("*"*30)
    print(pd.get_dummies(df,columns=['user_type']).head())
    print("\n")
    
    type_map={'a':0,'b':1,'c':2,'d':3,np.NAN:-1}
    df['encoded_user_type'] = df.user_type.map(type_map)
    print((df.head()))
    print("\n")
    
    print("*"*30)
    print("Random Sampling data from DataFrame")
    print("*"*30)
    print(df.sample(frac=0.2, replace=True, random_state=42).head())
    print("\n")
    
    print("*"*30)
    print("Normalizing Numeric Data")
    print("*"*30)
    
    print("Min-Max Scaler::")
    df_normalized = df.dropna().copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df_normalized['price'].values.reshape(-1,1))
    df_normalized['price'] = np_scaled.reshape(-1,1)
    print(df_normalized.head())
    print("\n")
    
    print("Robust Scaler::")
    df_normalized = df.dropna().copy()
    robust_scaler = preprocessing.RobustScaler()
    rs_scaled = robust_scaler.fit_transform(df_normalized['quantity_purchased'].values.reshape(-1,1))
    df_normalized['quantity_purchased'] = rs_scaled.reshape(-1,1)
    print(df_normalized.head())
    print("\n")
    
    print("*"*30)
    print("Data Summarization")
    print("*"*30)
    
    print("Aggregates based on condition::")
    print(df['price'][df['user_type']=='a'].mean())
    print("\n")
    
    print("Row Counts on condition::")
    print(df['purchase_week'].value_counts())
    print("\n")
    
    print("GroupBy attributes::")
    print(df.groupby(['user_class'])['quantity_purchased'].sum())
    print("\n")
    
    print("GroupBy with different aggregates::")
    print(df.groupby(['user_class'])['quantity_purchased'].agg([np.sum,
                                                                np.mean,
                                                                np.count_nonzero]))
    print("\n")                                                            
                                                                
    print("GroupBy with specific agg for each attribute::")
    print(df.groupby(['user_class','user_type']).agg({'price':np.mean,
                                                        'quantity_purchased':np.max}))
    print("\n")

    print("GroupBy with multiple agg for each attribute::")
    print(df.groupby(['user_class','user_type']).agg({'price':{
                                                                'total_price':np.sum,
                                                                'mean_price':np.mean,
                                                                'variance_price':np.std,
                                                                'count':np.count_nonzero},
                                                   'quantity_purchased':np.sum}))  
                                                                
    print("\n")
    
    
    print("Pivot tables::")
    print(df.pivot_table(index='date', columns='user_type', 
                         values='price',aggfunc=np.mean))
    print("\n")      

    print("Stacking::")
    print(df.stack())
    print("\n")               
                                                             
                                                            
    
    
    
    
    
    
    
    
    
    
    
    
    