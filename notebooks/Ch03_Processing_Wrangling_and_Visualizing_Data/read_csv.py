# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:51:47 2017

@author: Raghav Bali
"""

"""

This script showcases methods to extract data from CSVs:
    + csv containing delimiter separated values
    + csv containing tabular data

using csv and pandas packages

``Execute``
        $ python read_csv.py

"""

import csv
import pandas as pd
from pprint import pprint

def print_basic_csv(file_name, delimiter=','):
    """This function extracts and prints csv content from given filename
       Details: https://docs.python.org/2/library/csv.html
    Args:
        file_name (str): file path to be read
        delimiter (str): delimiter used in csv. Default is comma (',')

    Returns:
        None

    """
    csv_rows = list()
    csv_attr_dict = dict()
    csv_reader = None

    # read csv
    csv_reader = csv.reader(open(file_name, 'r'), delimiter=delimiter)
        
    # iterate and extract data    
    for row in csv_reader:
        print(row)
        csv_rows.append(row)
    
    # prepare attribute lists
    for col in csv_rows[0]:
        csv_attr_dict[col]=list()
    
    # iterate and add data to attribute lists
    for row in csv_rows[1:]:
        csv_attr_dict['sno'].append(row[0])
        csv_attr_dict['fruit'].append(row[1])
        csv_attr_dict['color'].append(row[2])
        csv_attr_dict['price'].append(row[3])
    
    # print the result
    print("\n\n")
    print("CSV Attributes::")
    pprint(csv_attr_dict)
            


def print_tabular_data(file_name,delimiter=","):
    """This function extracts and prints tabular csv content from given filename
       Details: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
    Args:
        file_name (str): file path to be read
        delimiter (str): delimiter used in csv. Default is comma ('\t')

    Returns:
        None

    """
    df = pd.read_csv(file_name,sep=delimiter)
    print(df)
    
    
    
if __name__=='__main__':
    print("\n\n")
    print("*"*30)
    print("Contents of sample csv file:")
    print("*"*30)
    print_basic_csv(r'tabular_csv.csv')
    
    print("\n\n")
    print("*"*30)
    print("Contents of a tabular csv file:")
    print("*"*30)
    print_tabular_data(r'tabular_csv.csv')