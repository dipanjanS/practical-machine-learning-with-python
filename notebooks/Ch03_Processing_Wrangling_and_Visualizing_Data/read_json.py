# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:51:47 2017

@author: Raghav Bali
"""

"""

This script showcases methods to read JSON type data:
    + using python's inbuilt utilities
    + using pandas


``Execute``
        $ python read_json.py

"""

import json
import pandas as pd


def print_nested_dicts(nested_dict,indent_level=0):
    """This function prints a nested dict object
    Args:
        nested_dict (dict): the dictionary to be printed
        indent_level (int): the indentation level for nesting
    Returns:
        None

    """
    
    for key, val in nested_dict.items():
        if isinstance(val, dict):
          print("{0} : ".format(key))
          print_nested_dicts(val,indent_level=indent_level+1)
        elif isinstance(val,list):
            print("{0} : ".format(key))
            for rec in val:
                print_nested_dicts(rec,indent_level=indent_level+1)
        else:
          print("{0}{1} : {2}".format("\t"*indent_level,key, val))

def extract_json(file_name,do_print=True):
    """This function extracts and prints json content from a given file
    Args:
        file_name (str): file path to be read
        do_print (bool): boolean flag to print file contents or not

    Returns:
        None

    """
    try:
        json_filedata = open(file_name).read() 
        json_data = json.loads(json_filedata)
        
        if do_print:
            print_nested_dicts(json_data)
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except ValueError:
        ValueError("JSON file has errors")
    except Exception:
        raise

def extract_pandas_json(file_name,orientation="records",do_print=True):
    """This function extracts and prints json content from a file using pandas
       This is useful when json data represents tabular, series information
    Args:
        file_name (str): file path to be read
        orientation (str): orientation of json file. Defaults to records
        do_print (bool): boolean flag to print file contents or not

    Returns:
        None

    """
    try:
        df = pd.read_json(file_name,orient=orientation)
        
        if do_print:
            print(df)
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except ValueError:
        ValueError("JSON file has errors")
    except Exception:
        raise
    
    
    
if __name__=='__main__':
    print("\n\n")
    print("*"*30)
    print("Contents of sample json file:")
    print("*"*30)
    extract_json(r'sample_json.json')
    
    print("\n\n")
    print("*"*30)
    print("Contents of a json file using pandas:")
    print("*"*30)
    extract_pandas_json(r'pandas_json.json')