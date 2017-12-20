# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:51:47 2017

@author: Raghav Bali
"""

"""

This script showcases methods to read XML type data using:
    + powerful xml library
    + xml to dict

``Execute``
        $ python read_xml.py

"""

import xml.etree.ElementTree as ET
import xmltodict



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
          
def print_xml_tree(xml_root,indent_level=0):
    """This function prints a nested dict object
    Args:
        xml_root (dict): the xml tree to be printed
        indent_level (int): the indentation level for nesting
    Returns:
        None

    """
    for child in xml_root:
            print("{0}tag:{1}, attribute:{2}".format(
                                                "\t"*indent_level,
                                                child.tag,
                                                child.attrib))
                                                
            print("{0}tag data:{1}".format("\t"*indent_level,
                                            child.text))
                                            
            print_xml_tree(child,indent_level=indent_level+1)
            


def read_xml(file_name):
    """This function extracts and prints XML content from a given file
    Args:
        file_name (str): file path to be read
    Returns:
        None

    """
    try:
        tree = ET.parse(file_name)
        root = tree.getroot()
        
        print("Root tag:{0}".format(root.tag))
        print("Attributes of Root:: {0}".format(root.attrib))
        
        print_xml_tree(root)
            
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except Exception:
        raise

    

def read_xml2dict_xml(file_name):
    """This function extracts and prints xml content from a file using xml2dict
    Args:
        file_name (str): file path to be read
    Returns:
        None

    """
    try:
        xml_filedata = open(file_name).read() 
        ordered_dict = xmltodict.parse(xml_filedata)
        
        print_nested_dicts(ordered_dict)
    except IOError:
        raise IOError("File path incorrect/ File not found")
    except ValueError:
        ValueError("XML file has errors")
    except Exception:
        raise    
    
if __name__=='__main__':
    print("\n\n")
    print("*"*30)
    print("Contents of sample xml file:")
    print("*"*30)
    read_xml(r'sample_xml.xml')
    
    print("\n\n")
    print("*"*30)
    print("Contents of a xml file using xml2dict:")
    print("*"*30)
    read_xml2dict_xml(r'sample_xml.xml')