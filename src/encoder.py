import pandas as pd
import numpy as np

def OneHotEncoder_preprocessor(data,columns):
    return data

def LabelEncoder_preprocessor(data,columns):
    return data

def BinarizeEncoder_preprocessor(data,columns):
    return data
def mappingEncoder_preprocesor(data,column,mapping):
    """Mapping Encoder

    Args:
        data (pandas.DataFrame): Any Data Frame
        column (string): list of features present in the dataframe
        mapping (dict): Dictionary of label and its values

    Returns:
        pandas.DataFrame: return same dataframe with replacement of mapped column
    """
    data.loc[:,column] = data[column].map(mapping)
    return data