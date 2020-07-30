import pandas as pd
import numpy as np
import sklearn import preprocessing

def OneHotEncoder_preprocessor(data,columns):
    return data

def LabelEncoder_preprocessor(data,columns):
    """LabelEncoder_preprocessor function take data and target columns list as perameter

    Args:
        data (pandas.DataFrame): DataFrame of data
        columns (list): list of target columns 

    Returns:
        DataFrame: data with replaced column with encodings
    """
    for column in columns:
        # fill na with new category as none
        data.loc[:,column] = data[column].fillna('None')
        # initialize LabelEncoder
        lbl_enc = preprocessing.LabelEncoder()
        # fit label encoder and transform values on column
        # P.S do not use this directly .fit first, then transform
        df.loc[:,column] = lbl_enc.fit_transform(data[column].values)
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