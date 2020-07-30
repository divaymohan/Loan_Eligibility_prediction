import pandas as pd
import numpy as np
import sklearn import preprocessing

def OneHotEncoder_preprocessor(data,columns):
    """[summary]

    Args:
        data ([type]): [description]
        columns ([type]): [description]
        sparse (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    for column in columns:
        # fill na with new category as none
        data.loc[:,column] = data[column].fillna('None')
    # initialize LabelEncoder
    ohe = preprocessing.OneHotEncoder()
    # fit label encoder and transform values on column
    ohe.fit(data[columns])
    # P.S do not use this directly .fit first, then transform
    data.loc[:,columns] = ohe.transform(data[columns])
    return data

def LabelEncoder_preprocessor(data,columns):
    """LabelEncoder_preprocessor function take data and target columns list as perameter

    Args:
        data (pandas.DataFrame): DataFrame of data
        columns (list): list of target columns 

    Returns:
        DataFrame: data with replaced column with encodings
    Algoriths:
        Decision Tree
        Random Forest
        Extra Trees
        XGBoost
        GBM
        LightGBM
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
    """[summary]

    Args:
        data ([type]): [description]
        columns ([type]): [description]

    Returns:
        [type]: [description]
    Algorithms:
        Support machine
        linear model
        neural model
    """
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