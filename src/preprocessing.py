from sklearn import preprocessing
import pandas as pd
import numpy as np
import create_folds

def preprocessing_train(filename):
    # read whole data
    data = pd.read_csv(filename)
    # split into train and test
    train = data[data['Loan_Status'] != -1].reset_index(drop=True)
    test = data[data['Loan_Status'] == -1].reset_index(drop=True)
    #create k folds
    create_folds.create_folds(train)

if __name__ == '__main__':
    preprocessing_train('../input/fulldata.csv')