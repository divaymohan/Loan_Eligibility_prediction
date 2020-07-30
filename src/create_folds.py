import pandas as pd
import numpy as np
from sklearn import model_selection

# this function can be used for regression problem
def create_folds(data):
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Struge's rule
    # i take the floor of the value, you can also just round it
    num_bins = np.floor(1 + np.log2(len(data)))

    # bin targets
    #data.loc[:, "bins"] = pd.cut(data["Loan_Status"], bins=num_bins, labels=False)
    y = data.Loan_Status.values
    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    # note that instead of targets, we use bins!
    for f, (trn_, val_) in enumerate(kf.split(X=data, y=y)):
        data.loc[val_, "kfolds"] = f
    data.to_csv("../input/train_folds.csv", index=False)

    # drop the bins column
    #data = data.drop("bins", axis=1)
    # return dataframe with folds
    return data


if __name__ == "__main__":
    # Training data is in a csv file called train.csv
    df = pd.read_csv("train.csv")

    # we create a new columns called kfolds and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    #kf = model_selection.kFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "Kfold"] = fold

    # save the new csv with kfold column
    df.to_csv("train_folds.csv", index=False)

