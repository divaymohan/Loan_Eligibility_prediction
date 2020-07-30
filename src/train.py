import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import config
import argparse
import model_dispatcher
from sklearn import ensemble
import encoder


def run(fold, model):
    # read the training data with folds
    df = pd.read_csv(config.TRAINING_FILE)
    df = df.drop("Unnamed: 0", axis=1)
    # training data is where k fold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfolds != fold].reset_index(drop=True)

    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    df_train = encoder.OneHotEncoder_preprocessor(
        df_train,
        [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ],
    )
    df_valid = encoder.OneHotEncoder_preprocessor(
        df_valid,
        [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Loan_Amount_Term",
            "Credit_History",
            "Property_Area",
        ],
    )

    # drop the label column from dataframe and convert it to
    # a numpy array by using values.
    # target is label column in the dataframe
    x_train = df_train.drop("Loan_Status", axis=1)
    y_train = df_train.Loan_Status

    # similarly, for validation we have
    x_valid = df_valid.drop("Loan_Status", axis=1)
    y_valid = df_valid.Loan_Status

    # initialize simple decision tree classifier from sklearn
    clf = ensemble.RandomForestClassifier()

    # fit the model on training data
    clf.fit(x_train, y_train)

    # create predictions for validation samples
    preds = clf.predict(x_valid)

    # calculate and print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold ={fold}, Accuracy = {accuracy}")

    # save the model
    joblib.dump(clf, f"../models/dt_{fold}.bin")


if __name__ == "__main__":
    # run(fold=0)
    # run(fold=1)
    # run(fold=2)
    # run(fold=3)
    # run(fold=4)
    # initialize Argument Parser class of argparse
    parser = argparse.ArgumentParser()

    # add the different arguments you need and their type
    # currently, we only need fold
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)

    # read the arguments from the command line
    args = parser.parse_args()

    # run the fold specified by command line arguments
    run(fold=args.fold, model=args.model)

