#!/bin/python
import dataset

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


parser = argparse.ArgumentParser(
        description='Provide input data file')
parser.add_argument('input_data_file', type=str, help='input data file')
args = parser.parse_args()

wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

SEED = 23

X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)

MODELS = (
    ('Logistic Regression', LogisticRegression()),
    ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
    ('KNeighbors', KNeighborsClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Gaussian NB', GaussianNB()),
    ('SVM', SVC())
)

print('Evaluating models...')
results = []
for desc, model in MODELS:
    kfold = KFold(n_splits=10, random_state=SEED)
    cv_score = cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_score)
    print("%s: %f (%f)" % (desc, cv_score.mean(), cv_score.std()))
