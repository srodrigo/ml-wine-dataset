#!/bin/python
import dataset
from results import save_model_metrics, print_model_metrics
from models import predict, calculate_metrics

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


parser = argparse.ArgumentParser(
        description='Provide input data file')
parser.add_argument('input_data_file', type=str, help='input data file')
parser.add_argument('graphs_folder', type=str, help='graphs folder')
parser.add_argument('results_folder', type=str, help='results folder')
args = parser.parse_args()

wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

SEED = 1234

X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)

MODELS = (
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
)

print('\nEvaluating models...')

best_features = SelectKBest(chi2, k=3)
X_train = best_features.fit_transform(X_train, y_train)
X_test = best_features.transform(X_test)

results = []
for name, model in MODELS:
    print("\nMaking predictions with %s..." % name)
    predictions = predict(model, X_train, y_train, X_test, y_test)
    metrics = calculate_metrics(y_test, predictions)
    metrics['model_name'] = name
    results.append(metrics)

    print_model_metrics(metrics)

results_file_name = args.results_folder + 'exp2_feature-selection.txt'
save_model_metrics(metrics=results, file_name=results_file_name)
