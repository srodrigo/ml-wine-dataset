#!/bin/python
import dataset
from results import save_model_metrics, print_model_metrics
from models import predict, calculate_metrics

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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

print("\nMaking predictions with SVM and grid search...")
param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
predictions = predict(grid, X_train, y_train, X_test, y_test)
metrics = calculate_metrics(y_test, predictions)
metrics['model_name'] = 'LDA'

print_model_metrics(metrics)

save_model_metrics(
    metrics=[metrics],
    file_name=args.results_folder + 'exp2_feature-selection_model.txt')
