#!/bin/python
import dataset

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
names = []
for name, model in MODELS:
    print("\nMaking predictions with %s..." % name)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc_score = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    results.append({
        'model_name': name,
        'acc_score': acc_score,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    })

    print("Accuracy: %f" % acc_score)
    print(conf_matrix)
    print(class_report)

results_file_name = args.results_folder + 'exp2_feature-selection.txt'
with open(results_file_name, 'w') as results_file:
    for res in results:
        results_file.write("Model: %s\n" % res['model_name'])
        results_file.write("Accuracy: %f\n" % res['acc_score'])
        results_file.write(str(res['conf_matrix']) + "\n")
        results_file.write(res['class_report'] + "\n")
