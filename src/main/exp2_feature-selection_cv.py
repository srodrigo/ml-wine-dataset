#!/bin/python
import dataset
from results import save_model_metrics

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
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
results = []
names = []
for name, model in MODELS:
    features = []
    features.append(('select_best', SelectKBest(k=3)))
    feature_union = FeatureUnion(features)

    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('model', model))
    pipeline_model = Pipeline(estimators)

    kfold = KFold(n_splits=10, random_state=SEED)
    cv_score = cross_val_score(
            pipeline_model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_score)
    print("%s: %f (%f)" % (name, cv_score.mean(), cv_score.std()))

print("\nMaking predictions for %s..." % name)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)

acc_score = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)
print("Accuracy: %f" % acc_score)
print(conf_matrix)
print(class_report)

save_model_metrics(
    metrics=[{
        'model_name': 'LDA',
        'acc_score': acc_score,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }],
    file_name=args.results_folder + 'exp2_feature-selection_cv_model.txt')
