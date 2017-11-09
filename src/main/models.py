#!/bin/python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def calculate_metrics(y_test, predictions):
    return {
        'acc_score': accuracy_score(y_test, predictions),
        'conf_matrix': confusion_matrix(y_test, predictions),
        'class_report': classification_report(y_test, predictions)
    }


def calculate_cv_score(model, X_train, y_train, seed, n_splits=10):
    kfold = KFold(n_splits, random_state=seed)
    return cross_val_score(
            model, X_train, y_train, cv=kfold, scoring='accuracy')
