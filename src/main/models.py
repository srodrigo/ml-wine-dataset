#!/bin/python
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def predict(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


def calculate_metrics(y_test, predictions):
    return {
        'acc_score': accuracy_score(y_test, predictions),
        'conf_matrix': confusion_matrix(y_test, predictions),
        'class_report': classification_report(y_test, predictions)
    }