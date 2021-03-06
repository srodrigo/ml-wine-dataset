#!/bin/python


def save_model_metrics(metrics, file_name):
    with open(file_name, 'w') as metrics_file:
        for metric in metrics:
            metrics_file.write("Model: %s\n" % metric['model_name'])
            metrics_file.write("Accuracy: %f\n" % metric['acc_score'])
            metrics_file.write(str(metric['conf_matrix']) + "\n")
            metrics_file.write(metric['class_report'] + "\n")


def print_model_metrics(metrics):
    print("Model: %s" % metrics['model_name'])
    print("Accuracy: %f" % metrics['acc_score'])
    print(metrics['conf_matrix'])
    print(metrics['class_report'])


def save_cv_metrics(metrics, file_name):
    with open(file_name, 'w') as metrics_file:
        for metric in metrics:
            metrics_file.write("%s: %f (%f)\n" % (
                metric['model_name'],
                metric['cv_acc_mean'],
                metric['cv_acc_std']))


def print_cv_metrics(metrics):
    print("%s: %f (%f)" % (
        metrics['model_name'],
        metrics['cv_acc_mean'],
        metrics['cv_acc_std']))
