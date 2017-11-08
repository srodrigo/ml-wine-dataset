#!/bin/python


def save_model_metrics(metrics, file_name):
    with open(file_name, 'w') as metrics_file:
        for metric in metrics:
            metrics_file.write("Model: %s\n" % metric['model_name'])
            metrics_file.write("Accuracy: %f\n" % metric['acc_score'])
            metrics_file.write(str(metric['conf_matrix']) + "\n")
            metrics_file.write(metric['class_report'] + "\n")
