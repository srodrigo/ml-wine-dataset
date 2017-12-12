#!/bin/python
import dataset
import inputs
from results import save_model_metrics
from results import print_model_metrics
from results import save_cv_metrics
from results import print_cv_metrics
from models import predict
from models import calculate_metrics
from models import calculate_cv_score
from models import SPOT_CHECK_MODELS

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


args = inputs.parse_args()

wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

SEED = 1234

X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)

print('\nEvaluating models...')
results = []
scores = []
names = []
for name, model in SPOT_CHECK_MODELS:
    names.append(name)

    cv_score = calculate_cv_score(
            model, X_train, y_train, SEED, n_splits=10)
    scores.append(cv_score)
    cv_score_mean = cv_score.mean()
    cv_score_std = cv_score.std()
    metrics = {
        'model_name': name,
        'cv_acc_mean': cv_score_mean,
        'cv_acc_std': cv_score_std
    }
    results.append(metrics)
    print_cv_metrics(metrics)

save_cv_metrics(
        metrics=results,
        file_name=args.results_folder + 'exp1_raw-data_cv_cv-results.txt')

print('\nPlotting algorithm comparison...')
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(scores)
ax.set_xticklabels(names)
ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
              alpha=0.5)
plt.show()
plt.savefig(args.graphs_folder + 'exp1_raw-data_alg-comparison.png')

print('\nPredicting...')
test_results = []
for name, model in SPOT_CHECK_MODELS:
    print("\nMaking predictions with %s..." % name)
    predictions = predict(model, X_train, y_train, X_test)
    metrics = calculate_metrics(y_test, predictions)
    metrics['model_name'] = name

    test_results.append(metrics)
    print_model_metrics(metrics)

save_model_metrics(
    metrics=test_results,
    file_name=args.results_folder + 'exp1_raw-data_cv_model-results.txt')
