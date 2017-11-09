#!/bin/python
import dataset
import inputs
from results import save_model_metrics, print_model_metrics
from models import predict
from models import calculate_metrics
from models import SPOT_CHECK_MODELS

import pandas as pd
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
for name, model in SPOT_CHECK_MODELS:
    print("\nMaking predictions with %s..." % name)
    predictions = predict(model, X_train, y_train, X_test, y_test)
    metrics = calculate_metrics(y_test, predictions)
    metrics['model_name'] = name
    results.append(metrics)

    print_model_metrics(metrics)

results_file_name = args.results_folder + 'exp1_raw-data_results.txt'
save_model_metrics(metrics=results, file_name=results_file_name)
