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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


args = inputs.parse_args()

wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

SEED = 1234

X = wine_data.drop('Class', axis=1)
y = wine_data['Class']

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED)

print('\nEvaluating models...')
results = []
names = []
for name, model in SPOT_CHECK_MODELS:
    features = []
    features.append(('select_best', SelectKBest(k=3)))
    feature_union = FeatureUnion(features)

    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('model', model))
    pipeline_model = Pipeline(estimators)

    cv_score = calculate_cv_score(
            model, X_train, y_train, SEED, n_splits=10)
    cv_score_mean = cv_score.mean()
    cv_score_std = cv_score.std()
    metrics = {
        'model_name': name,
        'cv_acc_mean': cv_score_mean,
        'cv_acc_std': cv_score_std
    }
    results.append(metrics)
    print_cv_metrics(metrics)

print("\nMaking predictions for LDA")
lda = LinearDiscriminantAnalysis()
predictions = predict(lda, X_train, y_train, X_test, y_test)
metrics = calculate_metrics(y_test, predictions)
metrics['model_name'] = 'LDA'

print_model_metrics(metrics)

save_model_metrics(
    metrics=[metrics],
    file_name=args.results_folder + 'exp2_feature-selection_cv_model.txt')

save_cv_metrics(
        metrics=results,
        file_name=args.results_folder + 'exp2_feature-selection_cv_cv-results.txt')
