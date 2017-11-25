#!/bin/python
import dataset
import inputs
from graphs import boxplot
from graphs import hist
from graphs import hist_by
from graphs import pairplot

import pandas as pd


args = inputs.parse_args()

print('\nLoading data...')
wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

print('\nData shape')
print(wine_data.shape)

print('\nData head')
print(wine_data.head(10))

print('\nStatistics')
print(wine_data.describe())

print('\nGenerating box plot...')
boxplot(
    data=wine_data,
    figsize=(75, 30),
    file_name=args.graphs_folder + 'wine-boxplot.png')

print('\nGenerating dist plot...')
hist(
    data=wine_data,
    bins=30,
    figsize=(16, 12),
    file_name=args.graphs_folder + 'wine-hist.png')

print('\nGenerating dist plot grouped by class...')
hist_by(
    data=wine_data,
    by='Class',
    by_values=(1, 2, 3),
    bins=30,
    figsize=(16, 12),
    folder_name=args.graphs_folder)

print('\nGenerating pair plot...')
pairplot(
    data=wine_data,
    file_name=args.graphs_folder + 'wine-pairplot.png')
