#!/bin/python
import dataset
import inputs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
plt.figure(figsize=(250, 90))
wine_boxplot = sns.boxplot(data=wine_data)
plt.show()
plt.savefig(args.graphs_folder + 'wine-boxplot.png')

print('\nGenerating dist plot...')
wine_data.hist(bins=30, figsize=(16, 12))
plt.show()
plt.savefig(args.graphs_folder + 'wine-hist.png')

print('\nGenerating dist plot grouped by class...')
wine_hist_by_class = wine_data.groupby('Class').hist(bins=30, figsize=(16, 12))
for wine_class, wine_hist in zip((1, 2, 3), wine_hist_by_class):
    fig = wine_hist[0, 0].get_figure()
    plt.title('Class: %d' % wine_class, fontsize=20)
    fig.suptitle('Class: %d' % wine_class, fontsize=20)
    fig.savefig('%swine-hist-class-%d.png' % (args.graphs_folder, wine_class))
    plt.close()

print('\nGenerating pair plot...')
wine_pairplot = sns.pairplot(wine_data)
wine_pairplot.savefig(args.graphs_folder + 'wine-pairplot.png')
plt.show()
