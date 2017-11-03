#!/bin/python
import dataset

import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
        description='Provide input data file and graphs folder')
parser.add_argument('input_data_file', type=str, help='input data file')
parser.add_argument('graphs_folder', type=str, help='graphs folder')
args = parser.parse_args()

print('Loading data...')
wine_data = pd.read_csv(args.input_data_file, names=dataset.COLUMNS)

print('Data shape')
print(wine_data.shape)

print('Statistics')
print(wine_data.describe())

print('Generating box plot...')
plt.figure(figsize=(25, 9))
wine_boxplot = sns.boxplot(data=wine_data)
plt.show()
plt.savefig(args.graphs_folder + 'wine-boxplot.png')

print('Generating dist plot...')
wine_data.hist(bins=30, figsize=(16, 12))
plt.show()
plt.savefig(args.graphs_folder + 'wine-hist.png')

print('Generating dist plot grouped by class...')
wine_hist_by_class = wine_data.groupby('Class').hist(bins=30, figsize=(16, 12))
for wine_class, wine_hist in zip((1, 2, 3), wine_hist_by_class):
    fig = wine_hist[0, 0].get_figure()
    plt.title('Class: %d' % wine_class, fontsize=20)
    fig.suptitle('Class: %d' % wine_class, fontsize=20)
    fig.savefig('%swine-hist-class-%d.png' % (args.graphs_folder, wine_class))
    plt.close()

print('Generating pair plot...')
wine_pairplot = sns.pairplot(wine_data)
wine_pairplot.savefig(args.graphs_folder + 'wine-pairplot.png')
plt.show()
