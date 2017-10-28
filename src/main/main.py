#!/bin/python
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
        description='Provide input data file and graphs folder')
parser.add_argument('input_data_file', type=str, help='input data file')
parser.add_argument('graphs_folder', type=str, help='graphs folder')
args = parser.parse_args()

columns = [
    'Class',
    'Alcohol',
    'Malic acid',
    'Ash',
    'Alcalinity of ash',
    'Magnesium',
    'Total phenols',
    'Flavanoids',
    'Nonflavanoid phenols',
    'Proanthocyanins',
    'Color intensity',
    'Hue',
    'OD280/OD315',
    'Proline'
]

print('Loading data...')
wine_data = pd.read_csv(args.input_data_file, names=columns)

print('Generating box plot...')
plt.figure(figsize=(25, 9))
wine_boxplot = sns.boxplot(data=wine_data)
wine_boxplot.get_figure().savefig(args.graphs_folder + 'wine-boxplot.png')

print('Generating dist plot...')
the_hist = wine_data.hist(figsize=(16, 12))
plt.savefig(args.graphs_folder + 'wine-hist.png')

print('Generating pair plot...')
wine_pairplot = sns.pairplot(wine_data)
wine_pairplot.savefig(args.graphs_folder + 'wine-pairplot.png')
