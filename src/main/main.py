#!/bin/python
import pandas as pd
import seaborn as sns
import argparse


parser = argparse.ArgumentParser(
        description='Provide input data file and graphs folder')
parser.add_argument('input_data_file', type=str, help='input data file')
parser.add_argument('graphs_folder', type=str, help='graphs folder')
args = parser.parse_args()

columns = [
    'Unknown',
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
    'OD280/OD315 of diluted wines',
    'Proline'
]

print('Loading data...')
wine_data = pd.read_csv(args.input_data_file, names=columns)
print('Generating pair plot...')
wine_pairplot = sns.pairplot(wine_data)
wine_pairplot.savefig(args.graphs_folder + 'wine.png')
