## Define the experiment
We want to be able to classify wine varieties using the given data.

## Collect data
https://archive.ics.uci.edu/ml/datasets/Wine

## Analyse data

### Statistics
We can see that there is a big difference in scale. *Proline* has the widest range of values. This means that our model will need feature scaling/normalisation.

There seem to be some outliers in:
  * Magnesium
  * Alcalinity of ash
  * Color intensity
  * Malic acid

### Feature Distributions (Box plot and Distribution plot)

There seem to be some outliers in:
  * Magnesium
  * Ash
  * Alcalinity of ash
  * Malic acid
  * Proanthocyanins
  * Color intensity
  * Hue

There doesn't seem to be any missing values.

Class 2 is the most common, followed by class 1 and 3.

Attributes with normal distribution: Alcalinity of ash, Ash, Proanthocyanins.
Others aren't that clear

### Feature-Class Relationships (Distribution plot)
If graphs for a feature-class have the same distributions for the different values of the class, this feature is useful for discriminating the class.

Total phenols seem to have a similar distribution for all class values.

### Feature-Feature Relationships (Pair plot)
Attributes that are highly correlated could be redundant.

Possible correlations:
  * Color intensity - Alcohol
  * Proline - Alcohol
  * Flavanoids - Alcohol
  * Ash - Alcohol
  * Mangesium - Alcohol
  * Ash - Alcalinity of ash
  * Flavanoids - Total phenols
  * Flavanoids - Hue
  * Flavanoids - OD280/OD315
  * Proanthocyanins - Flavanoids
  * Hue - OD280/OD315
  * Total phenols - OD280/OD315

Maybe try to remove Alcohol and Flavanoids later and see if we get more accuracy.

## Prepare data

### Fix missing data
There is no missing data.

### Remove outliers

### Feature scaling

### Feature engineering

## Train a model

## Feature selection

### Spot checking

### Select algorithm

## Test/Evaluate the model performance

## Improve the model performance

### Parameters tunning

## Results

