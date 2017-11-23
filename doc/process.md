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
No outliers have been removed.

### Feature regularisation
No feature scaling/normalisation performed.

### Feature engineering
No features added.

### Feature selection
Automatic feature selection performed on the Experiment 2.

## Train a model

### Spot checking
The spot checking has been performed using a few basic algorithms"
  * Logistic Regression
  * Linear Discriminant Analysis
  * KNeighbors
  * Decision Tree
  * Naive Bayes
  * Support Vector Machine

### Select algorithm
After applying cross-validation, the most performant algorithm was Naive Bayes: 0.992857 (0.021429)

Logistic Regression, Linear Discriminant Analysis and Decision Tree performed quite well: >0.95 (< 0.05)

## Test/Evaluate the model performance
Predicting on the test data, the best algorithm was Linear Discriminant Analysis: 0.972222

## Improve the model performance
In order to carry on with an interesting challenge, the improvements are going to be made on SVM (the one with the worst accuracy).

In the Experiment 2, we used automatic feature selection to select the best 3 features.

### Parameters tunning

In the Experiment 3, we performed a grid search with cross-validation to select the best combination of parameters for a few parameter values.

## Results

* Experiment 1
  Cross-validation score: 0.367
  Accuracy: 0.611

* Experiment 2
  Cross-validation score: improved in a 11.3%, up to 0.479
  Accuracy: improved in a 2.7%, up to 0.639

* Experiment 3
  Accuracy: improved in a 27.8%, up to 0.917
