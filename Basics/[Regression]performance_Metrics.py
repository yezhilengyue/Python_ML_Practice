'''
Model Evaluation - Performance Metrics 2: Regression

 - Performance metrics is important in ML algorithms as it determines:
     1. how you weight the importance of differnt characteristics
     2. which algorithm you ultimately choose
     
 - For classification matrics, the Pima Indian onset of diabetes dataset is used. (Logistic Regression)
 
 - For regression matrics, the Boston House Price dataset is used. (Linear Regression)
 
  - 10-fold cross validation test is used for metric demonstration
  
'''

''' Regression Metrics'''
    # Mean Absolute Error:      scoring = 'neg_mean_absolute_error'
    # Mean Squared Error
    # R^2
    
    
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, delim_whitespace = True, names = names)
array = data.values
X = array[:,0:13]
Y = array[:,13]

seed = 7
kfold = KFold(n_splits = 10, random_state = seed)
model = LinearRegression()

    
    
    
''' 1. Mean Absolute Error '''
# MAE is the sum of the absolute differences between predictions and actual values.
# Give an idea of magnitude of the error, but no idea of the direction
scoring = 'neg_mean_absolute_error'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
#   MAE: -4.005 (2.084)
# 0 indicates no error or perfect prediction




''' 2. Mean Squared Error '''
# MSE is much like mean absolute error as it provides a gross idea of the magnitude of error
scoring = 'neg_mean_squared_error'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
#   MSE: -34.705 (45.574)




''' 3. R^2 Metric '''
# R Squared metric provides a indication of the goodness of fit of a set of predictions to the actual values
# Coefficient of determination
# Range from 0 to 1
scoring = 'r2'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
#   R^2: 0.203 (0.595)
# These predictions have a poor fit to the actual values with a value closer to zero and less than 0.5.
