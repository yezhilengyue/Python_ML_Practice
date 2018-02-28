'''
Spot Checking   on  Regression

Dataset:    Boston House Price dataset
Test harness:  10-fold cross validation [To demonstrate how to spot-check ML algorithm]
Algorithm performance evaluation:   mean accuracy measures(MAE)

'''

''' Algorithm Overview '''
# Start with 4 linear machine learning algorithms:
#   - Linear Regression
#   - Ridge Regression
#   - LASSO Linear Regression
#   - Elastic Net Regression

# Then look at 3 non-linear machine learning  algorithms:
#   - kNN
#   - Classification and Regression Trees          DecisionTreeClassifier()
#   - SVM  


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.scv import SVR



filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
data = read_csv(filename, delim_whitespace = True, names = names)
array = data.values
X = array[:,0:13]
Y = array[:,13]

kfold = KFold(n_splits = 10, random_state = 7)
scoring = 'neg_mean_squared_error'


''' 1. Linear Regression '''
model = LinearRegression()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
#   -34.7052559445



''' 2. Ridge Regression '''
# As an extension of linear regression, the loss function of ridge regression is modified to minimize the complexity of the model measured as the sum squared value of the coefficient values. 
# [L2-norm: least squares]
model = Ridge()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
# -34.0782462093



''' 3. LASSO Regression '''
# Similar to linear regression, minimize the complexity of the model measured as the sum absolute value of the coefficient values 
# [L1-norm: least absolute errors]
modle = Lasso()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
#  -34.4640845883



''' 4. ElasticNet Regression '''
# ElasticNet is a form of regularization regression that combines the properties of both Ridge Regression and LASSO regression.

#   To minimize the complexity of the regression model [magnitute and number of regression coefficients]
#   penalizing L1-norm and L2-norm
model = ElasticNet()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
#   -31.1645737142






''' 5.kNN '''
model = KNeighborsRegressor()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
#   -107.28683898


''' 6. Classification and Regression Trees '''
# Use the training data to select the best points to split the data in order to minimize a cost metric.
model = DecisionTreeRegressor()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())


''' 7. SVM '''
model = SVR()
results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print(results.mean())
# -91.0478243332




