'''
Tuning: To improve performance

[hyperparameter optimization] - Optimization suggests the search-nature of the problem. 
'''

''' 1. Grid Search Parameter Tuning. '''
# Grid search is an approach to parameter tuning that will methodically build and evaluate a model for each combination of algorithm parameters specified in a grid.

import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha = alphas)
model = Ridge()
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid.fit(X, Y)

print(grid.best_score_)
print(grid.best_estimator_.alpha)


''' 2. Random Search Parameter Tuning '''
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'alpha': uniform()}
research = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 100, random_state = 7)
research.fit(X, Y)



print(research.best_score_)
#   0.279617354112
print(research.best_estimator_.alpha)
#   0.989527376274  An optimal alpha value near 1.0 is discovered.



















