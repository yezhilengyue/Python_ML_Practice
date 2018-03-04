'''
Regression ML project on Boston Housing Price

For this project we will investigate the Boston House Price dataset.

 - Problem Definition (Boston house price data).
 - Loading the Dataset.
 - Analyze Data (some skewed distributions and correlated attributes).
 - Evaluate Algorithms (Linear Regression looked good).
 - Evaluate Algorithms with Standardization (KNN looked good).
 - Algorithm Tuning (K=3 for KNN was best).
 - Ensemble Methods (Bagging and Boosting, Gradient Boosting looked good).   
 - Tuning Ensemble Methods (getting the most from Gradient Boosting).
 - Finalize Model (use all training data and confirm using validation dataset).
 
'''

## Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error



## Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
## specifying the short names for each attribute to reference them clearly later.
data = read_csv(filename, names = names, delim_whitespace = True)




## Descriptive data
print(data.shape) 
#(506, 14)
print(data.dtypes)

print(data.head(20))

set_option('precision', 1)
print(data.describe())
# to summarize the distribution of each attribute.

set_option('precision', 2)
pearson = data.corr(method = 'pearson')
# assume target attr is the last, then remove corr with itself
corr_with_target = pearson.ix[-1][:-1]
# attributes sorted from the most predictive
predictivity = corr_with_target.sort_values(ascending=False)
'''
RM         0.695360
ZN         0.360445
B          0.333461
DIS        0.249929
CHAS       0.175260
AGE       -0.376955
RAD       -0.381626
CRIM      -0.385832
NOX       -0.427321
TAX       -0.468536
INDUS     -0.483725
PTRATIO   -0.507787
LSTAT     -0.737663
Name: MEDV, dtype: float64
'''

# To find the attribute with strongest correlation with output, it would be better to sort the correlations by the absolute value:

'''
>>> corr_with_target[abs(corr_with_target).argsort()[::-1]]
LSTAT     -0.737663
RM         0.695360
PTRATIO   -0.507787
INDUS     -0.483725
TAX       -0.468536
NOX       -0.427321
CRIM      -0.385832
RAD       -0.381626
AGE       -0.376955
ZN         0.360445
B          0.333461
DIS        0.249929
CHAS       0.175260
Name: MEDV, dtype: float64
'''

# It might be interesting to select some strong correlations between attribute pairs.
attrs = pearson.iloc[:-1,:-1] # all except target
# only important correlations and not auto-correlations
threshold = 0.5
# {('LSTAT', 'TAX'): 0.543993, ('INDUS', 'RAD'): 0.595129, ...
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]).unstack().dropna().to_dict()
#     attribute pair  correlation
# 0     (AGE, INDUS)     0.644779
# 1     (INDUS, RAD)     0.595129
# ...

unique_important_corrs = data.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])

# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]


## Data visualization
#   Unimodal dv
data.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize =1)
pyplot.show()

data.plot(kind = 'density', subplots = True, layout = (4,4), sharex = False, legend = False, fontsize = 1)
pyplot.show()
# Add more evidence to our suspicion about possible exponential and bimodal distributions.

data.plot(kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False, fontsize = 8)
pyplot.show()
# This helps point out the skew in many distributions so much so that data looks like outliers


#   Multimodal dv [visualizations of the interactions between variables]
scatter_matrix(data)
pyplot.show()

# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
ticks = numpy.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()
# We can also see some dark color suggesting candidates removal for improvement of better model accuracy.





# Preparing data
#   - feature selection and removal of correlated attributes
#   - normalization to reduce scaling effect
#   - standardization to reduce distribution effect
#   - bining to improve accuracy for decision tree algorithm

# Split-out validation dataset
array = data.values
X = array[:,0:13]
Y = array[:,13]
validation_ratio = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_ratio, random_state = seed)
# To use validation hold-out set as a smoke test to give us confidence on our estimates of accuracy on unseen data.



# Evaluate Algorithm: Baseline

# Use 10-fold cross validation because the dataset is not too small
# Use MSE metric to give idea of how wrong all predictions are
num_folds = 10
#seed = 7
scoring = 'neg_mean_squared_error'

# create a baseline of performance
# spot-check a number of different algorithms:
#   - 3 linear algorithms: LR, LASSO,EN     (with default tuning params)
#   - 3 nonlinear algorithm: CART, SVR, KNN (with default tuning params)
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    names.append(name)
    
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    
    msg = name + ': ' + str(cv_results.mean()) + ' (' + str(cv_results.std()) + ')'
    print(msg)
    
#   LR: -21.3798557267 (9.41426365698) ************
#   LASSO: -26.4235611084 (11.6511099158)
#   EN: -27.5022593507 (12.3050222641)
#   KNN: -41.8964883902 (13.9016881498)
#   CART: -22.9675067073 (10.9475148015)
#   SVR: -85.5183418393 (31.9947982318)    
# Obviously, LR has the lowest MSE, followed closely by CART.

# take a look at the distribution of scores
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



# Evaluate Algorithms: Standardization
# suspect that the differing scales of the raw data may be negatively impacting the skill of some of the algorithms.
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR',SVR())])))

results = []
names = []
for name, model in pipelines:
    names.append(name)
    
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    
    msg = name + ': ' + str(cv_results.mean()) + ' (' + str(cv_results.std()) + ')'
    print(msg)


#   ScaledLR: -21.3798557267 (9.41426365698)
#   ScaledLASSO: -26.6073135577 (8.97876148589)
#   ScaledEN: -27.9323721581 (10.5874904901)
#   ScaledKNN: -20.1076204878 (12.3769491508) ********* effect on knn
#   ScaledCART: -24.4931573171 (10.4662755609)
#   ScaledSVR: -29.6330855003 (17.0091860524)
# We can see that KNN has both a tight distribution of error and has the lowest score.

# take a look at the standardized distribution of scores
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
   
    

# Improvements: params tuning (e.g. KNN tuning)

# The default value for the number of neighbors in KNN is 7. 
# The below tries odd k values from 1 to 21.

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors = k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid,scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)

print('Best: ' + str(grid_result.best_score_) + ' using ' + str(grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


    
    
# Improvements: ensemble methods
#   - 1. Boosting [AdaBoost, Gradient Boosting]
#   - 2. Bagging [Random Forest, Extra Trees]
ensemble = []
ensemble.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostRegressor())])))
ensemble.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
ensemble.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()), ('RF', RandomForestRegressor())])))
ensemble.append(('ScaledET', Pipeline([('Scaler', StandardScaler()), ('ET', ExtraTreesRegressor())])))

results = []
names = []
for name, model in ensemble:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
    
#   ScaledAB: -14.957611 (6.579588)
#   ScaledGBM: -10.054631 (4.522576)
#   ScaledRF: -13.718030 (7.982746)
#   ScaledET: -10.530828 (6.818533)
# Generally we can get better scores than our linear and nonlinear algorithms in previous sections.

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison') 
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()



# Tune scaled GBM: to do better with GBM
# The default number of boosting stages to perform is 100.
# Often, the larger the number of boosting stages, the better the performance but the longer the training time
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))

model = GradientBoostingRegressor(random_state = seed)
kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid,scoring = scoring, cv = kfold)

grid_result = grid.fit(rescaledX, Y_train)


print('Best: ' + str(grid_result.best_score_) + ' using ' + str(grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Best: -9.35647101741 using {'n_estimators': 400}

#   -10.812167 (4.724394) with: {'n_estimators': 50}
#   -10.040857 (4.441758) with: {'n_estimators': 100}
#   -9.694096 (4.275762) with: {'n_estimators': 150}
#   -9.539706 (4.270637) with: {'n_estimators': 200}
#   -9.448764 (4.262603) with: {'n_estimators': 250}
#   -9.429946 (4.273791) with: {'n_estimators': 300}
#   -9.369824 (4.254108) with: {'n_estimators': 350}
#   -9.356471 (4.267837) with: {'n_estimators': 400}

# We can see that the best configuration was n estimators=400, better than the untuned method.





# Finalize model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state = seed, n_estimators = 400)
model.fit(rescaledX, Y_train)


rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))
# 11.8752520792












