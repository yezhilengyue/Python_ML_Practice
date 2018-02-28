'''
Binary classification projecgt - Sonar and Rocks

    The problem is to predict metal or rock objects from sonar return data. The label associated with each record contains the letter R if the object is a rock and M if it is a mine (metal cylinder).
    
'''

# Load libraries

import numpy
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


filename = 'sonar.all-data.csv'
data = read_csv(filename, header = None)



print(data.shape)
# shape:  (208, 61)

set_option('display.max_rows', 500) 
print(dataset.dtypes)
# type: all attributes are float, only class value is object

set_option('display.width', 100)
print(data.head(20))
# Take a peek: all data are at same scale

set_option('precision', 3)
print(data.describe())

print(data.groupby(60).size())
# class distribution:  Mine (111)   Rock (97)






# Histogram for each attribute
data.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
pyplot.show()
# We can see that there are a lot of Gaussian-like distributions and perhaps some exponential- like distributions for other attributes.


# Density plots
data.plot(kind = 'density', subplots = True, layout = (8,8), sharex = False, legend = False, fontsize = 1)
pyplot.show()
# We can see that many of the attributes have a skewed distribution


'''

# Something Wrong with this part about Index error

# Box plots 
data.plot(kind = 'box', subplots = True, layout = (8,8), sharex = False, sharey = False, fontsize = 1)
pyplot.show()


'''
# Box plots
fig, axes = pyplot.subplots(nrows = 8, ncols = 8, figsize = (6,6))
for i in range(8):
    for j in range(8):
        if i == 7 and j > 3:
            break
        else:
            axes[i, j].boxplot(data[i * 8 + j])
        
fig.subplots_adjust(hspace = 0.4)
pyplot.show()


# correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data.corr(), vmin = -1, vmax = 1, interpolation = 'none')
fig.colorbar(cax)
pyplot.show()



# Validation dataset
array = data.values
X = array[:,0:60].astype(float)
Y = array[:,60]
validation_ratio = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_ratio, random_state=seed)


''' Baseline '''

num_folds = 10
seed = 7
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))


results = []
names = []
for name, model in models:
    names.append(name)
    
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


'''
LR: 0.782721 (0.093796)     ********* worth further study
LDA: 0.746324 (0.117854)
KNN: 0.808088 (0.067507)    ********* worth further study
CART: 0.682353 (0.111564)
NB: 0.648897 (0.141868)
SVM: 0.608824 (0.118656)
'''


fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# The results show a tight distribution for KNN which is encouraging, suggesting low variance.
# The poor results for SVM are surprising.
'''
Explanation: 
    It is possible that the varied distribution of the attributes is having an effect on the accuracy of algorithms such as SVM. 
'''















''' Improvement with Standardization '''
# Transform data with mean value of zero and standard deviation of one
# To avoid data leakage, use pipeline for standardization and build model for each fold in th CV test

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', 
SVC())])))

results = []
names = []
for name, model in pipelines:
    names.append(name)
    
    kfold = KFold(n_splits = num_folds, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
'''
ScaledLR: 0.734191 (0.095885)
ScaledLDA: 0.746324 (0.117854)
ScaledKNN: 0.825735 (0.054511)      ***** KNN is still doing well, even better.
ScaledCART: 0.741176 (0.105601)
ScaledNB: 0.648897 (0.141868)
ScaledSVM: 0.836397 (0.088697)      ***** Standardization lifts the skill of SVM
'''
    
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison after Standardization')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# The results suggest digging deeper into the SVM and KNN algorithms.
'''
Explanation:
    It is very likely that configuration beyond the default params of kNN and SVM may yield even more accurate models.
'''













''' Improvement with Tuning '''
# 1. Tuning kNN
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

neighbors = [1,3,5,7,9,13,17,19,21]

param_grid = dict(n_neighbors = neighbors)

model = KNeighborsClassifier()

kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params']


for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

    
'''
Best: 0.849398 using {'n_neighbors': 1}

0.849398 (0.059881) with: {'n_neighbors': 1}
0.837349 (0.066303) with: {'n_neighbors': 3}
0.837349 (0.037500) with: {'n_neighbors': 5}
0.765060 (0.089510) with: {'n_neighbors': 7}
0.753012 (0.086979) with: {'n_neighbors': 9}
0.734940 (0.105836) with: {'n_neighbors': 13}
0.710843 (0.078716) with: {'n_neighbors': 17}
0.722892 (0.084555) with: {'n_neighbors': 19}
0.710843 (0.108829) with: {'n_neighbors': 21}

'''
# The optimal configuration is K = 1. This is interesting as the algorithm will make predictions using the most similar instance in the training dataset alone.






# 2. Tuning SVM (2 params: the value of C and the type of kernel)
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

param_grid = dict(C = c_values, kernel = kernel_values)

model = SVC()

kfold = KFold(n_splits = num_folds, random_state = seed)
grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = scoring, cv = kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score'] 
params = grid_result.cv_results_['params']


for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))    
    
'''
Best: 0.867470 using {'C': 1.5, 'kernel': 'rbf'}

0.759036 (0.098863) with: {'C': 0.1, 'kernel': 'linear'}
0.530120 (0.118780) with: {'C': 0.1, 'kernel': 'poly'}
0.572289 (0.130339) with: {'C': 0.1, 'kernel': 'rbf'}
0.704819 (0.066360) with: {'C': 0.1, 'kernel': 'sigmoid'}
0.746988 (0.108913) with: {'C': 0.3, 'kernel': 'linear'}
0.644578 (0.132290) with: {'C': 0.3, 'kernel': 'poly'}
0.765060 (0.092312) with: {'C': 0.3, 'kernel': 'rbf'}
0.734940 (0.054631) with: {'C': 0.3, 'kernel': 'sigmoid'}
0.740964 (0.083035) with: {'C': 0.5, 'kernel': 'linear'}
0.680723 (0.098638) with: {'C': 0.5, 'kernel': 'poly'}
0.789157 (0.064316) with: {'C': 0.5, 'kernel': 'rbf'}
0.746988 (0.059265) with: {'C': 0.5, 'kernel': 'sigmoid'}
0.746988 (0.084525) with: {'C': 0.7, 'kernel': 'linear'}
0.740964 (0.127960) with: {'C': 0.7, 'kernel': 'poly'}
0.813253 (0.084886) with: {'C': 0.7, 'kernel': 'rbf'}
0.753012 (0.058513) with: {'C': 0.7, 'kernel': 'sigmoid'}
0.759036 (0.096940) with: {'C': 0.9, 'kernel': 'linear'}
0.771084 (0.102127) with: {'C': 0.9, 'kernel': 'poly'}
0.837349 (0.087854) with: {'C': 0.9, 'kernel': 'rbf'}
0.753012 (0.073751) with: {'C': 0.9, 'kernel': 'sigmoid'}
0.753012 (0.099230) with: {'C': 1.0, 'kernel': 'linear'}
0.789157 (0.107601) with: {'C': 1.0, 'kernel': 'poly'}
0.837349 (0.087854) with: {'C': 1.0, 'kernel': 'rbf'}
0.753012 (0.070213) with: {'C': 1.0, 'kernel': 'sigmoid'}
0.771084 (0.106063) with: {'C': 1.3, 'kernel': 'linear'}
0.819277 (0.106414) with: {'C': 1.3, 'kernel': 'poly'}
0.849398 (0.079990) with: {'C': 1.3, 'kernel': 'rbf'}
0.710843 (0.076865) with: {'C': 1.3, 'kernel': 'sigmoid'}
0.759036 (0.091777) with: {'C': 1.5, 'kernel': 'linear'}
0.831325 (0.109499) with: {'C': 1.5, 'kernel': 'poly'}
0.867470 (0.090883) with: {'C': 1.5, 'kernel': 'rbf'}
0.740964 (0.063717) with: {'C': 1.5, 'kernel': 'sigmoid'}
0.746988 (0.090228) with: {'C': 1.7, 'kernel': 'linear'}
0.831325 (0.115695) with: {'C': 1.7, 'kernel': 'poly'}
0.861446 (0.087691) with: {'C': 1.7, 'kernel': 'rbf'}
0.710843 (0.088140) with: {'C': 1.7, 'kernel': 'sigmoid'}
0.759036 (0.094276) with: {'C': 2.0, 'kernel': 'linear'}
0.831325 (0.108279) with: {'C': 2.0, 'kernel': 'poly'}
0.867470 (0.094701) with: {'C': 2.0, 'kernel': 'rbf'}
0.728916 (0.095050) with: {'C': 2.0, 'kernel': 'sigmoid'}

'''    
# We can see the most accurate configuration was SVM with an RBF kernel and a C value of 1.5. The accuracy 86.7470% is seemingly better than what KNN could achieve.
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    

''' Improvement with Ensemble '''
# Boosting Methods: AdaBoost (AB) and Gradient Boosting (GBM).   
# Bagging Methods: Random Forests (RF) and Extra Trees (ET).

# No data standardization is used in this case because all four ensemble algorithms are based on decision trees that are less sensitive to data distributions.

ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))


results = []
names = []
for name, model in ensembles:
    names.append(name)
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

'''
AB: 0.819853 (0.058293)
GBM: 0.823897 (0.101025)
RF: 0.795956 (0.095662)
ET: 0.795221 (0.083918)
'''


fig = pyplot.figure()
fig.suptitle('Algorithm Comparison after Ensemble')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
# The results suggest GBM may be worthy of further study, with a strong mean and a spread that skews up towards high 90s (%) in accuracy.

















''' Finalize the model '''
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)

model = SVC(C = 1.5)
model.fit(rescaledX, Y_train)

rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)

print(accuracy_score(Y_validation, predictions))
# 0.857142857143

print(confusion_matrix(Y_validation, predictions))
# [[23  4]
#  [ 2 13]]

print(classification_report(Y_validation, predictions))
#              precision    recall  f1-score   support

#           M       0.92      0.85      0.88        27
#           R       0.76      0.87      0.81        15

# avg / total       0.86      0.86      0.86        42




# Findings: A part of the findings was that SVM performs better when the dataset is standardized so that all attributes have a mean value of zero and a standard deviation of one.







