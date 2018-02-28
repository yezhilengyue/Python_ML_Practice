''' 
Compare ML Algorithm 
    
Goal:
    1. Formulate an experiment to directly compare machine learning algorithms
    
    2. Create a reusable template for evaluating the performance of multiple algorithms on one dataset
    
    3. Report and visualize the results of comparing algorithm performance
'''

''' Algorithms Overview'''
#  Logistic Regression
#  LDA
#  kNN
#  CART
#  Naive Bayes
#  SVM



from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]


# Prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# Evaluate each model in turn
results =[]
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 7)
    cv_results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print(name + ': ' + str(cv_results.mean()))

# LR: 0.769515 (0.048411)
# LDA: 0.773462 (0.051592)
# KNN: 0.726555 (0.061821)
# CART: 0.695232 (0.062517)
# NB: 0.755178 (0.042766)
# SVM: 0.651025 (0.072141)
    
# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()








