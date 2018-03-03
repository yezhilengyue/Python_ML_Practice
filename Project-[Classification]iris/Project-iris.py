'''
Six-step ML project template:
	- 1. Define problem
		1) Load libraries
		2) Load dataset

	- 2. Summarize data
		1) Descriptive statistics
		2) Data visualization

	- 3. Prepare data
		1) Data cleaning
		2) Feature selection
		3) Data transform

	- 4. Evaluate algorithm
		1) Split-out validation dataset
		2) Test options and evaluations metric
		3) Spot-Check algorithms
		4) Compare algorithms

	- 5. Improve results
		1) Algorithm tuning
		2) Ensembles

	- 6. Present results 
		1) Predictions on validation dataset
		2) Create standalone model on entire training dataset
		3) Save model for later use
'''


''' Iris Dataset '''

# 1.1 Load libraries
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# 1.2 Load dataset
filename = 'iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = read_csv(filename, names = names)


# 2.2 Descriptive statistics
print(data.shape)
print(data.head(20))
print(data.describe())
print(data.groupby('class').size())


# 2.3 Data visualization
data.plot(kind = 'box', subplots = True, layout=(2,2), sharex = False, sharey = False)
pyplot.show()

data.hist()
pyplot.show()


scatter_matrix(data)
pyplot.show()


# 3.1 Split-out validation dataset
array = data.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = seed)


# 3.2 Test options and evaluations metric
# We will use 10-fold cross validation
# to estimate accuracy


# 3.3 Spot-check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# 3.4 enaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = name + ': ' + str(cv_results.mean()) + ' (' + str(cv_results.std()) + ')'
    print(msg)

#   LR: 0.966667 (0.040825)
#   LDA: 0.975000 (0.038188)
#   KNN: 0.983333 (0.033333) * It looks that KNN has the highest accuracy score
#   CART: 0.975000 (0.038188)
#   NB: 0.975000 (0.053359)
#   SVM: 0.981667 (0.025000)


# 3.5 Compare Algorithm
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# 4.1 Make prediction on validation data
# The KNN algorithm was the most accurate model that we tested.
# We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#   0.9         
# The accuracy is 0.9

#   [[ 7 0 0] 
#    [ 0 11 1] 
#    [ 0 2 9]]  
# The confusion matrix shows that there are 3 errors made

#                     precision    recall  f1-score   support

#        Iris-setosa       1.00      1.00      1.00         7
#    Iris-versicolor       0.85      0.92      0.88        12
#     Iris-virginica       0.90      0.82      0.86        11

#        avg / total       0.90      0.90      0.90        30






