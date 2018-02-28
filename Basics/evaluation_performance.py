'''
Evaluate Machine Learning Algorithms: Resampling

 - Generally k-fold cross validation is the gold standard for evaluating the performance of a machine learning algorithm on unseen data with k set to 3, 5, or 10.
 
 - Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.
 
 - Techniques like leave-one-out cross validation and repeated random splits can be useful intermediates when trying to balance variance in the estimated performance, model training speed and dataset size.
'''

from pandas import read_csv
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import ShuffleSplit


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


seed = 7

''' 1. Split into Train/Test Set'''
# This algorithm evaluation technique is very fast,
# but it has a downside that it can have a high variance
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, random_state = seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test)

print("Accuracy: " + str(result * 100.0) + "%")
# Accuracy: 75.591%


''' 2. K-fold Cross Validation '''
# Split the dataset into k-parts. (Each part is called a 'fold')
# Train on k-1 folds and test on the remaining fold
# End up with k different performance scores and summarize with mean and standard deviation

num_folds = 7
kfold = KFold(n_splits = num_folds, random_state = seed)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv = kfold)

print("Accuracy with mean: " + str(results.mean()*100.0) + "%")
print("Accuracy with standard deviation: " + str(results.std()*100.0) + "%")
# Accuracy: 76.951% (4.841%)


''' 3. Leave One Out Cross Validation '''
# More reasonable estimate of the accuracy of model on unseen data
# Computationally expensive
num_folds = 10
loocv = LeaveOneOut()
#model = LogisticRegression()
results = cross_val_score(model,X,Y,cv = loocv)

print("Accuracy with mean: " + str(results.mean()*100.0) + "%")
print("Accuracy with standard deviation: " + str(results.std()*100.0) + "%")
# Accuracy: 76.823% (42.196%)


''' 4. Repeated Random Test-Train Splits '''
n_splits = 10
test_size = 0.33
kfold = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = seed)
# model = LogisticRegression()
results = cross_val_score(model,X,Y,cv = kfold)

print("Accuracy with mean: " + str(results.mean()*100.0) + "%")
print("Accuracy with standard deviation: " + str(results.std()*100.0) + "%")
# Accuracy: 76.496% (1.698%)




