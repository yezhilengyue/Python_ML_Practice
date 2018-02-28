''' 
Spot-Checking  on   Classification

Spot-checking algorithms is about getting a quick assessment of a bunch of different algorithms on your machine learning problem so that you know what algorithms to focus on and what to discard.

Benefits of spot-checking algorithms on machine learning problems:
    - Speed
    - Objective
    - Results
    
    
Dataset:    Pima Indians onset of Diabetes
Test harness:  10-fold cross validation [To demonstrate how to spot-check ML algorithm]
Algorithm performance evaluation:   mean accuracy measures(MAE)
'''

''' Algorithm Overview '''
# Start with 2 linear machine learning algorithms:
#   - Logistric Regression                        LogistricRegression()
#   - Linear Discriminant Analysis                LinearDiscriminantAnalysis()

# Then look at 4 non-linear machine learning  algorithms:
#   - kNN                                          KNeighborsClassifier()
#   - Naive Bayes                                  GaussianNB()
#   - Classification and Regression Trees          DecisionTreeClassifier()
#   - SVM                                          SVC()



from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]

num_folds = 10
kfold = KFold(n_splits=10, random_state=7)




''' 1. Logistic Regression '''
# 
# Logistic regression assumes a Gaussian distribution for the numeric input variables and can model binary classification problems
# 
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv = kfold)

print(results.mean())
#   0.76951469583



''' 2. Linear Discriminant Analysis '''
# LDA is a statistical technique for binary and multiclass classification.
model = LinearDiscriminantAnalysis()
results = cross_val_score(model, X, Y, cv= kfold)

print(results.mean())
#   0.773462064252



''' 3. kNN '''
# KNN) uses a distance metric to find the k most similar instances in the training data for a new instance and takes the mean outcome of the neighbors as the prediction.
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv = kfold)

print(results.mean)
# 0.726555023923



''' 4. naive bayes '''
# Naive Bayes calculates the probability of each class and the conditional probability of each class given each input value
# These probabilities are estimated for new data and multiplied together, assuming that they are all independent (a simple or naive assumption)
model = GaussianNB()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
#  0.75517771702



''' 5. CART/decision tree '''
# Decision tree construct a binary tree from the training data.
model = DecisionTreeClassifier()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
# 0.686056049214



''' 6. Support Vector Machine'''
# SVM seek a line that best separates two classes. The use of differnet kernel functions via the kernel parameter is important.
# By default, Radial Basis Function is used
model = SVC()

results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())
#   0.651025290499