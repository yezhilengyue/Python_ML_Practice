'''
Model Evaluation - Performance Metrics 1: Classification

 - Performance metrics is important in ML algorithms as it determines:
     1. how you weight the importance of differnt characteristics
     2. which algorithm you ultimately choose
     
 - For classification matrics, the Pima Indian onset of diabetes dataset is used. (Logistic Regression)
 
 - For regression matrics, the Boston House Price dataset is used. (Linear Regression)
 
  - 10-fold cross validation test is used for metric demonstration
  
'''

''' Classification Metrics'''
    # Classification Accuracy:  scoring = 'accuracy'
    # Logarithmic Loss:         scoring = 'neg_log_loss'
    # Area Uner ROC Curve:      scoring = 'roc_auc'
    
    # Confusion Matrix:         confusion_matrix() function
    # Classification Report:    classification_report() function


from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]

seed = 7
test_size = 0.33
kfold = KFold(n_splits = 10, random_state = seed)
model = LogisticRegression()





''' 1. Classification Accuracy'''
# Classification accuracy is the number of correct predictions made as a ratio of all predictions made.

# Only suitable when there are un equal number of observations in each class 
# and all predictions and predictions errors are equally important

scoring = 'accuracy'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
#   Accuracy: 0.770 (0.048)




''' 2. Logarithmic Loss '''
# To evaluate the prediction of probabilities of membership to a given class
# Between 0 and 1
# Can be seen as a measure of confidence for a prediction by an algorithm
# Predictions that are correct or incorrect are rewarded or punished proportionally to the confidence of the prediction.


#    Just replace 'scoring = xxx' part
scoring = 'neg_log_loss'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
#   Logloss: -0.493 (0.047)
#The smaller, the better. 
#Smaller logloss is better with 0 representing a perfect logloss.



''' 3. Area Under ROC Curve '''
# AUC is a performance metric for binary classification
# AUC represents a model's ability to discriminate between positive and negative class
# An area of 1.0 represents that model can make all predictions perfectly.
# An area of 0.5 means a model is as good as random

# Binary classification problem is really a trade-off between sensitivity and specificity

# Sensitivity(recall): [True Positive Rate] Number of instances from the positive(first) class that actuallypredicted correctly

# Specificity: [True Negative rate] Number of instances from the negative(second) class that were actually predicted correctly


#    Just replace 'scoring = xxx' part
scoring = 'roc_auc'

results = cross_val_score(model, X, Y, cv = kfold, scoring = scoring)

print("Accuracy with mean: " + str(results.mean()))
print("Accuracy with standard deviation: " + str(results.std()))
# AUC: 0.824 (0.041)
#AUC is relatively close to 1 and greater than 0.5




''' 4. Confusion Matrix '''
# Handy presentaion of accuracy of a model with two or more classes
# Predictions on the x-axis 
# Accuracy on the y-axis.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
    random_state=seed)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(Y_test, predicted)

print(matrix)
#[[141 21] 
# [ 41 51]]
# The majority of the predictions fall on the diagonal line of the matrix. (Correct predictions)




''' 5. Classification Report '''
# classification_report() function displays the precision, recall, F1-score, and support for each class

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
    random_state=seed)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)

print(report)
#             precision    recall  f1-score   support

#        0.0       0.77      0.87      0.82       162
#        1.0       0.71      0.55      0.62        92
#avg / total       0.75      0.76      0.75       254

