'''
Ensemble: To improve performance

    - 1.The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.
    
    - 2. Two families of ensemble methods are usually distinguished: Averaging and Boosting
    
    - 3. Three most popular methods for combining the predictions from different models:
        # Bagging 
            1) Bagged Decision Tree
            2) Random Forest
            3) Extra Tree
            
        # Boosting
        # Voting

'''

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)


''' # 1. Bagging Algorithm'''
# Bootstrap Aggregation (Bagging) involves taking multiple smaples from training dataset and training a model for each sample.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


'''     1.1 Bagged Decision Tree '''
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator = cart, n_estimators = num_trees, random_state = seed)


'''     1.2 Random Forest '''
num_trees = 100
max_features = 3
model = RandomForestClassifier(n_estimators = num_trees, max_features = max_features)


'''     1.3 Extra Trees '''
num_trees = 100
max_features = 7
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)


# Output results
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())





''' 2. Boosting Algorithm '''
# Boosting ensemble algorithms creates a sequence of models that attempt to correct the mistakes of the models before them in the sequence.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

'''     2.1 AdaBoost '''
# It generally works by weighting instances in the dataset by how easy or difficult they are to classify, allowing the algorithm to pay or less attention to them in the construction of subsequent models
num_trees = 30
model = AdaBoostClassifier(n_estimators = num_trees, random_state = seed)



'''     2.2 Stochastic Gradient Boosting '''
num_trees = 100
model = GradientBoostingClassifier(n_estimators = num_trees, random_state = seed)


# Output results
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())





''' 3. Voting Ensemble '''
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.

from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)

results = cross_val_score(ensemble, X, Y, cv = kfold)
print(results.mean())






