''' 
Automate ML workflow with pipeline

'''

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]



''' 1. Pipeline of data preparation and modeling ''' 
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)



''' 2. Pipeline for feature selection '''
features = []
features.append(('pca', PCA(n_components = 3)))
features.append(('select_best', SelectKBest(k = 6)))
feature_union = FeatureUnion(features)

estimators = []
estimators.append(('feature_union', feature_union)) 
# embedding pipelines within pipelines
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)



# evaluate pipeline
kfold = KFold(n_splits = 10, random_state = 7)
results = cross_val_score(model, X, Y, cv= kfold)
print(results.mean())

# 1)   0.773462064252
# 2)   0.776042378674



