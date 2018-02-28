'''
Feature Selection
Benefits of feature selection:
    1. reduce overfitting
    2. improve accuracy
    3. reduce training time

Statistical tests can be used to select those features that have the strongest relationship with the output variable.

'''

from pandas import read_csv
from numpy import set_printoptions


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA


from sklearn.ensemble import ExtraTreesClassifier

# load data
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]


''' 1. Univariate extaction '''
# use chi-squared statistical test for non-negative features
test = SelectKBest(score_func = chi2, k=4) 
fit = test.fit(X,Y)

# Summarize scores
set_printoptions(precision = 3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])



''' 2. Recursive Feature Elimination (RFE) '''
# works by recursively removing attributes AND building a model on these attributes
# e.g. uses RFE with the logistic regression algorithm to select the top 3 features
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X,Y)

print("Num Features: " + str(fit.n_features_))
print("Selected Features: " + str(fit.support_))
print("Feature Ranking: "+  str(fit.ranking_))




''' 3. Principal Component Analysis '''
# A data reduction technique
# PCA uses linear algebra to transform the dataset into a compressed form
pca = PCA(n_components = 3)
fit = pca.fit(X)

print("Explained Variance: " + str(fit.explained_variance_ratio_))
print(fit.components_)



''' 4. Feature Importance '''
# Bagged decision trees like Random Forest
# Extra Trees can be used to estimate the importance of features.
model = ExtraTreesClassifier()
model.fit(X,Y)

print(model.feature_importances_)