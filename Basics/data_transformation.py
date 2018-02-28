# Rescale data (between 0 and 1)
from pandas import read_csv
from numpy import set_printoptions

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
dataframe = read_csv(filename, names=names)
array = dataframe.values

# separate array into input and output component
X = array[:, 0:8]
Y = array[:, 8]

# 1. Rescale data (between 0 and 1)
'''
- useful for gradient descent
- weight inputs like regression and neural networks
- distance measures like k-Nearest Neighbors.
'''
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)


# 2. Standardize data (0 mean, 1 stdev)
'''
- assume a Gaussian distribution in the input variables
- work better with rescaled data
- linear regression
- logistic regression
- linear discriminate analysis
'''
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# 3. Normalize data (length of 1)
'''
- useful for sparse dataset
- weight input values such as neural networks
- distance measures such as k-Nearest Neighbors
'''
scaler = Normalizer().fit(X)
rescaledX = scaler.transform(X)

# 4. Binarizing
'''
- Above threshold marked 1; Below or equal threshold marked 0.
'''
binarizer = Binarizer(threshold = 0.0).fit(X)
binaryX = binarizer.transform(X)


# Summerize transformed data
set_printoptions(precision = 3)
print(rescaledX[0:5,:])