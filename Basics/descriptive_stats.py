from pandas import read_csv
from pandas import set_option

filename = "pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename,names = names)

peek = data.head()

shape = data.shape

types = data.dtypes

set_option('display.width',100)
set_option('precision',3)

# 1. To make data more readable
description = data.describe()
print(description)


# 2. Class Distribution
# On classification problems you need to know how balanced the class values are.

class_counts = data.groupby('class').size()
print(class_counts)

'''
class
0    500
1    268
'''

# 3. Correlations
# Some machine learning algorithms like linear and logistic regression can suffer poor performance if there are highly correlated attributes in your dataset.

correlations = data.corr(method = 'pearson')
print(correlations)

'''
        preg   plas   pres   skin   test   mass   pedi    age  class
preg   1.000  0.129  0.141 -0.082 -0.074  0.018 -0.034  0.544  0.222
plas   0.129  1.000  0.153  0.057  0.331  0.221  0.137  0.264  0.467
pres   0.141  0.153  1.000  0.207  0.089  0.282  0.041  0.240  0.065
skin  -0.082  0.057  0.207  1.000  0.437  0.393  0.184 -0.114  0.075
test  -0.074  0.331  0.089  0.437  1.000  0.198  0.185 -0.042  0.131
mass   0.018  0.221  0.282  0.393  0.198  1.000  0.141  0.036  0.293
pedi  -0.034  0.137  0.041  0.184  0.185  0.141  1.000  0.034  0.174
age    0.544  0.264  0.240 -0.114 -0.042  0.036  0.034  1.000  0.238
class  0.222  0.467  0.065  0.075  0.131  0.293  0.174  0.238  1.000
'''


# 4. Skewness
# The skew result show a positive (right) or negative (left) skew. Values closer to zero show less skew.
skew = data.skew()
print(skew)

'''
preg     0.902
plas     0.174
pres    -1.844
skin     0.109
test     2.272
mass    -0.429
pedi     1.920
age      1.130
class    0.635
'''
