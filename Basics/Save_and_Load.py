'''
Save and Load ML modules

In this script, you will learn how to save your model to file and load it later in order to make predictions. 

Serialize ML algorithms and save the serialized format to a file.
Load the file to deserialize models and use it to make new pridictions.

    - Pickle
    - Joblib

'''

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

seed = 7

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] dataframe = read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = seed)

# fit the model
model = LogisticRegressino()
model.fit(X_train, Y_train)



''' 1. Pickle '''
from pickle import dump
from pickle import load


# Save the model to disk
filename = 'finalized_model.sav'
dump(model, open(filename, 'wb'))

# some time later...


# load the model from disk
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)

print(result)




''' 2. Joblib '''
# This can be useful for some machine learning algorithms that require a lot of parameters or store the entire dataset

from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

# save the model to disk
filename = 'finalized_model.sav'
dump(model, filenmae)

# some time later


# load the model from disk
loaded_model = load(filename)
result = loaded_model.score(X_test,Y_test)

print(result)
