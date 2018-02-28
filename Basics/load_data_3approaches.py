# Python-DS exe on Pima Indians dataset
# This dataset is to describe the medical records for Pima Indians and whether or not each patient will have an onset of diabets within five years


# Load csv file with Python stdlib
import csv
import numpy

filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'r')
reader = csv.reader(raw_data, delimiter = ',', quoting = csv.QUOTE_NONE)
x = list(reader)
data = numpy.array(x).astype('float')

# Load csv file with NumPy
from numpy import loadtxt
filename = 'pima-indians-diabetes.data.csv'
raw_data = open(filename, 'r')
data = loadtxt(raw_data,delimiter = ',')

# Load dataset directly from a URL 
from numpy import loadtxt
import urllib.request
url = 'https://goo.gl/vhm1eU'
raw_data = urllib.request.urlopen(url)
dataset = loadtxt(raw_data, delimiter = ',')

# Load dataset with Pandas
# The function returns a pandas.DataFrame
from pandas import read_csv
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names = names)

