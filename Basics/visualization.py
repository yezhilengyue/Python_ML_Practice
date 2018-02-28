# Understand data with simple visualization

from matplotlib import pyplot
from pandas import read_csv
import numpy
import pandas.plotting.scatter_matrix

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)

# 1. Univariate Plots
# 1.1 Histogram
data.hist()
pyplot.show()

# 1.2 Density plots
data.plot(title = 'Density Plot', kind = 'density',subplots = True, layout=(3,3), sharex = False)
pyplot.show()

# 1.3 Box plots
data.plot(title = 'Box Plot', kind = 'box', subplots = True, layout=(3,3), sharex = False, sharey = False)
pyplot.show()


# 2. Multivariate plots
# 2.1 Correlation matrix plot
correlations = data.corr()

fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)

# Modify this part for generic
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax_set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
#########################

pyplot.show()


# 2.2 Scatter plots
scatter_matrix(data)
pyplot.show()
