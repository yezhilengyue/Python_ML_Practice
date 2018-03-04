# Boston House Price Prediction Report
The Boston Housing Dataset consists of price of houses in various places in Boston. Each record in the database describes a Boston suburb or town.

## Loading the Dataset
Some useful python packages:
  - ```numpy``` - efficient numerical computations
  - ```pandas``` - data structures for data analysis
  - ```scikit-learn``` - machine learning algorithms, dataset access
  - ```matplotlib``` - plotting (both interactive and to files)
  - ```seaborn``` - extra plot types, elegant and readable plot style
  
## Summarize Data
  - Statistical report
  ```
  print(dataset.shape)
  print(dataset.dtypes)
  print(dataset.head(20))
  print(dataset.describe())
  ```
  Here we have e 506 instances to work with and can confirm the data has 14 attributes including the output attribute MEDV. Also, all of the attributes are numeric, all real values (float) except ```CHAS``` and ```RAD``` as integers.
  Next, let’s now take a look at the correlation between all of the numeric attributes. Pandas offers us out-of-the-box three various correlation coefficients via ```DataFrame.corr()``` functions: standard Pearson correlation coefficient, Spearman rank correlation, Kendall Tau correlation coefficient.
  ```
>>> data.corr(method='pearson')
                        ...
             CRIM        ZN     INDUS      CHAS   
CRIM     1.000000 -0.200469  0.406583 -0.055892
ZN      -0.200469  1.000000 -0.533828 -0.042697
INDUS    0.406583 -0.533828  1.000000  0.062938
CHAS    -0.055892 -0.042697  0.062938  1.000000 
                      ...
  ```
  Besides correlation between attributes, we'd like to know the correlation between input attributes and the target one, ie. how each input attribute is able to predict the target. It is called predictivity.
  ```
>>> data.corr(method='pearson')
# assume target attr is the last, then remove corr with itself
>>> corr_with_target = pearson.ix[-1][:-1]
# attributes sorted from the most predictive
>>> predictivity = corr_with_target.sort_values(ascending=False)
  ```
  And the result:
  ```
RM         0.695360
ZN         0.360445
B          0.333461
DIS        0.249929
CHAS       0.175260
AGE       -0.376955
RAD       -0.381626
CRIM      -0.388305
NOX       -0.427321
TAX       -0.468536
INDUS     -0.483725
PTRATIO   -0.507787
LSTAT     -0.737663
Name: MEDV, dtype: float64
  ```
  To find the attribute with strongest correlation with output, it would be better to sort the correlations by the absolute value:
  
  
  - Visualization report
