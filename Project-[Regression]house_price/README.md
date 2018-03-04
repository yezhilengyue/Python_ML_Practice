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
  ```
>>> corr_with_target[abs(corr_with_target).argsort()[::-1]]
LSTAT     -0.737663
RM         0.695360
PTRATIO   -0.507787
INDUS     -0.483725
TAX       -0.468536
NOX       -0.427321
CRIM      -0.388305
RAD       -0.381626
AGE       -0.376955
ZN         0.360445
B          0.333461
DIS        0.249929
CHAS       0.175260
Name: MEDV, dtype: float64
  ```
This shows that ```LSTAT``` has a good negative correlation with the output variable MEDV with a value of -0.737663.  
Then let's dig deeper at important correlations between input attributes. It might be interesting to select some strong correlations between attribute pairs.

```
# all except target
attrs = pearson.iloc[:-1,:-1]
# only important correlations and not auto-correlations
threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]).unstack().dropna().to_dict()
```
    attribute pair  correlation
0     (AGE, INDUS)     0.644779
1     (INDUS, RAD)     0.595129

```
unique_important_corrs = data.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])
# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['correlation']).argsort()[::-1]]
```
And the results:
```
>>> unique_important_corrs
    attribute pair  correlation
9       (RAD, TAX)     0.910228
15      (DIS, NOX)    -0.769230
10    (INDUS, NOX)     0.763651
18      (AGE, DIS)    -0.747881
11      (AGE, NOX)     0.731470
6     (INDUS, TAX)     0.720760
17    (DIS, INDUS)    -0.708027
21      (NOX, TAX)     0.668023
2        (DIS, ZN)     0.664408
7     (AGE, INDUS)     0.644779
23     (CRIM, RAD)     0.625505
3      (LSTAT, RM)    -0.613808
5       (NOX, RAD)     0.611441
8   (INDUS, LSTAT)     0.603800
19    (AGE, LSTAT)     0.602339
22    (INDUS, RAD)     0.595129
12    (LSTAT, NOX)     0.590879
0      (CRIM, TAX)     0.582764
16       (AGE, ZN)    -0.569537
14    (LSTAT, TAX)     0.543993
20      (DIS, TAX)    -0.534432
13     (INDUS, ZN)    -0.533828
4        (NOX, ZN)    -0.516604
1       (AGE, TAX)     0.506456
```
  - Visualization report
  First, let's take a loot at some unimodal Data Visualizations. 
    1) Histograms
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5BU%5Dhistograms.png)
    2) Density
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5BU%5Dline_graph.png)
    From the graph, we can see that some attributes have possible exponential and bimodal distributions. It also looks like ```NOX```, ```RM``` and ```LSTAT``` may be skewed Gaussian distributions, which might be helpful later with transforms.
    3) Boxplot
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5BU%5Dbox_plots.png)
    
  Then, we take a look at visualizations of interactions between variables.
    4) Scatter plot matrix
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5BM%5Dscatter_plots.png)
    
    5) Correlation matrix
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/corr_matrix.png)
    
 
## Evaluate Algorithms
   - Evaludation Baseline
In this problem, we will evaluate algorithms using the **Mean Squared Error (MSE)** metric with **10-fold cross validation**. Six algs with default settings will be checked include Linear Regression (LR), Lasso Regression (LASSO), ElasticNet (EN), Classification and Regression Trees (CART), Support Vector Regression (SVR) and k-Nearest Neighbors (KNN). <br />
Here is the testing results:
```
   LR: -21.3798557267 (std: 9.41426365698) 
   LASSO: -26.4235611084 (std: 11.6511099158)
   EN: -27.5022593507 (std: 12.3050222641)
   KNN: -41.8964883902 (std: 13.9016881498)
   CART: -22.9675067073 (std: 10.9475148015)
   SVR: -85.5183418393 (std: 31.9947982318) 
```
From above, we can see that LR has the lowest MSE, followed closely by CART. Let's observe it more clearly by looking at the distribution of scores across all cross validation folds by algorithm:
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/algs_cmpsn.png)
It looks that there is a tighter distribution of scores for CART. Surprisingly, kNN' score is very low. We uspect that the differing scales of the raw data may be negatively impacting the skill of some of the algorithms.
<br />

   - Data Wrangling
Generally, there are 3 type of data wrangling techniques:
   - **Feature selection** and removing the most correlated attributes.
   - **Normalizing** the dataset to reduce the effect of differing scales.
   - **Standardizing** the dataset to reduce the effects of differing distributions.
At the meantime, to avoid data leakage, we decide to use pipelines that standardize the data and build the model for each fold in the cross validation test harness.
```
#   ScaledLR: -21.3798557267 (9.41426365698)
#   ScaledLASSO: -26.6073135577 (8.97876148589)
#   ScaledEN: -27.9323721581 (10.5874904901)
#   ScaledKNN: -20.1076204878 (12.3769491508) 
#   ScaledCART: -24.4931573171 (10.4662755609)
#   ScaledSVR: -29.6330855003 (17.0091860524)
```

![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5BScaled%5Dalgs_cmpsn.png)
We can see that KNN has both a tight distribution of error and has the lowest score. Also, from the plot of standardized algorithm distribution scores, we know that KNN has both a tight distribution of error and has the lowest score. <br />
Therefore, we can focus on improving kNN algorithms.

# Improvement
  - Params tuning
  The default value for the number of neighbors in KNN is 7. We try to add k values from 1 to 21.
  ```
Best: -18.172137 using {'n_neighbors': 3}

-20.169640 (14.986904) with: {'n_neighbors': 1} 
-18.109304 (12.880861) with: {'n_neighbors': 3} 
-20.063115 (12.138331) with: {'n_neighbors': 5} 
-20.514297 (12.278136) with: {'n_neighbors': 7} 
-20.319536 (11.554509) with: {'n_neighbors': 9} 
-20.963145 (11.540907) with: {'n_neighbors': 11} 
-21.099040 (11.870962) with: {'n_neighbors': 13} 
-21.506843 (11.468311) with: {'n_neighbors': 15} 
-22.739137 (11.499596) with: {'n_neighbors': 17} 
-23.829011 (11.277558) with: {'n_neighbors': 19} 
-24.320892 (11.849667) with: {'n_neighbors': 21}
  ```
The best for k (n neighbors) is 3 with a mean squared error of -18.172137.

  - Ensemble
Another way that we can improve the performance of algorithms on this problem is by using ensemble methods. Here we will evaluate four different ensemble machine learning algorithms, two boosting (AdaBoost and Gradient Boosting) and two bagging methods (Random Forests and Extra Trees):
```
   ScaledAB: -14.957611 (6.579588)
   ScaledGBM: -10.054631 (4.522576)
   ScaledRF: -13.718030 (7.982746)
   ScaledET: -10.530828 (6.818533)
```
![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BRegression%5Dhouse_price/%5Bscaled%5Dknn_ensemble.png)
It looks like Gradient Boosting has a better mean score, it also looks like Extra Trees has a similar distribution and perhaps a better median score. We can probably do better, given that the ensemble techniques used the default parameters. Then we will look at tuning the Gradient Boosting to further lift the performance using grid search. <br />
Often, the larger the number of boosting stages, the better the performance but the longer the training time. The default number of boosting stages to perform (n estimators) is 100. 
```
Best: -9.35647101741 using {'n_estimators': 400}

   -10.812167 (4.724394) with: {'n_estimators': 50}
   -10.040857 (4.441758) with: {'n_estimators': 100}
   -9.694096 (4.275762) with: {'n_estimators': 150}
   -9.539706 (4.270637) with: {'n_estimators': 200}
   -9.448764 (4.262603) with: {'n_estimators': 250}
   -9.429946 (4.273791) with: {'n_estimators': 300}
   -9.369824 (4.254108) with: {'n_estimators': 350}
   -9.356471 (4.267837) with: {'n_estimators': 400}
```
We can see that the best configuration was n estimators=400 with a mean squared error of -9.356471, about 0.65 units better than the untuned method.

# Finalize model and Make Predictions

