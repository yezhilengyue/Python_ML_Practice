# Sonar Mine and Rock Report
This project is about predict metal or rock objects from sonar return data using Sonar Mines v.c. Rocks dataset.

## Data Analysis
   - **Statistics** <br />
This time we load the data without header because we notice that the class attribute (the last column) is meaningless. So we set ```data = pd.readcsv(filename, header = None)``` to avoid file loading code taking the first record as the column names.
```
print(data.shape)
print(data.dtypes)
print(data.head(10))
print(data.groupby(60).size())
```
Using above command, we get a statistical glimps of data. There are 208 instances with 61 attributes including class attribute. All attributes are numeric except class attribute (object type). The data has the same range, but differing mean values. This indicates that standardization could help. Also, from the breakdown of class values, we can see that in this dataset, there are 111 mines and 97 rocks, an relatively balanced class distribution.

   - **Visualization** <br />
     - Histogram
     ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/histgram.png)
     There are quite a lof Gaussian-like distributions (e.g. 37, 39) and some exponential-like distributions (e.g. 30, 48). To see it clearly, let's take a look at density plots.
     
     - Density plots
     ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/density.png)
     We notice that many of the attributes have a skewed distribution. So let's did deeper to look at a box plots to get some ideas about the spread of values.
     
     - Box plots (Code modification for this plots)
     ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/boxplots.png)
     Obviously, attributes do have quite different spread. This again comfirms our guess that it is helpful to apply standardization to data for modeling to get all means lined up. Next, let's visualize the correlations between the attributes.
     
     - Correlation Matrix
     ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/correlation.png)
     We can see that the color around diagonal are much yellower which means that strong positive correlations between neighboring attributes. Also there is a part with dark purple around the center of the graph representing for negative correlations. "This makes sense if the order of the attributes refers to the angle of sensors for the sonar chirp."
     
     
## Evaluation Algorithm
   - **Split-out validation dataset, Test options and Evaluations metric** <br />
    We will use a validation hold-out set holding back from our analysis and modeling. This is a smoke test that we can use to see if we messed up and to give us confidence on our estimates of accuracy on unseen data. Specifically, it is 80% for modeling and 20% for validation, with 10-fold cross validation on accuracy evaluation metric.
```
seed = 7

validation_size_ratio = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size_ratio, random_state = seed)

num_folds = 10
scoring = 'accuracy'
```

   - **Baseline** <br />
   Create a baseline of performance on this problem and spot-check a number of different algorithms.
     - Logistic Regression (LR)
     - Linear Discriminant Analysis (LDA)
     - Classification and Regression Trees (CART)
     - Support Vector Machines (SVM)
     - Gaussian Naive Bayes (NB)
     - k-Nearest Neighbors (KNN).
```
models = []
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))
```
  With all default tuning parameters, we compare these algorithms by calculating the mean and standard deviation of accuracy for each algorithm. <br />
     
```
LR: 0.782721 (std: 0.093796)
LDA: 0.746324 (std: 0.117854)
KNN: 0.808088 (std: 0.067507)
CART: 0.740809 (std: 0.118120)
NB: 0.648897 (std: 0.141868)
SVM: 0.608824 (std: 0.118656)
```

   The results suggest that both Logistic Regression and k-Nearest Neighbors may be worth further study. Besides the mean accuracy values, let's take a look at the distribution of accuracy values calculated across cross-validation folds using box plots.
     ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/algs_cmpsn.png)
     The results show a tight distribution for KNN which is encouraging, suggesting low variance. The poor results for SVM are surprising.<br />
     We guess that this is probably credit to the varied distribution of the attributes which have an effect on the accuracy of algorithms such as SVM. Therefore, in the next step, we repeat this spot-check with standardized data.

     
   - **Data transformation (Standardization)** <br />
    Suspecting negative effect of varied distributions of the raw data on some algorithm, we standardize the training data by setting each attribute with 0 mean and 1 standard deviation.<br />
    Also, to avoid data leakage, we use piplelines to standardize data and build model for each fold in the cross validation test.
```
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
```
   The results:
   ```
ScaledLR: 0.734191 (0.095885)
ScaledLDA: 0.746324 (0.117854)
ScaledKNN: 0.825735 (0.054511)      
ScaledCART: 0.741176 (0.105601)
ScaledNB: 0.648897 (0.141868)
ScaledSVM: 0.836397 (0.088697)      
   ```
   Again, kNN is still doing well, even better than before. Moreover, after standardization, the performance of SVM improves a lot.
   ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/%5BStandardization%5Dalgs_cmpsn.png)
   The box plot show the promise of SVM and kNN which are worth further study with some tuning techniques.
   
   
## Algorithm Tuning
   - Params tuning with kNN <br />
     Below we try all odd values of k from 1 to 21, covering the default value of 7 using grid search. Each k value is evaluated using 10-fold cross validation on the training standardized dataset.
```
Best: 0.849398 using {'n_neighbors': 1}

0.849398 (0.059881) with: {'n_neighbors': 1}
0.837349 (0.066303) with: {'n_neighbors': 3}
0.837349 (0.037500) with: {'n_neighbors': 5}
0.765060 (0.089510) with: {'n_neighbors': 7}
0.753012 (0.086979) with: {'n_neighbors': 9}
0.734940 (0.105836) with: {'n_neighbors': 13}
0.710843 (0.078716) with: {'n_neighbors': 17}
0.722892 (0.084555) with: {'n_neighbors': 19}
0.710843 (0.108829) with: {'n_neighbors': 21}
```
   Interestingly, the optimal k is 1. This means that the algorithm will make predictions using the most similar instance in the training dataset alone. 
   - Params tuning with SVM <br />
       There are two parameters (the value of *C* and the type of kernel) of the SVM we can tune. By default, the SVM (the SVC class) uses the Radial Basis Function (RBF) kernel with a *C* value set to 1.0.
```
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
```
  We will try a number of simpler kernel types and C values with less bias and more bias (less than and more than 1.0 respectively).
```
Best: 0.867470 using {'C': 1.5, 'kernel': 'rbf'}

0.759036 (0.098863) with: {'C': 0.1, 'kernel': 'linear'}
0.530120 (0.118780) with: {'C': 0.1, 'kernel': 'poly'}
0.572289 (0.130339) with: {'C': 0.1, 'kernel': 'rbf'}
0.704819 (0.066360) with: {'C': 0.1, 'kernel': 'sigmoid'}
0.746988 (0.108913) with: {'C': 0.3, 'kernel': 'linear'}
0.644578 (0.132290) with: {'C': 0.3, 'kernel': 'poly'}
0.765060 (0.092312) with: {'C': 0.3, 'kernel': 'rbf'}
0.734940 (0.054631) with: {'C': 0.3, 'kernel': 'sigmoid'}
0.740964 (0.083035) with: {'C': 0.5, 'kernel': 'linear'}
0.680723 (0.098638) with: {'C': 0.5, 'kernel': 'poly'}
0.789157 (0.064316) with: {'C': 0.5, 'kernel': 'rbf'}
0.746988 (0.059265) with: {'C': 0.5, 'kernel': 'sigmoid'}
0.746988 (0.084525) with: {'C': 0.7, 'kernel': 'linear'}
0.740964 (0.127960) with: {'C': 0.7, 'kernel': 'poly'}
0.813253 (0.084886) with: {'C': 0.7, 'kernel': 'rbf'}
0.753012 (0.058513) with: {'C': 0.7, 'kernel': 'sigmoid'}
0.759036 (0.096940) with: {'C': 0.9, 'kernel': 'linear'}
0.771084 (0.102127) with: {'C': 0.9, 'kernel': 'poly'}
0.837349 (0.087854) with: {'C': 0.9, 'kernel': 'rbf'}
0.753012 (0.073751) with: {'C': 0.9, 'kernel': 'sigmoid'}
0.753012 (0.099230) with: {'C': 1.0, 'kernel': 'linear'}
0.789157 (0.107601) with: {'C': 1.0, 'kernel': 'poly'}
0.837349 (0.087854) with: {'C': 1.0, 'kernel': 'rbf'}
0.753012 (0.070213) with: {'C': 1.0, 'kernel': 'sigmoid'}
0.771084 (0.106063) with: {'C': 1.3, 'kernel': 'linear'}
0.819277 (0.106414) with: {'C': 1.3, 'kernel': 'poly'}
0.849398 (0.079990) with: {'C': 1.3, 'kernel': 'rbf'}
0.710843 (0.076865) with: {'C': 1.3, 'kernel': 'sigmoid'}
0.759036 (0.091777) with: {'C': 1.5, 'kernel': 'linear'}
0.831325 (0.109499) with: {'C': 1.5, 'kernel': 'poly'}
0.867470 (0.090883) with: {'C': 1.5, 'kernel': 'rbf'}
0.740964 (0.063717) with: {'C': 1.5, 'kernel': 'sigmoid'}
0.746988 (0.090228) with: {'C': 1.7, 'kernel': 'linear'}
0.831325 (0.115695) with: {'C': 1.7, 'kernel': 'poly'}
0.861446 (0.087691) with: {'C': 1.7, 'kernel': 'rbf'}
0.710843 (0.088140) with: {'C': 1.7, 'kernel': 'sigmoid'}
0.759036 (0.094276) with: {'C': 2.0, 'kernel': 'linear'}
0.831325 (0.108279) with: {'C': 2.0, 'kernel': 'poly'}
0.867470 (0.094701) with: {'C': 2.0, 'kernel': 'rbf'}
0.728916 (0.095050) with: {'C': 2.0, 'kernel': 'sigmoid'}
```
  We can see the most accurate configuration was SVM with an RBF kernel and a C value of 1.5. The accuracy 86.7470% is seemingly better than what KNN could achieve.
  
   - Ensemble tuning <br />
  Besides parameter tuning, another algs improvement technique is by using ensemble methods. We will use 2 boosting (AdaBoost and Gradient Boosting) and 2 bagging (Random Forests and Extra Trees) methods. <br />
  *"No data standardization is used in this case because all four ensemble algorithms are based on decision trees that are less sensitive to data distributions."*
```
ensembles = []
ensembles.append(('AB', AdaBoostClassifier())) 
ensembles.append(('GBM', GradientBoostingClassifier())) 
ensembles.append(('RF', RandomForestClassifier())) 
ensembles.append(('ET', ExtraTreesClassifier()))
```
 The results:
```
AB: 0.819853 (0.058293)
GBM: 0.823897 (0.101025)
RF: 0.795956 (0.095662)
ET: 0.795221 (0.083918)
```
![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification-Binary%5Dsonar_rock/Ensemble%5Dalgs_cmpsn.png)
  
  
  
  
