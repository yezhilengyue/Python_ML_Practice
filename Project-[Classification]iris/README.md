# Iris Flower Classification
Iris flower dataset perfhaps is the best known data set in the classification literature. It aims to classify iris flowers among three species (setosa, versicolor or virginica) based on measurements of length and width of sepals and petals.

## 1. Summary with Statistics and Visualization
When come across new dataset, we can take a look at its dimensions, first several rows, attribute type, statistical summary of each attribute and class distribution.
```
print(data.shape)
print(data.head(10))
print(data.dtype)
print(data.describe())
print(data.groupby('class').size())
```
In summary, in this iris dataset, we have 150 instances, 5 attributes (`sepal-length`, `sepal-width`, `petal-length`, `petal-width` and `class`) and 3 classes (`sepal-length`, `Iris-versicolor` and `Iris-virginica`). Among these 5 attributes, `sepal-length`, `sepal-width`, `petal-length` and `petal-width`  are numeric while `class` attribute is string data. Besides, in each category there are 50 instances.

After having a basic idea about the data, let's dig deeper through visualizations. Generally, univariate plots help to better understand each attribute, while multivariate plots is good to understand the relationships between attributes. <br />
   - **Box plots** <br />
   ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/boxplots.png)
   
   - **Histogram** <br />
   It looks like perhaps two of the input variables (`sepal-length` and `sepal-width`) have a Gaussian distribution
   ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/histgram.png)
   
   - **Scatter plot matrix**
   Take a look at the interactions between the variables. It shows that there is a high correlation and a predictable relationship.
   ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/scatter_plots.png)
   
   
   
## 2. Algorithms Evaluationg
   Now it's time to create model and make prediction on unseen data.
   - **Separate dataset and Test**
   In this problem, we will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation datas. Also, it will be a 10-fold cross validation to estimate accuracy.
   
   - **Build models**
   From the plots, we deduce that some of the classes are partially linearly separable in some dimensions. Let’s evaluate six different algorithms: <br />
     1) Logistic Regression (LR)
     2) Linear Discriminant Analysis (LDA)
     3) k-Nearest Neighbors (KNN).
     4) Classification and Regression Trees (CART).
     5) Gaussian Naive Bayes (NB).
     6) Support Vector Machines (SVM).
   *We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits.*
   ```
    LR: 0.966667 (std: 0.040825)
    LDA: 0.975000 (std: 0.038188)
    KNN: 0.983333 (std: 0.033333) **
    CART: 0.975000 (std: 0.038188)
    NB: 0.975000 (std: 0.053359)
    SVM: 0.981667 (std: 0.025000)
   ```
   It looks that KNN achieves the highest accuracy score. Draw this result using boxplot.
    ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/algorithm_comparison.png)
   
   
   - **Make Prediction**   
   Since kNN is the most accurate model for iris classification problem, we run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
 ```
 0.9  
 ```
  The accuracy is 0.9.
```
   [[ 7 0 0] 
    [ 0 11 1] 
    [ 0 2 9]] 
```
 The confusion matrix shows that there are 3 errors made
```
                     precision    recall  f1-score   support

        Iris-setosa       1.00      1.00      1.00         7
    Iris-versicolor       0.85      0.92      0.88        12
     Iris-virginica       0.90      0.82      0.86        11

        avg / total       0.90      0.90      0.90        30
 ```
    
