# Python_ML_Practice

These are my self-learning ML practice projects using Python(sklearn) following the instruction of "Machine Learning Mastery With Python BY Jason Brownlee".
Gnerally there are two parts in this repository. The first one is mainly about ML basics such as loading file and describing data. In this README file, basics exercises are on Pima Indians onset of diabetes dataset.  The second part is about three comprehensive ML projects on some well-known dataset using the basic knowledge in the first part. The details can be found using links below.


* [Classification Problem](https://github.com/yezhilengyue/Python_ML_Practice/tree/master/Projects/%5BClassification%5D_iris) - Iris Flower Dataset
* [Regression Problem](https://github.com/yezhilengyue/Python_ML_Practice/tree/master/Projects/%5BRegression%5Dhouse_price) - Boston Housing Price Dataset
* [Binary Classification Problem](https://github.com/yezhilengyue/Python_ML_Practice/tree/master/Projects/%5BClassification-bi%5Dsonar_rock) - Sonar, Mines and Rocks Dataset


## Getting Started with Basics

Overall, when given a new ML project, the workflow could be as following:
1. **Define problem**
  Investigate and characterize the problem and clarify the project goal.

2. **Summarize data**
  Use descriptive statistics and visualization techniques to get a grasp of data. 
   - Descriptive Statistics <br />
     data dimension, type, attribute features (count, mean, std, min/max, percentiles), class categories, correlations between attributes, skew of univariate distributions
     
   - Visualization <br />
     univariate plots(histograms, density plot, boxplot), multivariate plots(correlation matrix plot, scatter plot matrix)

3. **Data preprocessing [Incompleted]**
   - Transformation
   The reason for preprocessing data is that different algorithms make different assumptions about data requiring different transformation. Here are some common processing techniques:
     - Rescaling <br />
     To limit varying attributes ranges all between 0 and 1. Useful for weight-inputs regression/neural networks and kNN.
          
     - Standardization <br />
     To transform attributes with a Gaussian distribution to a standard Gaussian distribution (0 mean and 1 std). Useful for linear/logistic regression and LDA
     
     - Normalization <br />
     To rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). Useful for sparse dataset with varying attribute scales, weight-input neural network and kNN
   
     - Binarization <br />
     To transform data using a binary threshold.(1 for above threshold and 0 for below threshold)
     
   - Feature Selection <br />
   Irrelevant or partially relevant features can negatively impact model performance, such as decreasing the accuracy of many models. Feature Selection is to select features that contribute most to the prediction variable or output in which you are interested. It can help reduce overfiting, improve accuracy, reduce training time. Here are some common processing techniques:
     - Statistical Test Selection with *chi-2* <br />
     To select those features that have the strongest relationship with output variable
     
     - Recursive Feature Elimination (RFE) <br />  
     To recursively removing attributes and building a model on those attributes that remain.
     
     - Principal Component Analysis (PCA) <br />
     A kind of data reduction technique. It uses linear algebra to transform the dataset into a compressed form and choose the number of dimensions or principal components in the transformed result.
     
     - Feature importance <br />
     To use bagged decision trees such as Random Forest and Extra Trees to estimate the importance of features.

4. **Algorithm Evaluation**
   - Separate train/test dataset (Resampling) <br />
   In most cases, *k-fold* Cross Validation technique (e.g. k = 3, 5 or 10) will be used to estimate algorithm performance with less variance. At first, the dataset will be splited into *k* parts. Then the algorithm is trained on *k-1* folds with one held back and tested on the held back fold. Repeatedly, each fold of the dataset will be given a chance to be the held back test set. After all these, you can summarize using the mean and std of such *k* different performance scores.
     
   - Performance Metrics <br />
   Choice of metrics influence how the performance of ML algorithms is measure and compared, as it represents how you weight the importance of different characteristics in the output results and ultimate choice of which algorithm to choose.
     - For Classification Problem
       - Classification Accuracy
       - Logorithmic Loss
       - Area Under ROC Curve
   

5. Create a new Pull Request


## Acknowledgments

* Jason Brownlee
