# Python_ML_Practice

These are my self-learning ML practice projects using Python(sklearn) following the instruction of "Machine Learning Mastery With Python BY Jason Brownlee".
Gnerally there are two parts in this repository. The first one is mainly about ML basics such as loading file and describing data. I will show this part in this README.  The second part is about three comprehensive ML projects on some well-known dataset using the basic knowledge in the first part. The details can be found using links below.


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
  The reason for preprocessing data is that different algorithms make different assumptions about data requiring different transformation. Here are some common processing techniques:
   - Transformation
     - Rescaling <br />
       To limit varying attributes ranges all between 0 and 1. Useful for weight-inputs regression/neural networks and kNN.
          
     - Standardization <br />
       To transform attributes with a Gaussian distribution to a standard Gaussian distribution (0 mean and 1 std). Useful for linear/logistic regression and LDA
     
   - Normalization <br />
     To rescaling each observation (row) to have a length of 1 (called a unit norm or a vector with the length of 1 in linear algebra). Useful for sparse dataset with varying attribute scales, weight-input neural network and kNN
   
   - Binarization <br />
     To transform data using a binary threshold.(1 for above threshold and 0 for below threshold)

4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request


## Acknowledgments

* Jason Brownlee
