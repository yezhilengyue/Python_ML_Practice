# Iris Flower Classification
Iris flower dataset perfhaps is the best known data set in the classification literature. It aims to classify iris flowers among three species (setosa, versicolor or virginica) based on measurements of length and width of sepals and petals.

## 1. Summarize the Dataset with Statistics and Visualization
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
   It looks like perhaps two of the input variables have a Gaussian distribution
   ![lt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/histgram.png)
   
   - **Histogram**
