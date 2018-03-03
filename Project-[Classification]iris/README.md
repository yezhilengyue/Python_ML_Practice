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
In summary, in this iris dataset, we have 150 instances and 5 attributes. Each category has 50 instances.

After having a basic idea about the data, let's dig deeper through visualizations. Generally, univariate plots help to better understand each attribute, while multivariate plots is good to understand the relationships between attributes. <br />
   - Box plots <br />
   ![alt text](https://github.com/yezhilengyue/Python_ML_Practice/blob/master/Project-%5BClassification%5Diris/boxplots.png)
   
   - Histogram
   
