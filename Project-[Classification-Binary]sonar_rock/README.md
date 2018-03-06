# Sonar Mine and Rock Report
This project is about predict metal or rock objects from sonar return data using Sonar Mines v.c. Rocks dataset.

## Data Analysis
   - Statistics
This time we load the data without header because we notice that the class attribute (the last column) is meaningless. So we set ```data = pd.readcsv(filename, header = None)``` to avoid file loading code taking the first record as the column names.
```
print(data.shape)
print(data.dtypes)
print(data.head(10))
print(data.groupby(60).size())
```
Using above command, we get a statistical glimps of data. There are 208 instances with 61 attributes including class attribute. All attributes are numeric except class attribute (object type). The data has the same range, but differing mean values. This indicates that standardization could help. Also, from the breakdown of class values, we can see that in this dataset, there are 111 mines and 97 rocks, an relatively balanced class distribution.

   - Visualization
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
     
     
     
     
