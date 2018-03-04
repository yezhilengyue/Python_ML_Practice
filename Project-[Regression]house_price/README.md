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
  ```
  Here we have e 506 instances to work with and can confirm the data has 14 attributes including the output attribute MEDV.
  ```
  print(dataset.dtypes)
  ```
  We can see that all of the attributes are numeric, all real values (float) except ```CHAS``` and ```RAD``` as integers
  ```
  print(dataset.head(20))
  ```
  Take a peek at the first 20 rows of the data
  ```
  print(dataset.describe())
  ```
  
  - Visualization report
