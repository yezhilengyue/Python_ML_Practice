# Bike Sharing Trends
The Capital Bike Sharing dataset from UCI contains information about a bike sharing program underway in Washington DC. 
In other words, given this augmented (bike sharing details along with weather information) dataset, 
  can we forecast bike rental demand for this program?

## Data Analysis
   - **Statistics** <br />
This time we load the data without header because we notice that the class attribute (the last column) is meaningless. So we set ```data = pd.readcsv(filename, header = None)``` to avoid file loading code taking the first record as the column names.
```
print(data.shape)
print(data.dtypes)
print(data.head(10))
print(data.groupby(60).size())
```
