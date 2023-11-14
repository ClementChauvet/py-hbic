# Heterogeneous Biclustering
Heterogeneous Biclustering (Hbic) is a biclustering algorithm for data with heterogeneous types mixing numeric and categorical attributes natively. Biclustering algorithms are a family of unsuppervised machine learning algorithms that cluster the rows and the columns of a data table. The aims is to create meaningful and coherent sub-matrices when dealing with very high dimensional data where classical clustering algorithms fail to detect key-differences in a limited number of variables.

## Paper
This is the official python implementation repository of the hbic paper: &link and credentials&

## Tutorial
Examples of how to use the package can be found in hbic_examples.ipynb where we present the algorithm on synthetic and real datasets

### Installing 

Install this package using pip with the command

```
pip install git+https://github.com/ClementChauvet/heterogeneous_biclustering
```

### Usage

Here's how to identify biclusters for a numeric dataset

```
import heterogeneous_biclustering.hbic as hbic
import numpy as np

data = np.random.uniform(0, 10, (100, 100))
model = hbic.Hbic()
biclusters = model.fit_predict(data)
print(biclusters)
```

If you want to identify biclusters for a dataset with other types of variables, you will have to specify the var_type parameter. You can set it to "Categorical" if all your data are categorical or specify an array-like of the length of the number of columns specifying the type of each column between "Numeric" and "Categorical" such as: 

```
import heterogeneous_biclustering.hbic as hbic
import numpy as np

data = np.empty((100,100))
data[:, :50] = np.random.uniform(0, 10, (100, 50))
data[:, 50:] = np.random.randint(0, 10, (100, 50))
var_type = ["Numeric" if i < 50 else "Categorical" for i in range(100)]
model = hbic.Hbic()
biclusters = model.fit_predict(data)
print(biclusters)
```


If you don't specify a clustering reduction method, the algorithm will output all the biclusters found in the phase 1 of the algorithm. More information about this is present in the paper.
To have better performances it is advised to specify a clustering method chosen from :
+ "tree_selection"
+ "selection"
+ "merging"

The best algorithm will depend on the data you have but the most robust and stable from our experiences seemed to be tree_selection
If you know the number of biclusters to find, you can also specify it by passing it through the "n_clusters" parameter of hbic. The algorithm will then return at most this number of biclusters

Here is an example with a reduction and a specified number of biclusters to look for: 
```
import heterogeneous_biclustering.hbic as hbic
import numpy as np

data = np.random.uniform(0, 10, (100, 100))
model = hbic.Hbic(reduction = "tree_selection", n_clusters = 4)
biclusters = model.fit_predict(data)
print(biclusters)
```


### More advanced stuff

+ If the data is too noisy and the algorithm doesn't find as many biclusters as needed, lowering the nbins parameter will help find more biclusters but they will be less relevant and of lower quality. The opposite is also true, increasing nbins will augment the quality of identified biclusters but will lower their number and size
+ By default the algorithm will rule out any bicluster covering less than 2 columns or 2 rows or 3% of the total columns or 3% of the total rows. This can be changed through the min_rows and min_cols parameters. It's possible to pass it as a proportion of the number of columns or as a flat number.


