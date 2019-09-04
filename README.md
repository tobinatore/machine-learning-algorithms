# Machine learning algorithms
This repo contains my implementations of various machine learning algorithms using sklearn and Python.


# Prerequisites

To run the scripts found in this repository, you'll need to install the following Python packages:

 - sklearn
 - matplotlib
 - numpy
 - pandas

Additionally, if you want a graphical reprentation of the decision tree, you'll need to get GraphViz from [here](https://graphviz.gitlab.io/download/). Make sure the dot.exe from GraphViz's bin directory is in your PATH, or else the script will throw a FileNotFound error.

## Algorithms currently implemented:

 1. [Linear Regression](#lin-reg)
 2. [K-Nearest Neighbour](#knn)
 3. [Support Vector Machine](#svm)
 4. [K-Means Clustering](#k-means)
 5. [Decision Trees](#dec-trees)

   <a id="lin-reg"></a>
## Linear Regression
![Random data points and their linear regression.](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)  
Image taken from [wikimedia](https://commons.wikimedia.org/wiki/File:Linear_regression.svg).

**Definition:**
> **Linear Regression** is the process of finding a line that best fits the data points available on the plot, so that we can use it to predict output values for inputs that are not present in the data set we have, with the belief that those outputs **would** fall on the line.
> -- [Anas Al-Masri](https://towardsdatascience.com/how-does-linear-regression-actually-work-3297021970dd)

 **Problem solved using the algorithm:** Estimating the grades of students in G3 based on their results in G1 and G2 as well as their absences during the academic year, their failures and the time studied per week.  

Besides predicting the final grade of a student, the linear_regression.py can also plot the relationship between two sets of data.

**Accuracy:** ~75% to ~90%

<a id="knn"></a>
## K-Nearest Neighbour
![The 1NN classification map based on the CNN extracted prototypes.](https://upload.wikimedia.org/wikipedia/commons/e/e9/Map1NNReducedDataSet.png)  
Image taken from [wikimedia](https://commons.wikimedia.org/wiki/File:Map1NNReducedDataSet.png) and made by user Agor153.

**Definition:**   
>KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).
>-- [Onel Harrison](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

 **Problem solved using the algorithm:** Classify the acceptability of a car based on buying and maintenance price, number of doors, capacity in terms of persons, size of luggage boot and the car's safety rating.  


**Accuracy:** ~95% - ~98%

<a id="svm"></a>
## Support Vector Machine
 ![Kernel machines are used to compute non-linearly separable functions into a higher dimension linearly separable function.](https://upload.wikimedia.org/wikipedia/commons/f/fe/Kernel_Machine.svg)  
  Image taken from [wikimedia](https://commons.wikimedia.org/wiki/File:Kernel_Machine.svg) and made by user Alisneaky.

**Definition:**      
> A **Support Vector Machine** (**SVM**) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.
> -- [Savan Patel](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72)

**Problem solved using the algorithm:** Classify tumors as benign or malign, based on about 30 criteria such as size, growth and many more.

**Accuracy:** ~92% - ~96%

<a id="k-means"></a>
## K-Means Clustering

![Convergence of k-means clustering from an unfavorable starting position (two initial cluster centers are fairly close).](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/617px-K-means_convergence.gif)  
  Image taken from [wikimedia](https://commons.wikimedia.org/wiki/File:K-means_convergence.gif) and made by user [Chire](https://commons.wikimedia.org/wiki/User:Chire).

**Definition:**      
> The **k**-**means clustering algorithm** attempts to split a given anonymous data set (a set containing no information as to class identity) into a fixed number (**k**) of **clusters**. Initially **k** number of so called centroids are chosen. A centroid is a data point (imaginary or real) at the center of a **cluster**.
>  -- [Ola Söder](http://www.fon.hum.uva.nl/praat/manual/k-means_clustering_1__How_does_k-means_clustering_work_.html)


**Problem solved using the algorithm:** Classifing handwritten digits.

**Accuracy:** Measured on 69510 data points
- homogeneity: ~0.61 	
- completeness: ~0.66
- v-measure: ~0.63 	
- adjusted-rand: ~0.48 	
- adjusted-mutual-info: ~0.61 	
- silhouette: ~0.14

### Explanation:
 (can also be found as a comment in the k_means_cluster.py)
 - **homogeneity:** each cluster contains only members of a single class (range 0 - 1)  
- **completeness:** all members of a given class are assigned to the same cluster (range 0 - 1)  
- **v-measure:** harmonic mean of homogeneity and completeness  
- **adjusted_rand:** similarity of the actual values and their predictions,                    ignoring permutations and with chance normalization (range -1 to 1, -1 being bad, 1 being perfect and 0 being random)
- **adjusted_mutual_info:** agreement of the actual values and predictions, ignoring permutations (range 0 - 1, with 0 being random agreement and 1 being perfect agreement)  
- **silhouette:** uses the mean distance between a sample and all other points in the same class, as well as the mean distance between a sample and all other points in the nearest cluster to calculate a score (range: -1 to 1, with the former being incorrect, and the latter standing for highly dense clustering. 0 indicates overlapping clusters.


<a id="dec-trees"></a>
## Decision Trees
![A sample tree](https://i.gyazo.com/c8254a7a1ca3603ef61f6a1440588d0c.png)  
Image generated using decision_tree.py.

**Definition:**  
> In computer science, **Decision tree learning** uses a decision tree (as a predictive model) to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves).
> -- [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

**Problem solved using the algorithm:**  Predicting the onset of diabetes based on diagnostic measures.


**Accuracy:**
- **~78%** with a depth of *4* and *information gain* as attribute selection measure
- **~76%** with a depth of *3* and *information gain* as attribute selection measure
- **~76%** with a depth of *4* and *gini impurity* as attribute selection measure
