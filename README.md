# Gender_and_age_detection_system
This code performs K-means clustering on a dataset of customer information to group customers based on their annual income and spending score. Here's a breakdown of what the code does:

1. Imports necessary libraries: `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, and `KMeans` from `sklearn.cluster`.

2. Loads the customer data from a CSV file called 'Mall_Customers.csv' into a Pandas DataFrame named `customer_data`.

3. Checks the shape of the DataFrame to find the number of rows and columns in the dataset using `customer_data.shape`.

4. Retrieves information about the DataFrame, such as column data types and non-null counts, using `customer_data.info()`.

5. Checks for missing values in the dataset using `customer_data.isnull().sum()`.

6. Selects the relevant features for clustering, which are 'Annual Income' and 'Spending Score', and stores them in a new variable `X`.

7. Calculates the Within-Cluster Sum of Squares (WCSS) value for different numbers of clusters (ranging from 1 to 10) and stores the results in a list `wcss`. WCSS represents the sum of squared distances between data points and their assigned cluster centroids.

8. Plots an elbow graph to visualize the WCSS values for different numbers of clusters. The goal is to find the optimal number of clusters where the graph's elbow point occurs, suggesting a suitable number of clusters for the data.

9. From the elbow graph analysis, it is determined that 5 clusters is an appropriate choice. A K-means model is then created with `n_clusters=5`.

10. The data points are assigned to their respective clusters based on their cluster label (0 to 4) using `fit_predict` method.

11. The cluster labels for each data point are printed to the console using `print(Y)`.

12. The clusters and their centroids are plotted using `matplotlib.pyplot.scatter`. Each cluster is shown in a different color, and the cluster centroids are marked with cyan color.

13. Appropriate labels and titles are added to the plot for clarity.

Overall, this code performs K-means clustering on the customer dataset, determines the optimal number of clusters using an elbow graph, and visualizes the resulting clusters and centroids in a 2D plot.
