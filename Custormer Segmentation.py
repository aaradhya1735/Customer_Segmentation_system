import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('Mall_Customers.csv')

# Finding the number of rows and columns in the dataset
customer_data.shape

# Getting some information about the dataset, such as column data types and non-null counts
customer_data.info()

# Checking for missing values in the dataset
customer_data.isnull().sum()

# Selecting the relevant features (Annual Income and Spending Score) for clustering
X = customer_data.iloc[:, [3, 4]].values

# Finding the Within-Cluster Sum of Squares (WCSS) value for different number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting an elbow graph to find the optimal number of clusters
sns.set()
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# After analyzing the elbow graph, it seems that 5 clusters is an appropriate choice
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# Assigning each data point to a cluster based on their cluster label (0 to 4)
Y = kmeans.fit_predict(X)

# Printing the cluster labels for each data point
print(Y)

# Plotting all the clusters and their centroids
plt.figure(figsize=(8, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')

# Plotting the centroids of each cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
