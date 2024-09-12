# Dependencies
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# Class for implementing the K-Means algorithm
class kMeans:

    def __init__(self, k=5, max_iters=100, plot_steps=False):
        self.k = k  # Define the number of clusters (default is 5)
        self.max_iters = max_iters  # Maximum number of iterations for convergence
        self.plot_steps = plot_steps  # Flag to control whether steps are visualized

        # Initialize list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]  # Empty clusters for k clusters

        # The centers (mean vector) for each cluster
        self.centroids = []  # This will hold the centroid (mean) values of each cluster

    def predict(self, x):
        self.x = x  # Input data (features)
        self.n_samples, self.n_features = x.shape  # Number of samples and features

        # Initialize centroids
        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.x[idx] for idx in random_sample_indices]

        # Optimization loop to update clusters and centroids
        for _ in range(self.max_iters):
            # Assign samples to closest centroids to form clusters
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()  # Plot the steps if visualization is enabled

            # Calculate new centroids from the clusters
            centroids_old = self.centroids  # Store the previous centroids
            self.centroids = self._get_centroids(self.clusters)  # Recompute centroids

            # Check if the algorithm has converged (centroids did not change)
            if self._is_converged(centroids_old, self.centroids):
                break  # Exit the loop if convergence is reached

            if self.plot_steps:
                self.plot()  # Plot steps again after centroids are updated

        # Assign each sample the label of the cluster it belongs to
        return self._get_cluster_labels(self.clusters)

    # Helper function to return cluster labels based on the final clusters
    def _get_cluster_labels(self, clusters):
        # Create an array to store the label (cluster index) for each sample
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx   # Assign the cluster index to the sample

        return labels

    # Helper function to assign samples to the closest centroids
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]  # Initialize empty clusters
        for idx, sample in enumerate(self.x):
            # Find the closest centroid for each sample
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)  # Assign sample index to the closest cluster
        return clusters

    # Helper function to find the index of the closest centroid for a given sample
    def _closest_centroid(self, sample, centroids):
        # Calculate the distance of the sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)  # Find the index of the closest centroid
        return closest_idx

    # Helper function to assign mean value of clusters to centroids
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))  # Initialize centroids array
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.x[cluster], axis=0)  # Calculate mean of the cluster
            centroids[cluster_idx] = cluster_mean  # Update the centroid with the mean
        return centroids

    # Helper function to check if the centroids have converged (no change)
    def _is_converged(self, centroids_old, centroids):
        # Calculate the distances between old and new centroids for all clusters
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.k)]
        return sum(distances) == 0  # Converged if all distances are zero

    # Helper function to plot the clusters and centroids (only for 2D data)
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))

        # Plot all points in each cluster
        for i, index in enumerate(self.clusters):
            point = self.x[index].T  # Transpose for plotting
            ax.scatter(*point)  # Scatter plot the points in the cluster

        # Plot the centroids
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)  # Mark centroids with 'x'

        # Display the plot
        plt.show()
