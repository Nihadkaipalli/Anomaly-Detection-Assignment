# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Load the football dataset
data = pd.read_csv('football.csv')

##########################
# Exploratory Data Analysis
##########################

# Summary statistics
print(data.describe())

# Correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for visualization of feature relationships
sns.pairplot(data[['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF']])
plt.suptitle('Pairplot of Features', y=1.02)
plt.show()

#############################
# Clustering-based Anomaly Detection
#############################

# Define features for anomaly detection
anomaly_features = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF']
X_anomaly = data[anomaly_features]

# Scale the features using RobustScaler
scaler = RobustScaler()
X_anomaly_scaled = scaler.fit_transform(X_anomaly)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_anomaly_pca = pca.fit_transform(X_anomaly_scaled)

# Fit an Elliptic Envelope for outlier detection
envelope = EllipticEnvelope(contamination=0.05)
envelope.fit(X_anomaly_scaled)
anomaly_pred = envelope.predict(X_anomaly_scaled)

# Plot anomaly detection results
plt.figure(figsize=(10, 6))
plt.scatter(X_anomaly_pca[:, 0], X_anomaly_pca[:, 1], c=(anomaly_pred == -1), cmap='coolwarm', alpha=0.7)
plt.title('Clustering-based Anomaly Detection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Outlier')
plt.show()

###################
# K-means Clustering
###################

# Define features for K-means clustering
kmeans_features = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF']
X_kmeans = data[kmeans_features]

# Scale the features using RobustScaler
scaler_kmeans = RobustScaler()
X_kmeans_scaled = scaler_kmeans.fit_transform(X_kmeans)

# Determine optimal K using Elbow method
visualizer = KElbowVisualizer(KMeans(), k=(2, 10), timings=False)
visualizer.fit(X_kmeans_scaled)
visualizer.show()

# Perform K-means clustering with the optimal K
optimal_k = visualizer.elbow_value_
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_kmeans_scaled)

# Apply PCA for visualization
X_kmeans_pca = pca.transform(X_kmeans_scaled)

# Plot K-means clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X_kmeans_pca[:, 0], X_kmeans_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()
