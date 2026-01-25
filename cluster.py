import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def perform_clustering(data, optimal_k=5):
    # Filter for numeric columns
    numeric_data = data.select_dtypes(include=[float, int])
    
    # Check for missing values and handle them
    if numeric_data.isnull().values.any():
        numeric_data.fillna(numeric_data.mean(), inplace=True)

    # Scale the numeric data
    scaler = StandardScaler()
    cdata_scaled = scaler.fit_transform(numeric_data)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(cdata_scaled)
    
    # Get cluster centers back in original space
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return clusters, cluster_centers
