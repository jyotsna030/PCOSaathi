#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.mixture import GaussianMixture


# In[2]:


# Load Dataset
file_path = r"C:/Users/jyots/OneDrive/Desktop/Women-Health/CLEAN- PCOS SURVEY SPREADSHEET.csv"
df = pd.read_csv(file_path)

# Display basic information
print(df.info())
print(df.head())


# In[3]:


# Rename columns for better readability
df.columns = [
    "Age", "Weight", "Height", "Blood_Group", "Period_Cycle", "Weight_Gain", 
    "Facial_Hair", "Skin_Darkening", "Hair_Loss", "Acne", "Fast_Food", "Exercise", 
    "PCOS_Diagnosed", "Mood_Swings", "Regular_Periods", "Period_Duration"
]

# Check for missing values
missing_values = df.isnull().sum()

# Normalize numerical features (excluding categorical ones like Blood_Group, PCOS_Diagnosed)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_cols = ["Age", "Weight", "Height", "Period_Cycle", "Period_Duration"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Display cleaned dataset info
df.info(), missing_values



# In[4]:


# Selecting relevant features for clustering
features = ["Age", "Weight", "Height", "Period_Cycle", "Weight_Gain", "Facial_Hair", 
            "Skin_Darkening", "Hair_Loss", "Acne", "Fast_Food", "Exercise", 
            "Mood_Swings", "Regular_Periods", "Period_Duration"]

X = df[features]

# Checking feature correlation to validate relevance
correlation_matrix = X.corr()

# Display correlation matrix
correlation_matrix



# In[5]:


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Compute linkage matrix using Ward's method
linkage_matrix = sch.linkage(X, method='ward')

# Plot dendrogram to determine optimal clusters
plt.figure(figsize=(12, 6))
sch.dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# In[6]:


from sklearn.cluster import AgglomerativeClustering

# Apply Hierarchical Clustering with corrected parameters
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
df['HC_Cluster'] = hc.fit_predict(X)

# Display cluster distribution
df['HC_Cluster'].value_counts()



# In[7]:


from sklearn.mixture import GaussianMixture

# Apply GMM with k=4
gmm = GaussianMixture(n_components=4, random_state=42)
df["GMM_Cluster"] = gmm.fit_predict(X)

# Display cluster distribution from GMM
df["GMM_Cluster"].value_counts()


# In[8]:


from sklearn.decomposition import PCA
import seaborn as sns

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Add PCA components to dataframe
df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]

# Plot GMM Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df["PCA1"], y=df["PCA2"], hue=df["GMM_Cluster"], palette="viridis", alpha=0.7)
plt.title("GMM Clusters Visualization (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()


# In[9]:


from sklearn.metrics import silhouette_score

# Compute Silhouette Score for GMM clusters
silhouette_gmm = silhouette_score(X, df["GMM_Cluster"])

# Compute Silhouette Score for Hierarchical Clustering
silhouette_hc = silhouette_score(X, df["HC_Cluster"])

silhouette_gmm, silhouette_hc


# In[10]:


# Compute mean feature values for each Hierarchical Clustering group
hc_cluster_summary = df.groupby("HC_Cluster")[features].mean()

# Compute mean feature values for each GMM cluster
gmm_cluster_summary = df.groupby("GMM_Cluster")[features].mean()

hc_cluster_summary, gmm_cluster_summary

