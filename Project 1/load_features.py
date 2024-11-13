import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import silhouette_score,calinski_harabasz_score

def load_data(train_data: bool =True) -> Tuple[np.ndarray, np.ndarray]:
    """
    loads train/test features with image labels.  
    """
    if train_data:
        data = np.load(f'train_data.npz')
    else:
        data = np.load(f'test_data.npz')

    features = data['features']
    img_labels = data['img_labels']

    return features, img_labels

def load_data_with_domain_label() -> Tuple[np.ndarray, np.ndarray]:
    """
    loads portion of training features with domain label
    """
    data = np.load(f'train_data_w_label.npz')
    train_features = data['features']
    domain_labels = data['domain_labels']

    return train_features, domain_labels

### Load data
train_features, image_labels = load_data(True)
train_5percent,image_labelst = load_data_with_domain_label()
### Cluster images based on feature vectors
kmeans = KMeans(n_clusters=3)
cluster_labels = kmeans.fit_predict(train_features)
### Compute the Calinski-Harabasz Index before dividing the largest cluster
calinski_before = calinski_harabasz_score(train_features, cluster_labels)
print(f"Calinski-Harabasz Index before dividing largest cluster: {calinski_before}")

figure,axis=plt.subplots(ncols=4)
### Project feature vectors onto 2D using PCA
pca = PCA(n_components=2)
pca_results2 = pca.fit_transform(train_features)
df2 = pd.DataFrame()
df2["label"] = image_labels
df2["cluster"] = cluster_labels
df2["pca-2d-one"] = pca_results2[:, 0]
df2["pca-2d-two"] = pca_results2[:, 1]
sns.scatterplot(x="pca-2d-one", y="pca-2d-two", hue="cluster", data=df2,ax=axis[3]).set(title="20k data with kmeans 3")


largest_cluster = np.argmax(np.bincount(cluster_labels))
kmeans_sub = KMeans(n_clusters=3)
largest_cluster_features = train_features[cluster_labels == largest_cluster]
largest_cluster_sub_labels = kmeans_sub.fit_predict(largest_cluster_features)
cluster_labels[cluster_labels == largest_cluster] = largest_cluster_sub_labels + 3
new_cluster_labels = np.concatenate((cluster_labels, image_labels + 6))

##----------pca--------------
pca_results = pca.fit_transform(train_5percent)

### Create DataFrame for visualization
df = pd.DataFrame()
df1 = pd.DataFrame()
df["label"] = image_labelst
#df["cluster"] = train_5percent
df["pca-2d-one"] = pca_results[:, 0]
df["pca-2d-two"] = pca_results[:, 1]

df1["label"] = image_labels
df1["cluster"] = cluster_labels
df1["pca-2d-one"] = pca_results2[:, 0]
df1["pca-2d-two"] = pca_results2[:, 1]
### Plot the scatter plot
sns.scatterplot(x="pca-2d-one", y="pca-2d-two", hue="label", data=df,ax=axis[0], palette="deep").set(title="true labels of 5k train data")
sns.scatterplot(x="pca-2d-one", y="pca-2d-two", hue="cluster", data=df1,ax=axis[1]).set(title="20k data with kmeans")


### Compute the Calinski-Harabasz Index after dividing the largest cluster
calinski_after = calinski_harabasz_score(train_features, cluster_labels)
print(f"Calinski-Harabasz Index after dividing largest cluster: {calinski_after}")

plt.show()
