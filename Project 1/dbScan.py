from typing import Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(train_data: bool=True) -> Tuple[np.ndarray, np.ndarray]:
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

### Cluster images based on feature vectors
dbscan = DBSCAN(eps=0.5, min_samples=3)
cluster_labels = dbscan.fit_predict(train_features)

### Compute the Calinski-Harabasz Index before dividing the largest cluster
calinski_before = calinski_harabasz_score(train_features, cluster_labels)
print(f"Calinski-Harabasz Index before dividing largest cluster: {calinski_before}")

### Project feature vectors onto 2D using TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=123)
tsne_results = tsne.fit_transform(train_features)

### Create DataFrame for visualization
df = pd.DataFrame()
df["label"] = image_labels
df["cluster"] = cluster_labels
df["tsne-2d-one"] = tsne_results[:, 0]
df["tsne-2d-two"] = tsne_results[:, 1]


### Plot the scatter plot
sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue="cluster", data=df)
plt.title("Clustering using t-SNE")
plt.show()