from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
iris = datasets.load_iris()

x = iris.data
y = iris.target
n=10
#the label given by the clusters to the dataset
clusterLabels=[]
# print(x.shape)
# print(y.shape)
# print(np.unique(y))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters =n)

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=42,test_size=0.2)
kmeans.fit(x_train)
x_pred = kmeans.predict(x_train)
# print(x_pred.shape)

for i in range(n):
    #labels contain all true labels in cluster i 
    labels = y_train[x_pred == i]
    #true_label contain the most common label in that cluster
    true_label = np.bincount(labels).argmax()
    #choose the most common true label as the main label of cluster i by saving it in a list
    clusterLabels.append(true_label)
    print("the label of the", i, "th cluster is ", clusterLabels[i])

#execute kmeans on test
kmeans.fit(x_test)
x_pred2 = kmeans.predict(x_test)


# evaluate the clustering based on how well the true labels 
# match the predicted cluster labels assigned by KMeans.
for i in range(n):
    #find the most common label in the cluster
    Tlabels = y_test[x_pred2 == i]
    true_Tlabel = np.bincount(Tlabels).argmax()
    print("Cluster", i, "most common label:", true_Tlabel)
    #check if the true label match
    if true_Tlabel == clusterLabels[i]:
        print("Cluster", i, "correctly labeled")
    else:
        print("Cluster", i, "mis-labeled")
