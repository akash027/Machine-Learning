import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs


x, y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=2)

plt.scatter(x[:,0], x[:,1], s=75)
plt.show()


est = KMeans(5)
est.fit(x)

y_kmeans = est.predict(x)


plt.scatter(x[:,0],x[:,1], c=y_kmeans, s=75, cmap='rainbow')
plt.show()