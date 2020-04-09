import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


x = np.array([[1,1],[1.5,1],[3,3],[4,4],[3,3.5],[3.5,4]])

plt.scatter(x[:,0],x[:,1],s=50)
plt.show()


# 'single' -> nearest point algorithm d(u,v) = min(dist(point_i, point_j))

linkage_matrix = linkage(x, "single")


print(linkage_matrix)


# we can prune the tree like structure with truncate_mode
# nonce: do not prune, do not truncate !!!

dendrogram = dendrogram(linkage_matrix, truncate_mode='none')

plt.title("Hierarchical clustering")

plt.show()
