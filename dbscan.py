from math import sqrt
from random import choice

from numpy import zeros


class DBSCAN:

	def __init__(self, eps, min_samples):
		self.eps = eps
		self.min_samples = min_samples

	@staticmethod
	def euclid(a, b):
		return sqrt(((a - b) ** 2).sum())

	def get_neighbors(self, index, data, eps):
		neighbors = []
		for j in range(len(data)):
			if j != index and self.euclid(data[j], data[index]) <= eps:
				neighbors.append(j)
		return neighbors

	def fit(self, data):
		assigned_clusters = list(zeros(len(data)))
		cluster_number = 0  # 0: no cluster, -1: noise, 1,2,... are clusters
		while 0 in assigned_clusters:
			ind = choice([k for k in range(len(assigned_clusters)) if assigned_clusters[k] == 0])
			neighbors = self.get_neighbors(ind, data, self.eps)
			if len(neighbors) >= self.min_samples:
				cluster_number += 1
				assigned_clusters[ind] = cluster_number
				for i in neighbors:
					if assigned_clusters[i] == -1:
						assigned_clusters[i] = cluster_number
					if assigned_clusters[i] == 0:  # repetition and border
						assigned_clusters[i] = cluster_number
						neighbors2 = self.get_neighbors(i, data, self.eps)
						if len(neighbors2) >= self.min_samples:
							neighbors += neighbors2
			else:
				assigned_clusters[ind] = -1
		return assigned_clusters



from sklearn.datasets import load_wine, make_moons
from sklearn.decomposition import PCA
from numpy import array
from matplotlib.pyplot import scatter, show

model = DBSCAN(eps=0.1, min_samples=4)
# data = PCA(n_components=2).fit_transform(load_wine().data)
data, labels = make_moons(n_samples=1000, noise=0.1)
clusters = model.fit(data)

colors = array(['red', 'blue', 'green', 'black', 'pink', 'yellow', 'cyan'])
scatter(data[:,0], data[:,1], c=colors[clusters])
show()