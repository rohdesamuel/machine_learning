import numpy as np

class KNearestNeighbor:
	def __init__(self, description):
		self.description = description

	def _distance(self, x1, x2):
		return np.sum(np.square(x1 - x2))

	def fit(self, features, labels, validate):
		self.x = np.array(features)
		self.y = np.array(labels)

		x = np.array(validate)
		self.distances = []
		for x1 in x:
			distances = [(self._distance(x1, x2), i) for i, x2 in enumerate(self.x)]
			distances.sort(key=lambda _: _[0])
			self.distances.append(distances)

	def predict(self, features, k, threshold=0.5):
		x = np.array(features)
		m, n = x.shape

		predictions = []
		for distances in self.distances:
			closest = distances[:k]
			labels = [self.y[i] for _, i in closest]
			num_ones = np.sum(np.equal(labels, 1))
			num_zeros = np.sum(np.equal(labels, 0))
			predictions.append(int((num_ones / (num_ones + num_zeros)) >= threshold))
		return predictions