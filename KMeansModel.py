import numpy as np
import random

class KMeans:
  def __init__(self, seed=None):
    random.seed(seed)

  def _distance(self, x1, x2):
    return np.sum(np.square(x1 - x2))

  def fit(self, points, clusters, iterations, verbose=False):
    x = np.array(points)
    m, n = x.shape

    centers = [x[random.randint(0, m-1)] for _ in range(clusters)]
    for iter in range(iterations):
      assignments = [[] for _ in range(clusters)]
      for i in range(m):
        x1 = x[i]
        distances = [self._distance(x1, x2) for x2 in centers]
        assignments[distances.index(min(distances))].append(x1)
      assignments = np.array(assignments)

      for i in range(clusters):
        assigned = assignments[i]
        centers[i] = np.sum(assigned, 0) / len(assigned)
      if verbose:
        print(','.join([str(v) for v in np.array(centers).flatten()]))

    closest = [[] for _ in range(clusters)]
    indices = [[] for _ in range(clusters)]
    for i in range(m):
      x1 = x[i]
      distances = [self._distance(x1, x2) for x2 in centers]
      closest[distances.index(min(distances))].append(x1)
      indices[distances.index(min(distances))].append(i)

    for cluster in range(clusters):
      distances = [self._distance(x1, centers[cluster]) for x1 in closest[cluster]]
      index = distances.index(min(distances))
      closest[cluster] = indices[cluster][index]
    self._closest = closest
    self._means = centers
   
  def predict(self, x):
    pass

  def means(self):
    return self._means

  def closest(self):
    return self._closest