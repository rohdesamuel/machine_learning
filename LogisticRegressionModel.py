import math
import numpy as np
import EvaluationsStub

import apache_beam as beam
class BeamLogisticRegression(object):
	def __init__(self):
		pass

	class LogisticRegression(beam.DoFn):
		WEIGHTS = beam.BagStateSpec('weights', beam.coders.FastPrimitivesCoder())

		def process(self, element, weight_state=beam.DoFn.StateParam(WEIGHTS)):
			weight_state.read()

	def fit(self, features, labels, epochs, step=0.01):

		with beam.Pipeline() as p:
			weights = beam.Create(np.array([ [0.0] for _ in range(n)]))
			(p
	         | beam.Create(np.array(features))
			 | beam.Map(printer))

		


class LogisticRegressionModel(object):
	"""A model that predicts the most common label from the training data."""
	def __init__(self, description, log_every_n=0):
		self._description = description
		self._log_every_n = log_every_n

	def _sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def fit(self, features, labels, epochs, step=0.01):
		assert(len(features) == len(labels))
		
		x = np.array(features)
		y = np.array([ [y] for y in labels])

		m, n = x.shape

		self.bias = 0.0
		self.weights = np.array([ [0.0] for _ in range(n)])
		
		for epoch in range(epochs):
			if epoch % 100 == 0:
				print('>> \'{}\' {}% complete'.format(self._description, 100 * epoch / epochs), end='\r', flush=True)

			y_hat = np.dot(x, self.weights) + self.bias
			pred = np.apply_along_axis(self._sigmoid, 0, y_hat)

			losses = np.subtract(pred, y)
			
			bias_delta = np.mean(losses) * step
			feature_losses = np.multiply(losses, x)
			feature_mean = np.mean(feature_losses, 0) * step
			weight_deltas = np.reshape(feature_mean, (n, 1))

			self.weights = np.subtract(self.weights, weight_deltas)
			self.bias -= bias_delta

		print('>> \'{}\' 100% complete'.format(self._description), end='\r', flush=True)
		print()

	def loss(self, features, labels):
		assert(len(features) == len(labels))

		n = len(features)
		loss = 0.0
		for i in range(n):
			x = features[i]
			y = labels[i]
			prediction = self._predict(x)
			loss += (-y * math.log(prediction) - 
					 (1 - y) * math.log(1 - prediction))
		return loss / n

	def _predict(self, features):
		x = np.array(features)
		y_hat = np.dot(x, self.weights) + self.bias
		return np.apply_along_axis(self._sigmoid, 0, y_hat)

	def predict(self, features, threshold=0.5):
		return np.greater(self._predict(features), threshold).flatten().astype(int)

	def context(self, features):
		return self._predict(features)