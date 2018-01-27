import math
import numpy as np
from pprint import pprint
from random import random
from numpy import genfromtxt

default_settings = {
	"lower_bound_weights" : 0.1,
	"upper_bound_weights" : 0.1,
	"init_bias" : 0.01
}

class NeuralNet:

	def __init__(self, n_hidden_layers, n_inputs, n_outputs):
		self.n_hidden_layers = n_hidden_layers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.network = list()
		hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden_layers)]
		self.network.append(hidden_layer)
		output_layer = [{'weights':[random() for i in range(self.n_hidden_layers + 1)]} for i in range(self.n_outputs)]
		self.network.append(output_layer)


	def read_test_data(self):
		self.test_data = genfromtxt('test_data.csv', delimiter=',')

	def read_test_labels(self):
		self.test_labels = genfromtxt('test_labels.csv', delimiter=',')

	def read_training_data(self):
		self.training_data = genfromtxt('train.csv', delimiter=',')

	def read_sample_set(self):
		self.sample_data = genfromtxt('sample_set.csv', delimiter=',')

	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def activate(self, weights, inputs):
		activation = weights[-1]
		for i in range(len(weights)-1):
			activation += weights[i] * inputs[i]
		return activation

	def forward_propagate(self, network, row):
		inputs = row
		for layer in network:
			new_inputs = []
			for neuron in layer:
				activation = self.activate(neuron['weights'], inputs)
				neuron['output'] = self.sigmoid(activation)
				new_inputs.append(neuron['output'])
			inputs = new_inputs
		return inputs


neural_net = NeuralNet(1,2,2)
pprint(neural_net.network)
print()
row = [1, 0, None]
output = neural_net.forward_propagate(neural_net.network, row)
print(output)

#neural_net.network = neural_net.initialize_network()
#for layer in neural_net.network:
#	print(layer)
#pprint(neural_net.initialize_network())

#my_data = genfromtxt('test_data.csv', delimiter=',')	


#count=0
#for row in neural_net.sample_data:
#	print(row)
#	print("------------------------------------------------------------")
#	print("------------------------------------------------------------")
#	count+=1
#	print(count)