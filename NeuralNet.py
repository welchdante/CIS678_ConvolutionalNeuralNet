import math
import numpy as np
from pprint import pprint
from random import random
from numpy import genfromtxt

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

	def forward_propagate(self, row):
		inputs = row
		for layer in self.network:
			new_inputs = []
			for neuron in layer:
				activation = self.activate(neuron['weights'], inputs)
				neuron['output'] = self.sigmoid(activation)
				new_inputs.append(neuron['output'])
			inputs = new_inputs
		return inputs

	def back_propagate(self, expected):
		for i in reversed(range(len(self.network))):
			layer = self.network[i]
			errors = list()
			if i != len(self.network) - 1:
				for j in range(len(layer)):
					error = 0.0
					for neuron in self.network[i + 1]:
						error += neuron['weights'][j] * neuron['delta']
					errors.append(error)
			else: 
				for j in range(len(layer)):
					neuron = layer[j]
					errors.append(expected[j] - neuron['output'])
			for j in range(len(layer)):
				neuron = layer[j]
				neuron['delta'] = errors[j] * self.sigmoid_derivative(neuron['output'])					
		#calculate error for the output of y
		
		#calculate the error for each hidden unit h_i



neural_net = NeuralNet(1,2,1)
row = [1, 0]
output = neural_net.forward_propagate(row)
pprint(output)
neural_net.network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
expected = [0, 1]
neural_net.back_propagate(expected)
for layer in neural_net.network:
	print(layer)

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