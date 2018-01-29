import math
from pprint import pprint
from random import random
from random import seed
from csv import reader

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

	def update_weights(self, row, learning_rate):
		for i in range(len(self.network)):
			inputs = row[:-1]
			if i != 0:
				inputs = [neuron['output'] for neuron in self.network[i - 1]]
			for neuron in self.network[i]:
				for j in range(len(inputs)):
					neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
				neuron['weights'][-1] += learning_rate * neuron['delta']

	def train(self, training_data, learning_rate, num_epochs, num_outputs):
		for epoch in range(num_epochs):
			error = 0
			for row in training_data:
				outputs = self.forward_propagate(row)
				expected = [0 for i in range(num_outputs)]
				print(expected[row[-1]])
				expected[row[-1]] = 1
				error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
				self.back_propagate(expected)
				self.update_weights(row, learning_rate)
			print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate, error))

	def predict(self, row):
		output = self.forward_propagate(row)
		return output.index(max(output))


def read_csv(filename):
		dataset = list()
		with open(filename, 'r') as file:
			csv_reader = reader(file)
			for row in csv_reader:
				if not row:
					continue
				dataset.append(row)
		return dataset

def convert_data_to_int(dataset):
	converted_dataset = [[int(value) for value in sublist] for sublist in dataset]
	return converted_dataset

def normalize_data(dataset):
	for row in dataset:
		minimum = min(row)
		maximum = max(row)
		for data in range(len(row)):
			minmax = (row[data] - minimum) / (maximum - minimum)
			row[data] = minmax

filename = 'sample_train.csv'
dataset = read_csv(filename)
dataset = convert_data_to_int(dataset)
normalize_data(dataset)

n_inputs = len(dataset[0]) - 1
n_outputs = 10
neural_net = NeuralNet(1, n_inputs, n_outputs)
neural_net.train(dataset, 0.1, 50, n_outputs)


#for row in dataset:
#	prediction = neural_net.predict(row)
#	print('Expected=%d, Got=%d' % (row[-1], prediction))

'''
seed(1)
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))

neural_net = NeuralNet(1, n_inputs, n_outputs)
neural_net.train(dataset, 0.5, 50, n_outputs)
#for layer in neural_net.network:
#	print(layer)
for row in dataset:
	prediction = neural_net.predict(row)
	print('Expected=%d, Got=%d' % (row[-1], prediction))
'''
