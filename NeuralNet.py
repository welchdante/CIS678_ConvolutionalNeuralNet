import math
from pprint import pprint
from random import random
from random import seed
from csv import reader
import csv

class NeuralNet:

	def __init__(self, n_hidden_layers, n_inputs, n_outputs):
		self.n_hidden_layers = n_hidden_layers
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.network = list()
		self.hidden_layer = [{'weights':[random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden_layers)]
		self.network.append(self.hidden_layer)
		self.output_layer = [{'weights':[random() for i in range(self.n_hidden_layers + 1)]} for i in range(self.n_outputs)]
		self.network.append(self.output_layer)

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

	def back_propagate_error(self, expected):
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
					if i == 0 and j == 0:
						print("\t\t\t", inputs[j])
						print("\t\t\t", neuron['delta'])
						print("\t\t\t", learning_rate * neuron['delta']*inputs[j])
				neuron['weights'][-1] += learning_rate * neuron['delta']


	def train(self, training_data, learning_rate, num_epochs, num_outputs):
		num_correct = 0
		total_guesses = 0
		for epoch in range(num_epochs):			
			error = 0
			random_row = int(random()*len(training_data))
			#print(training_data[random_row])
			row = training_data[random_row]
			outputs = self.forward_propagate(row)
			expected = [0 for i in range(num_outputs)]
			expected[row[-1]] = 1
			if outputs.index(max(outputs)) == expected.index(max(expected)):
				num_correct += 1
			total_guesses += 1
			accuracy = num_correct / total_guesses
			#print('Epoch: %d, Prediction: %d, Actual: %d, Accuracy: %f' %(epoch, outputs.index(max(outputs)), expected.index(max(expected)), accuracy))				
			error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			self.backpropagate(row, expected, outputs)
			#self.back_propagate_error(expected)
			self.update_weights(row, learning_rate)

			# for row in training_data:
			# 	outputs = self.forward_propagate(row)
			# 	expected = [0 for i in range(num_outputs)]
			# 	expected[row[-1]] = 1
			# 	if outputs.index(max(outputs)) == expected.index(max(expected)):
			# 		num_correct += 1
			# 	total_guesses += 1
			# 	accuracy = num_correct / total_guesses
			# 	#print('Epoch: %d, Prediction: %d, Actual: %d, Accuracy: %f' %(epoch, outputs.index(max(outputs)), expected.index(max(expected)), accuracy))				
			# 	error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			# 	self.back_propagate_error(expected)
			# 	self.update_weights(row, learning_rate)
			#print(outputs)
			#print('Prediction: %d, Actual: %d' %(outputs.index(max(outputs)), expected.index(max(expected))))					
			#print('Epoch = %d, Learning Rate = %.3f, Error = %.3f' % (epoch, learning_rate, error))
			#if epoch % 100 == 0:
			print('Epoch = %d, Learning Rate = %.3f, Error = %.3f' % (epoch, learning_rate, error))
		
def read_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def write_csv(dataset):
	with open("this_better_work.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerows(dataset)

def convert_data(dataset):
	for i in range(len(dataset[0]) - 1):
		convert_data_to_float(dataset, i)
	convert_class_data_to_int(dataset, -1)

def convert_data_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def convert_class_data_to_int(dataset, column):
	for row in dataset:
		row[column] = int(row[column])

def normalize_data(dataset):
	print(len(dataset[0]))
	for i in range(len(dataset)):
		label = dataset[i][-1]
		for j in range(len(dataset[i])):
			dataset[i][j] = dataset[i][j] / 255
			dataset[i][-1] = label
	return dataset

def predict(neural_net, row):
	outputs = neural_net.forward_propagate(row)
	print(outputs)
	return outputs.index(max(outputs))

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores

seed(1)
filename = 'normalized_train.csv'
dataset = read_csv(filename)
convert_data(dataset)

# dataset = [
# 	[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1],
# 	[12.627531214,2.759262235,2],
# 	[11.332441248,2.088626775,2],
# 	[13.922596716,1.77106367,2],
# 	[14.675418651,-0.242068655,2]
# 	]

#normalized_data = normalize_data(dataset)
#print(normalized_data)

#test_filename = 'sample_test_data.csv'
#test_dataset = read_csv(filename)
#convert_data(test_dataset)

#write_csv(dataset)

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
n_hidden_layers = 1
learning_rate = .1
n_epochs = 10000
neural_net = NeuralNet(n_hidden_layers, n_inputs, n_outputs)
neural_net.train(dataset, learning_rate, n_epochs, n_outputs)

#predictions = list()
#for row in test_dataset:
#	prediction = predict(neural_net, row)
#	predictions.append(prediction)

#print(predictions)

#scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)


#predictions = list()
#for row in test:
#	prediction = predict(network, row)
#	predictions.append(prediction)
#return(predictions)


#for layer in neural_net.network:
	#print(layer['weights'])
#	print()
#	print()

#for row in dataset:
#	prediction = neural_net.predict(row)
#	print('Expected=%d, Got=%d' % (row[-1], prediction))

