import math
import numpy as np
from numpy import genfromtxt

class NeuralNet:

	def __init__(self, test):
		self.test = test

	def read_test_data(self):
		self.test_data = genfromtxt('test_data.csv', delimiter=',')

	def read_test_labels(self):
		self.test_labels = genfromtxt('test_labels.csv', delimiter=',')

	def read_training_data(self):
		self.training_data = genfromtxt('train.csv', delimiter=',')

	#def print_values	

neural_net = NeuralNet(5)
neural_net.read_test_data()
neural_net.read_test_labels()
neural_net.read_training_data()

#my_data = genfromtxt('test_data.csv', delimiter=',')	


#count=0
#for row in my_data:
#	print(row)
#	print("------------------------------------------------------------")
#	print("------------------------------------------------------------")
#	count+=1
#	print(count)