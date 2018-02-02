from csv import reader
import csv
import numpy as np

def read_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

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

filename = 'train.csv'
dataset = read_csv(filename)
convert_data(dataset)
#print(type(dataset))

data = np.array([np.array(xi) for xi in dataset])
pprint(data)