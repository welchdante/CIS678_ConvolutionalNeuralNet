import math
import random
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all = 'ignore')

class NeuralNetwork(object):
    def __init__(self, input, hidden, output, num_epochs, learning_rate, momentum, rate_decay):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rate_decay = rate_decay
        
        self.input = input + 1
        self.hidden = hidden
        self.output = output

        self.input_list = [1.0] * self.input
        self.hidden_list = [1.0] * self.hidden
        self.output_list = [1.0] * self.output

        input_range = 1.0 / self.input ** (1/2)
        output_range = 1.0 / self.hidden ** (1/2)
        self.input_weights = np.random.normal(loc = 0, scale = input_range, size = (self.input, self.hidden))
        self.output_weights = np.random.normal(loc = 0, scale = output_range, size = (self.hidden, self.output))
    
        self.input_update = np.zeros((self.input, self.hidden))
        self.output_weights = np.zeros((self.hidden, self.output))

    def feedForward(self, inputs):
        for i in range(self.input -1):
            self.input_list[i] = inputs[i]

        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.input_list[i] * self.input_weights[i][j]
            self.hidden_list[j] = sigmoid(sum)

        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.hidden_list[j] * self.output_weights[j][k]
            self.output_list[k] = sigmoid(sum)

        return self.output_list[:]

    def backPropagate(self, targets):
        # calculate error terms for output
        # the delta tell you which direction to change the weights
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.output_list[k])
            output_deltas[k] = dsigmoid(self.output_list[k]) * error

        # calculate error terms for hidden
        # delta tells you which direction to change the weights
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.output_weights[j][k]
            hidden_deltas[j] = dsigmoid(self.hidden_list[j]) * error

        # update the weights connecting hidden to output
        for j in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.hidden_list[j]
                self.output_weights[j][k] -= self.learning_rate * change + self.output_weights[j][k] * self.momentum
                self.output_weights[j][k] = change

        # update the weights connecting input to hidden
        for i in range(self.input):
            for j in range(self.hidden):
                change = hidden_deltas[j] * self.input_list[i]
                self.input_weights[i][j] -= self.learning_rate * change + self.input_update[i][j] * self.momentum
                self.input_update[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.output_list[k]) ** 2
        return error

    def test(self, dataset):
        for data in dataset:
            print(data[1], '->', self.feedForward(data[0]))

    def train(self, dataset):
        error_list = []
        epoch_num_list = []
        epoch_num = 1
        for i in range(self.num_epochs):
            error = 0.0
            random.shuffle(dataset)
            for data in dataset:
                inputs = data[0]
                targets = data[1]
                self.feedForward(inputs)
                error += self.backPropagate(targets)
            if i % 10 == 0:
                print('error %-.5f' % error)
            error_list.append(error)
            epoch_num_list.append(epoch_num)
            epoch_num += 1
            self.learning_rate = self.learning_rate * (self.learning_rate / (self.learning_rate + (self.learning_rate * self.rate_decay)))
        
        plt.scatter(epoch_num_list,error_list)
        plt.plot(epoch_num_list, error_list)
        plt.show()

    def predict(self, dataset):
        predictions = []
        for data in dataset:
            predictions.append(self.feedForward(data[0]))
        return predictions
    
    def get_answers(self, dataset):
        answers = []
        for data in dataset:
            answers.append(data[1])
        return answers

    def check_predictions(self, test_dataset, predictions, answers):
        num_guesses = 0
        num_correct = 0
        accuracy = 0
        for prediction, answer in zip(predictions, answers):
            if prediction.index(max(prediction)) == answer.index(max(answer)):
                num_correct += 1
            num_guesses += 1
            accuracy = num_correct / num_guesses
        return accuracy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1.0 - y)