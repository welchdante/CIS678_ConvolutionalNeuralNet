from NeuralNet import *

def load_data(filename):
    data = np.loadtxt(filename, delimiter = ',')
    
    y = data[:,0:10]
    
    data = data[:,10:]
    data -= data.min() 
    data /= data.max() 
    
    out = []
    print(data.shape)
    for i in range(data.shape[0]):
        tuple_data = list((data[i,:].tolist(), y[i].tolist())) 
        out.append(tuple_data)
    return out

X = load_data("sklearn_digits_train.csv")
y = load_data("sklearn_digits_test.csv")
print(X[9])
NN = NeuralNetwork(64, 10, 10, num_epochs = 100, learning_rate = 0.1, momentum = 0.5, rate_decay = 0.01)
NN.train(X)
NN.test(X)
predict = NN.predict(y)
answers = NN.get_answers(y)
NN.check_predictions(y, predict, answers)