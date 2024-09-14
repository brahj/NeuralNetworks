# importing the numpy library to make math easier
import numpy as np

# defining the sigmoid function - a standard activation function
def sigmoid(x):
    # returns f(x) = 1 / (1 + e^(-x)) -> the sigmoid function which turns all inputs into values between 0 and 1
    return 1 / (1 + np.exp(-x))

# defining the Neuron class - the building block of the Neural Network
class Neuron:

    # Default constructor - initializes a Neuron object
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    # method to handle the feedforward operation
    def feedforward(self, inputs):  
        # Total up the weighted inputs (by getting the dot product) and add bias
        total = np.dot(self.weights, inputs) + self.bias

        # Return the result of the activation function
        return sigmoid(total)
    
# Testing Neurons
weighs = np.array([0,0])
bias = 4
n = Neuron(weighs, bias)

# should use the defined Neuron [with preset weights and bias] to return the sigmoid of 2 inputs [using the total found on line 18]
x = np.array([2, 3])
print(n.feedforward(x))





# Combining Neurons into a Basic Network
class NeuralNetwork1:
    '''
        a neural network class with:
            - 2 inputs
            - a single hidden layer with 2 neurons (h1, h2)
            - an output layer with 1 neuron (o1)

        each neuron has the same weights and bias:
            - w = [0, 1]
            - b = 0
    '''

    # Default constructor - initializes a Basic Neural Network object with 2 hidden neurons and an output neuron
    def __init__(self):
        weights = np.array([0, 1]) # constant weights
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)

        self.o1 = Neuron(weights, bias)
    
    def feedforward(self, x):
        # calling the feedforward method on each Neuron Object
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # the inputs for o1 are the outputs for h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

# Testing the Basic Neural Network
network = NeuralNetwork1()

# should use the defined Network1 [with preset weights and bias] to return the sigmoid of a weightedtotal of two inputs
                                                        # that were got by getting sigmoid of a weighted total 2 other inputs each 
x = np.array([2, 3])
print(network.feedforward(x))