'''
    This is as basic a neural network as it gets
    It doesn't work 100% if the intial random weights and biases are unlucky
    This is clear if the loss isn't going down when running
'''

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid
    return np.exp(-x) / (1 + np.exp(-x))**2


# function that calculates the mean squared error - a measure of inaccuracy => we want to minimize this
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
class Network:
    '''
        A neural network class that will determine the gender of a person based on their height and weight
    '''

    def __init__(self):
        # 6 different weights, they start random and will then be trained
        # two weights for each neuron as in this network there are two inputs [height, weight]
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # 3 biases as well
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # two hidden layer neurons that each recieve the same inputs
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1])
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1])

        # an output neuron that recieves the outputs of the two hidden layer neurons
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2)
        return o1
    
    def training(self, data, all_y_trues):
        '''
            data is an (n x 2) numpy array
            y_trues is an (n x 1) numpy array
        '''
        learning_rate = 0.15 # the rate at which the weights will be adjusted per iteration
        epochs = 100000 # the amount of iterations the weights will be adjusted

        for epoch in range(epochs+1):

            # zip function => combines elements of the two arrays (data[x]'s and all_y_trues[y_true]'s) together into an array of tuples
            for x, y_true in zip(data, all_y_trues):

                # in-loop feedforward to get y_pred
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1


                # calculating partial derivatives
                # L is loss, w is a weight, and b is a bias
                # d_L_d_w1 would indicate the partial derivative of loss / partial derivative of w1

                d_L_d_ypred = -2 * (y_true - y_pred)

                # Hidden Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Hidden Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # Output Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)


                # update weights and biases

                # Hidden Neuron h1
                self.w1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Hidden Neuron h2
                self.w3 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Output Neuron o1
                self.w5 -= learning_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learning_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learning_rate * d_L_d_ypred * d_ypred_d_b3


            # Calculate total loss
            if epoch % 100 == 0:
                    y_preds = np.apply_along_axis(self.feedforward, 1, data)
                    loss = mse_loss(all_y_trues, y_preds)
                    print("Epoch %d loss: %.3f" % (epoch, loss));    
                
                

# Testing
data = np.array([[5.1, 3.5], [4.9, 3.0], [4.7, 3.2], [7.0, 3.2], [6.4, 3.2], [6.9, 3.1], [4.6, 3.1], [5.0, 3.4], [7.2, 3.5], [6.1, 2.9]])
y_trues = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])

network = Network()
network.training(data, y_trues)

# Predictions
Setosa = np.array([5.5, 3.8])
notSetosa = np.array([6.6, 3.0])
print("Actually a Setosa Flower - Prediction: %.0f" % network.feedforward(Setosa))
print("Not a Setosa Flower - Prediction: %.0f" % network.feedforward(notSetosa))

