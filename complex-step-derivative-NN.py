from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np


##########################################################################################################

def createNN(inputs, structure):
    bias = []
    weights = []
    for idx, value in enumerate(structure):
        bias.append(np.array(np.random.random(value), dtype='complex'))  # changed dtype to complex
        if idx == 0:
            weights.append(np.array(np.random.random((value, inputs)), dtype='complex'))  # changed dtype to complex
        else:
            weights.append(
                np.array(np.random.random((value, structure[idx - 1])), dtype='complex'))  # changed dtype to complex
    return weights, bias


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neuron(inputs_neuron, weights_neuron, bias_neuron):  # determine activation
    s = np.dot(inputs_neuron, weights_neuron) + bias_neuron
    return sigmoid(s)


def evaluate_layer(inputs_layer, multi_weights, multi_bias):  # call each neuron
    new_layer_inputs = []
    for w, b in zip(multi_weights, multi_bias):
        new_layer_inputs.append(neuron(inputs_layer, w, b))
    return new_layer_inputs


def evaluate_nn(inputs, w, b):  # call each layer
    output = inputs
    for i in range(len(w)):
        output = evaluate_layer(output, w[i], b[i])
    return output


##########################################################################################################
# Until here same code as in previous 'neural network with numpy' delivery
##########################################################################################################

def generate_learning_set(num):
    learn_data = np.random.rand(num, 2)
    class_label = np.zeros(num)
    for idx, (x, y) in enumerate(learn_data):
        if (x > 0.5 and y > 0.5) or (x < 0.5 and y < 0.5):
            class_label[idx] = 1
    return learn_data, class_label


def update_weights(l_data, c_label, epochs=500, learning_rate=0.3):
    weights, bias = createNN(2, [4, 1])  # number of inputs, number of layers and neurons per layer, can be changed
    h = 0.0001
    new_weights = deepcopy(weights)
    new_bias = deepcopy(bias)
    temp_error = float('inf')
    for ep in range(epochs):
        weights = new_weights
        bias = new_bias
        error_sum = 0
        for idx, point in enumerate(l_data):
            error = c_label[idx] - evaluate_nn(point, weights, bias)
            error_sum += 1 / 2 * error ** 2
            for layer in range(len(weights)):
                for row in range(len(weights[layer])):
                    for w in range(len(weights[layer][row])):
                        weights_array = deepcopy(weights)
                        weights_array[layer][row][w] = weights[layer][row][w] + 1j * h
                        out = evaluate_nn(point, weights_array, bias)
                        der = out[0].imag / h
                        err_der = error * der
                        new_weights[layer][row][w] = weights[layer][row][w] + learning_rate * err_der
                    bias_array = deepcopy(bias)
                    bias_array[layer][row] = bias[layer][row] + 1j * h
                    out = evaluate_nn(point, weights, bias_array)
                    der = out[0].imag / h
                    err_der = error * der
                    new_bias[layer][row] = bias[layer][row] + learning_rate * err_der
        if ep % 5 == 0:  # print error of every 5th epoch
            print("Epoch (%i / %i), Error: %.4f" % (ep, epochs, error_sum.real))
        if error_sum.real > temp_error:  # if the error starts increasing, the training is stopped
            break
        temp_error = error_sum.real
    return new_weights, new_bias


if __name__ == '__main__':
    learn_data, class_label = generate_learning_set(300)  # number of data for the learning set, can be changed
    plt.figure("Learning Data")
    plt.title("Learning Set Data (class 0: green, class 1: blue)")
    # will display a plot that shows were the points of the learning set are lying (class 0: green, class 1: blue)
    for idx, data in enumerate(learn_data):
        if class_label[idx] == 0:
            plt.scatter(x=data[0], y=data[1], c='green')
        else:
            plt.scatter(x=data[0], y=data[1], c='blue')

    weights, bias = update_weights(learn_data,
                                   class_label)  # number of epochs and learning_rate can also be given as an input, otherwise default values will be taken

    test_data = np.random.rand(500, 2)  # number of data for the test set, can be changed
    plt.figure("Test Data")
    plt.title("Test Set Data (class 0: green, class 1: blue)")
    # will display a plot that shows were the points of the test set are classified (class 0: green, class 1: blue)
    for data in test_data:
        res = evaluate_nn(data, weights, bias)
        if res[0] < 0.5:
            plt.scatter(x=data[0], y=data[1], c='green')
        else:
            plt.scatter(x=data[0], y=data[1], c='blue')

    plt.show()
