##############################################################################
# Simple Backpropergation Neural Network to learn to recoginse numbers 0 - 9
# in a very limited grid. Based upon my 1992 undergrad Project; yes, neural
# networks arn't new! Just thought I'd see what it was like re-implementing
# it in python. It's not meant to be the best example of a backpropergation
# neural net but just a play thing.
#
# Numbers are represented in a 5x7 pxiel grid. Each pixel is either 0/1.
# Each number is shown below and this is the training set. The ASCII under
# the text "Number = N" displays the current status of the output neurons
# and a measure of their level of activation.
#
# Activation runs from 0 - 1
# >  0.85 '*'
# >  0.55 '^'
# >  0.30 '-'
# <= 0.30 '.'
# There are ten characters representing the ten output neurons activation and
# the target output starts 0 (left most). Using this you can watch the network
# learn.
#
# Number = 0          Number = 1          Number = 2
# *.........          .*........          ..*.......
#   #                   #                 ####
#  # #                 ##                     #
# #   #               # #                    #
# #   #                 #                   #
# #   #                 #                  #
#  # #                  #                 #
#   #                 #####                ####
#
# Number = 3          Number = 4          Number = 5
# ...*......          ....*.....          .....*....
#  ###                   #                 ####
# #   #                 ##                #
#     #                # #                #
#  ####               #  #                ####
#     #               #####                   #
# #   #                  #                #   #
#  ###                   #                 ###
#
# Number = 6          Number = 7          Number = 8
# ......*...          .......*..          ........*.
#  ###                #####                ###
# #   #                   #               #   #
# #                      #                #   #
# ####                  #                  ###
# #   #                #                  #   #
# #   #               #                   #   #
#  ###                #                    ###
#
# Number = 9
# .........*
#  ###
# #   #
# #   #
#  ####
#     #
#     #
#     #
#
# A prompt for [y|Y] after all 10 grid numbers have been presented and errors
# backpropergated. Press return to carry on with the next 10 presentations or
# enter [y|Y] to end.
##############################################################################

import numpy as np
import math
from itertools import cycle

training_set = {
    0: [0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0],
    1: [0, 0, 1, 0, 0,
        0, 1, 1, 0, 0,
        1, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        1, 1, 1, 1, 1],
    2: [1, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        0, 1, 1, 1, 1],
    3: [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0],
    4: [0, 0, 0, 1, 0,
        0, 0, 1, 1, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 1, 0,
        1, 1, 1, 1, 1,
        0, 0, 0, 1, 0,
        0, 0, 0, 1, 0],
    5: [0, 1, 1, 1, 1,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        0, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0],
    6: [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 0,
        1, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0],
    7: [1, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 0, 0,
        1, 0, 0, 0, 0,
        1, 0, 0, 0, 0],
    8: [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 0],
    9: [0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 1, 1, 1, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1,
        0, 0, 0, 0, 1],
}

target_set = {
    0: [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    1: [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    2: [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    3: [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    4: [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
    5: [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
    6: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
    7: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
    8: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
    9: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]
}

current_status_of_learning = {
    0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    4: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    6: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    8: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

# Cached lengths
patterns_to_train = len(training_set)
input_layer_size = len(training_set[0]) + 1
hidden_layer_size = 15
output_layer_size = len(target_set[0])

# Neural activations layers
input_neurons = np.zeros(shape=input_layer_size)
hidden_neurons = np.zeros(shape=hidden_layer_size)
output_neurons = np.zeros(shape=output_layer_size)

# Random weight and errors interconnects for input --> hidden
input_hidden_weights = np.random.uniform(
    -0.1, 0.1, size=(input_layer_size, hidden_layer_size))
input_hidden_weights_delta = np.zeros(
    shape=(input_layer_size, hidden_layer_size))
hidden_errors = np.zeros(shape=hidden_layer_size)

# Random weight and errors interconnects for hidden --> output
hidden_output_weights = np.random.uniform(
    -0.1, 0.1, size=(hidden_layer_size, output_layer_size))
hidden_output_weights_delta = np.zeros(
    shape=(hidden_layer_size, output_layer_size))
output_errors = np.zeros(shape=output_layer_size)

# Threshold (bias) neurons always activated
input_neurons[0] = 1.0
hidden_neurons[0] = 1.0

# Tweakable terms
learning_rate = 0.45
momentum = 0.9


def print_training():
    ''' Output the training set to the console
    '''
    print('Training so far...')

    count = 0
    keys = list(training_set.keys())
    pixels_per_pattern = {'width': 5, 'height': 7}
    char_height = pixels_per_pattern['height']
    char_width = pixels_per_pattern['width']
    number_to_print_per_row = 3

    while count < patterns_to_train:
        print()

        number_this_row = min(number_to_print_per_row,
                              patterns_to_train - count)

        for column_no in range(count, count + number_this_row):
            print('Number =', keys[column_no], end='')
            print(' ' * 10, end='')

        print(flush=True)

        for column_no in range(count, count + number_this_row):
            for val in current_status_of_learning[column_no]:
                marker = '.'
                if val > 0.85:
                    marker = '*'
                elif val > 0.55:
                    marker = '^'
                elif val > 0.3:
                    marker = '-'
                print(marker, end='')
            print(' ' * 10, end='')

        print(flush=True)

        for height in range(0, char_height):
            for column_no in range(count, count + number_this_row):
                start_inx = height * char_width
                end_inx = start_inx + char_width
                for width in range(start_inx, end_inx):
                    pixel = (
                        '#' if training_set[column_no][width] == 1 else ' ')
                    print(pixel, end='')
                print(' ' * 15, end='')
            print(flush=True)

        count += number_to_print_per_row

    print()


def learning_threshold_met(iteration):
    ''' Returns true if the network has converged enough on a
        solution.
    '''
    print_training()
    to_stop = input("Enter [Y|y] to stop training (%d): " % (iteration))
    return to_stop.lower() == 'y'


def store(item):
    ''' Stores off the results from the ouput against the
        testing vaule.
    '''
    current_status_of_learning[item] = output_neurons.tolist()


def adjust():
    ''' Adjusts the weights (backpropergation)
    '''
    for i in range(1, hidden_layer_size):
        for j in range(0, output_layer_size):
            hidden_output_weights_delta[i][j] = (
                learning_rate *
                output_errors[j] *
                hidden_neurons[i] +
                (momentum * hidden_output_weights_delta[i][j]))

    for i in range(1, hidden_layer_size):
        for j in range(0, output_layer_size):
            hidden_output_weights[i][j] += hidden_output_weights_delta[i][j]

    for i in range(1, input_layer_size):
        for j in range(1, hidden_layer_size):
            input_hidden_weights_delta[i][j] = (
                learning_rate *
                hidden_errors[j] *
                input_neurons[i] +
                (momentum * input_hidden_weights_delta[i][j]))

    for i in range(1, input_layer_size):
        for j in range(1, hidden_layer_size):
            input_hidden_weights[i][j] += input_hidden_weights_delta[i][j]


def error(item):
    ''' Computes the error from the expected.
    '''
    # Compute the output layer errors
    for j in range(0, output_layer_size):
        output_errors[j] = (
            output_neurons[j] *
            (1 - output_neurons[j]) *
            (target_set[item][j] - output_neurons[j]))

    # Compute the error in the hidden layer errors
    for j in range(1, hidden_layer_size):
        sum = 0
        for i in range(0, output_layer_size):
            sum += output_errors[i] * hidden_output_weights[j][i]
        hidden_errors[j] = hidden_neurons[j] * (1 - hidden_neurons[j]) * sum


def propergate():
    ''' Propogates the activation through the network
    '''
    # Input -> Hidden
    for j in range(1, hidden_layer_size):
        activation = 0
        for i in range(0, input_layer_size):
            activation += input_hidden_weights[i][j] * input_neurons[i]
        hidden_neurons[j] = 1 / (1 + math.exp(-activation))

    # Hidden -> Output
    for j in range(0, output_layer_size):
        activation = 0
        for i in range(0, hidden_layer_size):
            activation += hidden_output_weights[i][j] * hidden_neurons[i]
        output_neurons[j] = 1 / (1 + math.exp(-activation))


def present(item):
    ''' Presents the training item to the input neurons
    '''
    input_neurons.flat[1:] = training_set[item]


def teach_network():
    ''' Present the input to the network, propergate the activation,
        compute the errors and backpropergate by adjusting the weights
    '''
    train_value = cycle(list(training_set.keys()))
    iterations = 0
    for item in train_value:

        present(item)
        propergate()
        error(item)
        adjust()
        store(item)

        iterations += 1
        if (iterations % patterns_to_train) == 0:
            if learning_threshold_met(iterations):
                break


if __name__ == '__main__':
    teach_network()
