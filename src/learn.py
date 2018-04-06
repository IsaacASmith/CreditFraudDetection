#!/usr/bin/python

import tensorflow as tf
import pandas as pa

def main():

    # Open the credit card data file
    csv = openCSV("creditcard.csv")

    # Define neural net hyper parameters
    learning_rate = 0.1
    epochs = 1
    batch_size = 10000


    # Set up


    # Define input layer of size 31 (There are 31 values per data point)
    inputLayer = tf.placeholder(tf.float32, [None, 31])

    # Define output layer of size 1 (There is only one output value)
    outputLayer = tf.placeholder(tf.float32, [None, 1])

    # Define input -> 1 weights
    Weights1 = tf.Variable(tf.random_normal([31, 20], stddev = .03), name = 'Weights1')
    Bias1 = tf.Variable(tf.random_normal([20]), name = 'Bias1')

    # Define layer 1 -> layer 2 weights
    Weights2 = tf.Variable(tf.random_normal([20, 20], stddev = .03), name = 'Weights2')
    Bias2 = tf.Variable(tf.random_normal([20]), name = 'Bias2')

    #Define layer 2 -> layer 3 weights
    Weights3 = tf.Variable(tf.random_normal([20, 20], stddev = .03), name = 'Weights3')
    Bias3 = tf.Variable(tf.random_normal([20]), name = 'Bias3')

    #Define layer 3 -> output weights
    Weights4 = tf.Variable(tf.random_normal([20, 1], stddev = .03), name = 'Weights4')
    Bias4 = tf.Variable(tf.random_normal([1]), name = 'Bias4')


    # Set up connections between layers


    # Matrix Multiply the input and weights plus the bias for layer 1
    hidden_out1 = tf.add(tf.matmul(inputLayer, Weights1), Bias1)
    # RELU the result of the matrix multiply
    hidden_out1 = tf.nn.relu(hidden_out1)

    hidden_out2 = tf.add(tf.matmul(hidden_out1, Weights2), Bias2)
    hidden_out2 = tf.nn.relu(hidden_out2)

    hidden_out3 = tf.add(tf.matmul(hidden_out2, Weights3), Bias3)
    hidden_out3 = tf.nn.relu(hidden_out3)

    # Output layer
    output = tf.nn.softmax(tf.add(hidden_out3, Weights4), Bias4)


    #http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

def openCSV(filename):
    return pa.read_csv(filename)


# Run it
main()
