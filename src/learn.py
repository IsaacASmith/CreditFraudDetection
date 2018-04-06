#!/usr/bin/python

import tensorflow as tf
import pandas as pa

def main():


    # --- Format the data into a tf usable format ---


    # Open the credit card data file
    csv = openCSV("creditcard.csv")
    csv.set_index("Time", inplace = True)
    
    # Break the list into two lists, one of fraudulent and one of non-fraudulent
    fraudList = csv[csv.Class == 1]
    nonFraudList = csv[csv.Class == 0]

    # Build the list of training data
    trainList = fraudList.sample(frac = .85)
    trainList = pa.concat([trainList, nonFraudList.sample(frac = .85)], axis = 0)

    #Build the list of testing data -- anything not in the training data
    testList = csv.loc[~csv.index.isin(trainList.index)]

    # --- Set up initial layer weights ---


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


    # --- Set up connections between layers ---


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
    # Normalize output to be between 0 and 1 (ish)
    normalized = tf.clip_by_value(output, 1e-10, 0.9999999)

    # Define the cost function
    tmp = tf.placeholder(tf.float32, [None, 2])
    cost = -tf.reduce_sum(normalized * tf.log(tmp))

    # Define the backpropagation step
    backprop = tf.train.GradientDescentOptimizer(learning_rate = step_size).minimize(normalized)

    initialize = tf.global_variables_initializer()


    # --- Run the learning ---


    # Define neural net hyper parameters
    step_size = 0.1
    epochs = 1
    batch_size = 10000

    #with tf.Session() as sess:
    #    sess.run(initialize)

    #    for epoch in range(epochs):
    #        avg_cost = 0
            


    #http://adventuresinmachinelearning.com/python-tensorflow-tutorial/

def openCSV(filename):
    return pa.read_csv(filename)


# Run it
main()
