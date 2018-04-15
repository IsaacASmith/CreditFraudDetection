#!/usr/bin/python


#http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
#https://www.kaggle.com/mlg-ulb/creditcardfraud


import tensorflow as tf
import pandas as pa

def main():


    # --- Define neural net hyper parameters ---


    step_size = 0.0001
    epochs = 3
    batch_size = 4096

    input_layer_width = 30
    layer1_width = 40
    layer2_width = 50
    layer3_width = 60
    output_layer_width = 1

    data_count = 284806


    # --- Format the data into a tf usable format ---


    # Open the credit card data file
    csv = openCSV("creditcard.csv")

    # Break the list into two lists, one of fraudulent and one of non-fraudulent
    fraudList = csv[csv.Class == 1]
    nonFraudList = csv[csv.Class == 0]

    # Build the list of training data in a random order
    trainListInput = nonFraudList.sample(frac = .85)
    trainListInput = pa.concat([trainListInput, fraudList.sample(frac = .85)], axis = 0)
    trainListInput = trainListInput.sample(frac = 1)
    trainListExpected = trainListInput.Class.as_matrix()

    # Build the list of testing data -- anything not in the training data
    testListInput = (csv.loc[~csv.index.isin(trainListInput.index)]).sample(frac = 1)
    testListExpected = testListInput.Class.as_matrix()

    # Remove the class from the input and convert the lists to matrices
    trainListInput = trainListInput.drop('Class', axis = 1).as_matrix()
    testListInput = testListInput.drop('Class', axis = 1).as_matrix()


    # --- Set up initial layer weights ---


    # Define input layer 
    inputLayer = tf.placeholder(tf.float32, [None, input_layer_width])

    # Define input -> 1 weights
    Weights1 = tf.Variable(tf.random_normal([input_layer_width, layer1_width], stddev = .02), name = 'Weights1')
    Bias1 = tf.Variable(tf.random_normal([layer1_width]), name = 'Bias1')

    # Define layer 1 -> layer 2 weights
    Weights2 = tf.Variable(tf.random_normal([layer1_width, layer2_width], stddev = .02), name = 'Weights2')
    Bias2 = tf.Variable(tf.random_normal([layer2_width]), name = 'Bias2')

    #Define layer 2 -> layer 3 weights
    Weights3 = tf.Variable(tf.random_normal([layer2_width, layer3_width], stddev = .02), name = 'Weights3')
    Bias3 = tf.Variable(tf.random_normal([layer3_width]), name = 'Bias3')

    #Define layer 3 -> output weights
    Weights4 = tf.Variable(tf.random_normal([layer3_width, output_layer_width], stddev = .02), name = 'Weights4')
    Bias4 = tf.Variable(tf.random_normal([output_layer_width]), name = 'Bias4')

    # Define output layer
    outputLayer = tf.placeholder(tf.float32, [None])


    # --- Set up connections between layers ---


    # Matrix Multiply the input and weights plus the bias for the layer, then RELU it
    hidden_out1 = tf.nn.sigmoid(tf.add(tf.matmul(inputLayer, Weights1), Bias1))
    hidden_out2 = tf.nn.relu(tf.add(tf.matmul(hidden_out1, Weights2), Bias2))
    hidden_out3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out2, Weights3), Bias3))
    output = tf.add(tf.matmul(hidden_out3, Weights4), Bias4)

    # Define the cost function
    cost = tf.reduce_mean(output - outputLayer) #-tf.losses.mean_squared_error(labels = outputLayer, predictions = output)

    # Define the backpropagation step
    backprop = tf.train.GradientDescentOptimizer(learning_rate = step_size).minimize(cost)


    # --- Run the learning ---


    forwardOutput = tf.reduce_sum(output)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batchesPerEpoch = (int(len(trainListInput) / batch_size))

        for epoch in range(epochs):
            avg_cost = 0
            for batchIndex in range(batchesPerEpoch):

                inputBatch = trainListInput[batchIndex * batch_size : (1 + batchIndex) * batch_size]
                expectedBatch = trainListExpected[batchIndex * batch_size : (1 + batchIndex) * batch_size]
                
                _, c = sess.run([backprop, cost], feed_dict={inputLayer: inputBatch, outputLayer: expectedBatch})

                avg_cost += (c / batchesPerEpoch)

                print("Batch Cost: ", c)
            print("------------------------------------------------------------")
            print("Epoch Number: ", (epoch + 1), "      Epoch Cost: ", avg_cost)
            print("------------------------------------------------------------")


def openCSV(filename):
    return pa.read_csv(filename)


def normalize(tensor):
    return tf.div(
       tf.subtract(
          tensor, 
          tf.reduce_min(tensor)
       ), 
       tf.subtract(
          tf.reduce_max(tensor), 
          tf.reduce_min(tensor)
       )
    )




# Run it
main()
