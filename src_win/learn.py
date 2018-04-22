#!/usr/bin/python


#http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
#https://www.kaggle.com/mlg-ulb/creditcardfraud


import tensorflow as tf
import pandas as pa
import sys
import os

EPOCH_LOG_FILE = "epoch_log.csv"

# --- Define neural net hyper parameters ---

step_size = 0.000001
epochs = 5000
batch_size = 4096
dropout = .9

input_layer_width = 30
layer1_width = 60
layer2_width = 90
layer3_width = 60
output_layer_width = 2


def main():


    # --- Data Pre-Processing ---


    # Open the credit card data file
    csv = openCSV("creditcard.csv")
    csv.loc[csv.Class == 0, 'NonFraud'] = 1
    csv.loc[csv.Class == 1, 'NonFraud'] = 0
    csv = csv.rename(columns={'Class':'Fraud'})

    # Break the list into two lists, one of fraudulent and one of non-fraudulent
    fraudList = csv[csv.Fraud == 1]
    nonFraudList = csv[csv.NonFraud == 1]

    # Create the fraud lists
    fraudListTrainInput = fraudList.sample(frac = .75)
    fraudListTestInput = fraudList.loc[~fraudList.index.isin(fraudListTrainInput.index)]

    # Create the training fraud lists for later
    trainInputFraud = fraudListTrainInput[fraudListTrainInput.Fraud == 1]
    trainExpectedFraud = pa.concat([trainInputFraud.Fraud, trainInputFraud.NonFraud], axis = 1)
    trainInputFraud = trainInputFraud.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()

    # Augment the fraud lists to fix the dataset skewing
    fraudListTrainInput = fraudListTrainInput.sample(replace = True, frac = 600)
    fraudListTestInput = fraudListTestInput.sample(replace = True, frac = 600)
    fraudListTestExpected = pa.concat([fraudListTestInput.Fraud, fraudListTestInput.NonFraud], axis = 1)

    # Build the lists of training data in a random order
    trainListInput = nonFraudList.sample(frac = .75)
    trainListInput = pa.concat([trainListInput, fraudListTrainInput.sample(frac = 1)], axis = 0)

    trainListInput = trainListInput.sample(frac = 1)
    trainListExpected = pa.concat([trainListInput.Fraud, trainListInput.NonFraud], axis = 1)

    # Build list of non fraud training data for later
    trainInputNonFraud = trainListInput[trainListInput.NonFraud == 1]
    trainExpectedNonFraud = pa.concat([trainInputNonFraud.Fraud, trainInputNonFraud.NonFraud], axis = 1)
    trainInputNonFraud = trainInputNonFraud.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()

    # Build the lists of testing data -- anything not in the training data
    csv = csv[csv.NonFraud == 1]
    testListInput = (csv.loc[~csv.index.isin(trainListInput.index)]).sample(frac = 1)
    testListInput = pa.concat([testListInput, fraudListTestInput.sample(frac = 1)], axis = 0)

    nonFraudTestInput = testListInput[testListInput.NonFraud == 1]
    nonFraudTestExpected = pa.concat([nonFraudTestInput.Fraud, nonFraudTestInput.NonFraud], axis = 1)
    nonFraudTestInput = nonFraudTestInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()

    fraudListTestInput = fraudListTestInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()
    testListExpected = pa.concat([testListInput.Fraud, testListInput.NonFraud], axis = 1)

    # Remove the class from the input and convert the lists to matrices
    trainListInput = trainListInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()
    testListInput = testListInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()

    # Convert expected values to matrices
    trainListExpected = trainListExpected.as_matrix()
    testListExpected = testListExpected.as_matrix()


    # --- Set up initial layer weights ---


    # Define input layer 
    x = tf.placeholder(tf.float32, [None, input_layer_width])

    # Define output layer
    y = tf.placeholder(tf.float32, [None, output_layer_width])

    # Define input -> 1 weights
    Weights1 = tf.Variable(tf.random_normal([input_layer_width, layer1_width], stddev = .1), name = 'Weights1')
    Bias1 = tf.Variable(tf.random_normal([layer1_width]), name = 'Bias1')

    # Define layer 1 -> layer 2 weights
    Weights2 = tf.Variable(tf.random_normal([layer1_width, layer2_width], stddev = .1), name = 'Weights2')
    Bias2 = tf.Variable(tf.random_normal([layer2_width]), name = 'Bias2')

    # Define layer 2 -> layer 3 weights
    Weights3 = tf.Variable(tf.random_normal([layer2_width, layer3_width], stddev = .1), name = 'Weights3')
    Bias3 = tf.Variable(tf.random_normal([layer3_width]), name = 'Bias3')

    # Define layer 3 -> output weights
    Weights4 = tf.Variable(tf.random_normal([layer3_width, output_layer_width], stddev = .1), name = 'Weights4')
    Bias4 = tf.Variable(tf.random_normal([output_layer_width]), name = 'Bias4')


    # --- Set up connections between layers ---


    # Create a placeholder for the dropout
    dropout_keep = tf.placeholder(tf.float32)

    # Matrix Multiply the input and weights plus the bias for the layer, then RELU it
    hidden_out1 = tf.nn.relu(tf.add(tf.matmul(x, Weights1), Bias1))
    hidden_out2 = tf.nn.relu(tf.add(tf.matmul(hidden_out1, Weights2), Bias2))
    hidden_out3 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(hidden_out2, Weights3), Bias3)), dropout_keep)
    output = tf.add(tf.matmul(hidden_out3, Weights4), Bias4)

    # Define the cost function
    #y_clipped = tf.clip_by_value(output, 1e-10, .999999)
    #cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    MSE = tf.reduce_mean(tf.squared_difference(y, output))

    # Define the backpropagation step
    optimizer = tf.train.AdamOptimizer(learning_rate = step_size).minimize(MSE)

    # Define an accuracy assessment
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(output,1)), tf.float32))


    # --- Run the learning ---

    initEpochLog()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batchesPerEpoch = (int(len(trainListInput) / batch_size))

        for epoch in range(epochs):
            for batchIndex in range(batchesPerEpoch):

                # Form mini batches
                inputBatch = trainListInput[batchIndex * batch_size : (1 + batchIndex) * batch_size]
                expectedBatch = trainListExpected[batchIndex * batch_size : (1 + batchIndex) * batch_size]

                # Run the defined graph
                _, c = sess.run([optimizer, MSE], feed_dict={x: inputBatch, y: expectedBatch, dropout_keep: dropout})

            # Pull out metrics after each epoch
            fraud_precision_test, fraud_cost_test = sess.run([accuracy, MSE], feed_dict={x: fraudListTestInput, y: fraudListTestExpected, dropout_keep: 1})
            non_fraud_precision_test, non_fraud_cost_test = sess.run([accuracy, MSE], feed_dict={x: nonFraudTestInput, y: nonFraudTestExpected, dropout_keep: 1})
            fraud_precision_train, fraud_cost_train = sess.run([accuracy, MSE], feed_dict={x: trainInputFraud, y: trainExpectedFraud, dropout_keep: 1})
            non_fraud_precision_train, non_fraud_cost_train = sess.run([accuracy, MSE], feed_dict={x: trainInputNonFraud, y: trainExpectedNonFraud, dropout_keep: 1})
            train_accuracy, train_cost = sess.run([accuracy, MSE], feed_dict={x: trainListInput, y: trainListExpected, dropout_keep: 1})
            test_accuracy, test_cost= sess.run([accuracy, MSE], feed_dict={x: testListInput, y: testListExpected, dropout_keep: 1})

            # Log epoch results to csv
            logEpoch(test_accuracy, test_cost, fraud_precision_test, non_fraud_precision_test, train_accuracy, train_cost, fraud_precision_train, non_fraud_precision_train)

            print("-----------------------------------------------------------------------")
            print("Number: ", (epoch + 1), " Test Accuracy: ", test_accuracy, " Train Cost: ", train_cost)
            
        print("")
        print("Finished Optimizing")
        print("Final Accuracy: ", test_accuracy)


def openCSV(filename):
    return pa.read_csv(filename)

def initEpochLog():
    with open(EPOCH_LOG_FILE, 'w') as logfile:
        logfile.write("test_accuracy, test_cost, test_fraud_precision, test_non_fraud_precision, train_accuracy, train_cost, train_fraud_precision, train_non_fraud_precision")


def logEpoch(test_accuracy, test_cost, test_fraud_precision, test_non_fraud_precision, train_accuracy, train_cost, train_fraud_precision, train_non_fraud_precision):
    x = 5
    with open(EPOCH_LOG_FILE, 'a') as logfile:
        logfile.write('\n' + str(test_accuracy) + ',' + str(test_cost) + ',' + str(test_fraud_precision) + ',' + str(test_non_fraud_precision) + ',' + str(train_accuracy) + ',' + str(train_cost) + ',' + str(train_fraud_precision) + ',' + str(train_non_fraud_precision))


# Run it
main()