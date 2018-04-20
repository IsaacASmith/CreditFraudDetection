#!/usr/bin/python


#http://adventuresinmachinelearning.com/python-tensorflow-tutorial/
#https://www.kaggle.com/mlg-ulb/creditcardfraud


import tensorflow as tf
import pandas as pa


# --- Define neural net hyper parameters ---

step_size = 0.0001
epochs = 5000
batch_size = 4096
dropout = .9

input_layer_width = 30
layer1_width = 40
layer2_width = 50
layer3_width = 60
output_layer_width = 2


def main():


    # --- Format the data into a tf usable format ---


    # Open the credit card data file
    csv = openCSV("creditcard.csv")
    csv.loc[csv.Class == 0, 'NonFraud'] = 1
    csv.loc[csv.Class == 1, 'NonFraud'] = 0
    csv = csv.rename(columns={'Class':'Fraud'})

    # Break the list into two lists, one of fraudulent and one of non-fraudulent
    fraudList = csv[csv.Fraud == 1]
    nonFraudList = csv[csv.NonFraud == 1]

    # Build the list of training data in a random order
    trainListInput = nonFraudList.sample(frac = .75)
    trainListInput = pa.concat([trainListInput, fraudList.sample(frac = .75)], axis = 0)
    trainListInput = trainListInput.sample(frac = 1)
    trainListExpected = pa.concat([trainListInput.Fraud, trainListInput.NonFraud], axis = 1)

    # Build the list of testing data -- anything not in the training data
    testListInput = (csv.loc[~csv.index.isin(trainListInput.index)]).sample(frac = 1)
    testListExpected = pa.concat([testListInput.Fraud, testListInput.NonFraud], axis = 1)

    # Remove the class from the input and convert the lists to matrices
    trainListInput = trainListInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()
    testListInput = testListInput.drop('Fraud', axis = 1).drop('NonFraud', axis = 1).as_matrix()

    # A value to increase the weight of the fraudulent transactions, since it's pretty skewed
    augmentation = len(trainListInput) / (len(fraudList) * .75)
    trainListExpected.Fraud *= augmentation
    testListExpected.Fraud *= augmentation
    
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
    y_clipped = tf.clip_by_value(output, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # Define the backpropagation step
    optimizer = tf.train.AdamOptimizer(learning_rate = step_size).minimize(cross_entropy)

    # Define an accuracy assessment
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(output,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # --- Run the learning ---


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        batchesPerEpoch = (int(len(trainListInput) / batch_size))
        minCost = 1e10
        epochsFailed = 0
        epochsFailedMax = 100

        for epoch in range(epochs):
            avg_cost = 0
            for batchIndex in range(batchesPerEpoch):

                inputBatch = trainListInput[batchIndex * batch_size : (1 + batchIndex) * batch_size]
                expectedBatch = trainListExpected[batchIndex * batch_size : (1 + batchIndex) * batch_size]

                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: inputBatch, y: expectedBatch, dropout_keep: dropout})

                avg_cost += (c / batchesPerEpoch)

            test_accuracy, test_cost = sess.run([accuracy, cross_entropy], feed_dict={x: testListInput, y: testListExpected, dropout_keep: 1})

            #Determine if optimization has leveled off
            #if abs(test_accuracy ) < minCost:
            #    minCost = test_cost
            #    epochsFailed = 0
            #elif epochsFailed > epochsFailedMax:
            #    break
            #else:
            #    epochsFailed = epochsFailed + 1

            print("-----------------------------------------------------------------------")
            print("Number: ", (epoch + 1), " Test Accuracy: ", test_accuracy, " Test Cost: ", test_cost)
            
        print("")
        print("Finished Optimizing")
        print("Final Accuracy: ", test_accuracy)


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