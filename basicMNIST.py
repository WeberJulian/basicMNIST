# Imports

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Hyperparameters

learningRate = 0.01
trainingIteration = 30
batchSize = 100
displayStep = 2

# Define placeholder for input and output (x and y)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

# Define weight and bias

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


# Define scopes
with tf.name_scope("xwb") as scope:
    model = tf.nn.softmax(tf.matmul(x, w) + b) # Linear model

with tf.name_scope("costFunction") as scope:
    costFunction = -tf.reduce_sum(y*tf.log(model)) # Cross entropy
    tf.summary.scalar("costFunction", costFunction)

with tf.name_scope("train") as scope: 
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(costFunction) # Training


init = tf.initialize_all_variables()

mergedOperator = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    summaryWriter = tf.summary.FileWriter("./logs", graph_def=sess.graph_def)

    # Training
    for i in range(trainingIteration):
        avgCost = 0.
        totalBatch = int(mnist.train.num_examples/batchSize)
        for j in range(totalBatch):
            batchXs, batchYs = mnist.train.next_batch(batchSize)
            sess.run(optimizer, feed_dict={x: batchXs, y: batchYs})
            avgCost += sess.run(costFunction, feed_dict={x: batchXs, y: batchYs})/totalBatch
            summaryStr = sess.run(mergedOperator, feed_dict={x: batchXs, y: batchYs})
            summaryWriter.add_summary(summaryStr, i * totalBatch + j)
        if i % displayStep == 0:
            print("Iteration : " + str(i) + " Cost : " + str(avgCost))
    print("Training completed")
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy : " + str(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
        


    










