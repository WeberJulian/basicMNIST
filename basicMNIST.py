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









