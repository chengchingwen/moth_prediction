import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as ran
BATCH=600 
TEST=100
ERROR=25
learning_rate = 0.001
training_iters = 3000
batch_size = BATCH-TEST
display_step = 100
n_input = 7 
n_steps = 20 
n_hidden = 20
n_classes = 7



import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.ops import rnn, rnn_cell


def add_layer( x , in_size, out_size, fun = None, dropout=0.9, status=None):
    W = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(0.1)
    Z =  tf.matmul(x, W) + b
    if fun:
        Z= fun(Z)
    if status is None:
        Z = tf.nn.dropout( Z ,dropout)
    else:
        Z *= dropout
    return Z

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, 1])
z = tf.placeholder("float", [None, 2])
s = tf.placeholder("float")
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(0, n_steps, x)
    cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0,state_is_tuple=True)
    state = cell.zero_state(batch_size, dtype=tf.float32)
    cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.7)
    cell = rnn_cell.MultiRNNCell([cell] * 3, state_is_tuple=True)
    cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=0.7)
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
    return (tf.matmul(outputs[19], weights['out']) + biases['out'])

n1 = RNN(x, weights, biases)
l1 = tf.concat(1,[z, n1])
l2 = add_layer(l1, n_classes+2 ,20, fun=tf.nn.tanh,status=s)
l3 = add_layer(l2, 20, 10, fun=tf.nn.tanh,status=s)
pred = add_layer(l3, 10 ,1,status=s)
cost = tf.reduce_sum(tf.reduce_mean(tf.square(pred - y)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct = tf.abs(pred-y) 
correct_pred = correct < ERROR
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.initialize_all_variables()
saver = tf.train.Saver()

def predict(i, i2):
    with tf.Session() as sess:
        saver.restore(sess, "./tmp/model.2500")
        sess.run(init)
        batch_x, batch_z = i, i2
        batch_x = batch_x.reshape((1, n_steps, n_input))
        batch_z = batch_z.reshape((1, 2))
        predition = sess.run(pred, feed_dict={x: batch_x, z:batch_z})
        print(predition[0][0])
