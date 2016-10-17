import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import get_ft as g
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

def get_set():
    a = pd.read_csv("fin.csv")
    Y = []
    X = []
    X2 = []
    k=0
    for row in a.iterrows():
        if k >= BATCH:
            break
        try:
            x = np.nan_to_num(g.get_ft(row[1]).as_matrix()[:,1:].flatten().astype(np.float32))
            c = 0
            if np.array_equal(x, []):
                continue
            for i in x:
                if i == 0:
                    c+=1
            if c > 120:
                continue
            if len(x) != 140:
                continue
            if ran.random() >= 10.9:
                continue
            x=list(x)
            X2.append([row[1].ave, row[1].ave0])
            X.append( x )
            Y.append([row[1].ave2])
            k+=1
        except Exception as e:
    return np.array(X).astype(np.float32)[:,:,np.newaxis], np.array(Y).astype(np.float32),np.array(X2).astype(np.float32)
     
                 
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


_X,_Y,_X2 = shuffle(*get_set())
_X=_X.reshape(BATCH,20,7)
_Y=_Y.reshape(BATCH,1)
_X2=_X2.reshape(BATCH,2)
def get_batch():
    Xtrain = _X[:-TEST]
    Ytrain = _Y[:-TEST]
    Xtest  = _X[-TEST:]
    Ytest  = _Y[-TEST:]
    X2train = _X2[:-TEST]
    X2test  = _X2[-TEST:]
    
    return Xtrain, Ytrain, X2train, Xtest, Ytest, X2test


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

ll = []
tl = []
with tf.Session() as sess:
    saver.restore(sess, "./tmp/800/model.2400")
    sess.run(init)
    step = 1
    while step  < training_iters:
        batch_x, batch_y,batch_z, batch_xt, batch_yt,batch_zt = get_batch()
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_xt = batch_xt.reshape((TEST, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,z:batch_z})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: _X, y: _Y,z:_X2,s:0})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y,z:batch_z,s:0})
            losst = sess.run(cost, feed_dict={x: batch_xt, y: batch_yt,z:batch_zt,s:0})
            p = sess.run(n1, feed_dict={x: batch_xt, y: batch_yt,z:batch_zt,s:0})
            _C = sess.run(correct, feed_dict={x: batch_xt, y: batch_yt,z:batch_zt,s:0})
            ll.append(loss)
            tl.append(losst)
            print("Iter " + str(step) + \
                  ", testing Loss= " + \
                  "{:.6f}".format(losst) +   ", training Loss= " + \
                  "{:.6f}".format(loss) +", Testing Accuracy= " + \
                  "{:.5f}".format(acc))
            save_path = saver.save(sess, "./tmp/model.%d" % step)
            print( "Model saved in file: ", save_path)
        step += 1
    final = sess.run(pred, feed_dict={x: batch_xt, y: batch_yt,z:batch_zt,s:0})
    print("Optimization Finished!")
    plt.plot()
    plt.show()
    legend1 = plt.plot(ll, label='test_cost')
    legend2 = plt.plot(tl, label = "train_cost")
    plt.legend()
    plt.show()




