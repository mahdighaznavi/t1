import tensorflow as tf
import numpy as np

hidden_layer = 64
window_size = 20



X = tf.placeholder(tf.float32, shape=[window_size])
Y = tf.placeholder(tf.float32, shape=[window_size])

fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer)
bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer)

outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X)

y = tf.concat(outputs[0], outputs[1])

W = tf.Variable(tf.random_normal(tf.float32, shape=[]))

b = tf.Variable(tf.random_normal(tf.float32, shape=[]))




