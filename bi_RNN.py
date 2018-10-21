import tensorflow as tf
import numpy as np

hidden_layer = 64
window_size = 21
batch_size = 64
embedding_dim = 128
word_number = 100


corpus = 'salam'
embedding = {}

def create_batches(corpus, em):
    x_batch = []
    y_batch = []
    for start in range(0, len(corpus)):
        arr = []
        for j in range(int(start - window_size / 2), start + window_size + 1):
            if 0 <= j <= start + window_size:
                arr.append(embedding[corpus[j]])
                y_batch.append(one_hot[corpus[j]])
            else:
                arr.append(0)
        ret.append(arr)
    return ret


X = tf.placeholder(tf.float32, shape=[batch_size, window_size, embedding_dim])
Y = tf.placeholder(tf.float32, shape=[batch_size, word_number])
fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_layer)
bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_layer)
(output_fw, output_bw), output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X)
y = tf.concat(output_fw[:][window_size / 2 - 1][:], output_bw[:][window_size / 2 + 1][:], 0)
y = tf.manip.reshape(y, [batch_size, 2 * embedding_dim])
W = tf.Variable(tf.random_normal(tf.float32, shape=[2 * embedding_dim, word_number]))
b = tf.Variable(tf.random_normal(tf.float32, shape=[embedding_dim]))

prediction = tf.nn.softmax(tf.add(tf.matmul(y, W), b), axis=1)

loss = tf.reduce_mean(-tf.log(tf.reduce_sum(Y * prediction, axis=1)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

for i in range(100):
    sess.run(train_step, feed_dict=[X : , Y : ])
