import numpy as np
import tensorflow as tf
import random

corpus = 'this is a simple text for train word embedding impl emention'

words = []
EM_SIZE = 15

for w in corpus.split(' '):
    words.append(w)

words = set(words)  # create set of words
number_of_words = len(words)

one_hot_vectors = []

word2number = {}

number2word = []

for i, word in enumerate(words):
    # create one_hot vectors
    v = np.zeros(EM_SIZE)
    v[i] = 1
    one_hot_vectors.append(v)
    # map words with integers
    word2number[word] = i
    number2word.append(word)

one_hot_vectors = np.asarray(one_hot_vectors)

# create pairs
window_size = 2
pairs = []
sentence = corpus.split(' ')
for i in range(0, len(sentence)):
    for j in range(max(0, i - window_size), min(len(sentence), i + window_size)):
        if sentence[i] != sentence[j]:
            pairs.append([sentence[i], sentence[j]])

# create centers and contexts one_hot array
centers = []
contexts = []
for p in pairs:
    centers.append(one_hot_vectors[word2number[p[0]]])
    contexts.append(one_hot_vectors[word2number[p[1]]])

# create placeholder for inputs
xx = tf.placeholder(tf.float32, shape=[None, EM_SIZE])
yy = tf.placeholder(tf.float32, shape=[None, EM_SIZE])

H_LAYER_SIZE = 30
# create center word embedding matrix
window1 = tf.Variable(tf.random_normal(shape=[EM_SIZE, H_LAYER_SIZE]))
b1 = tf.Variable(tf.random_normal(shape=[H_LAYER_SIZE]))

center_embedding = tf.add(tf.matmul(xx, window1), b1)

# create context word embedding matrix
window2 = tf.Variable(tf.random_normal(shape=[H_LAYER_SIZE, EM_SIZE]))
b2 = tf.Variable(tf.random_normal(shape=[EM_SIZE]))

contexts_embedding = tf.add(tf.matmul(center_embedding, window2), b2)

prediction = tf.nn.softmax(contexts_embedding, axis=1)

# loss function
loss = tf.reduce_mean(-tf.log(tf.reduce_sum(yy * prediction, axis=1)))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

for i in range(100):
    sess.run(train_step, feed_dict={yy: contexts, xx: centers})
    print('loss: ', sess.run(loss, feed_dict={xx: centers, yy: contexts}))
