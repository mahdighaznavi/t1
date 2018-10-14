import tensorflow as tf
import numpy as np

# source of lstm: https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay

# Hyper parameters
EMBEDDING_DIM = 30
NUM_OF_WORDS = 15
WINDOW_SIZE = 2
NUM_OF_BATCH = 10
BATCH_SIZE = 10
A_SIZE = EMBEDDING_DIM
VARIABLE_SIZE = 8

embeddings = np.random.rand(NUM_OF_WORDS, 1)

data = np.asarray([np.asarray([embeddings[np.random.rand(1)][0] for i in range(4)]) for j in range(10)])

X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 4 * EMBEDDING_DIM])

# forward variables(f in the last of their names refers to it)
Af = tf.placeholder(tf.float32, shape=[BATCH_SIZE, EMBEDDING_DIM])
Cf = tf.placeholder(tf.float32, shape=[VARIABLE_SIZE, EMBEDDING_DIM])

Wcf = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Bcf = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wuf = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Buf = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wff = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Bff = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wof = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, BATCH_SIZE]))
Bof = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Guf = tf.sigmoid(tf.add(tf.matmul(Wuf, [Af, X]), Buf))
Gff = tf.sigmoid(tf.add(tf.matmul(Wcf, [Af, X]), Bff))
Gof = tf.sigmoid(tf.add(tf.matmul(Wof, [Af, X]), Gff))

CTf = tf.tanh(tf.add(tf.matmul(Wcf, [Af, X]), Bcf))

# backward variables
Ab = tf.placeholder(tf.float32, shape=[BATCH_SIZE, EMBEDDING_DIM])
Cb = tf.placeholder(tf.float32, shape=[VARIABLE_SIZE, EMBEDDING_DIM])

Wcb = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Bcb = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wub = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Bub = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wfb = tf.Variable(tf.random_normal(shape=[VARIABLE_SIZE, BATCH_SIZE]))
Bfb = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Wob = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, BATCH_SIZE]))
Bob = tf.Variable(tf.random_normal(shape=[5 * EMBEDDING_DIM]))

Gub = tf.sigmoid(tf.add(tf.matmul(Wub, [Ab, X]), Bub))
Gfb = tf.sigmoid(tf.add(tf.matmul(Wcb, [Ab, X]), Bfb))
Gob = tf.sigmoid(tf.add(tf.matmul(Wob, [Ab, X]), Gfb))

CTb = tf.tanh(tf.add(tf.matmul(Wcb, [Ab, X]), Bcb))

Wy = tf.Variable(tf.random_normal(shape=[BATCH_SIZE, 2*EMBEDDING_DIM]))
By = tf.Variable(tf.random_normal(shape=[2*EMBEDDING_DIM]))

Y = tf.nn.softmax(tf.add(tf.matmul(Wy, tf.concat(0, [Af, Ab])), By))

initCf = np.random.rand(VARIABLE_SIZE, EMBEDDING_DIM)
initCb = np.random.rand(VARIABLE_SIZE, EMBEDDING_DIM)

initAf = np.random.rand(BATCH_SIZE, EMBEDDING_DIM)
initAb = np.random.rand(BATCH_SIZE, EMBEDDING_DIM)

Cf_series = [initCf]

Cb_series = [initCb]

Af_series = [initAf]

Ab_series = [initAb]

for i in range(NUM_OF_BATCH):
    Ab_series.append(tf.matmul(Gob, tf.tanh(Cb)))
    Cb_series.append(tf.add(tf.matmul(Gub, CTb), tf.matmul(Gob, Cb_series[len(Cb_series)-1])))

    Af_series.append(tf.matmul(Gof, tf.tanh(Cf)))
    Cf_series.append(tf.add(tf.matmul(Guf, CTf), tf.matmul(Gof, Cf_series[len(Cf_series)-1])))

loss =