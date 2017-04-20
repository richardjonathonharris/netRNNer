import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np

with open('card_text.txt', 'r') as f:
	data = f.read()

sentences = data.split('\n')

chars = set(data)
print('total chars', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print(char_indices)
print(indices_char)

print(len(data), len(chars), len(sentences))

# TF parameters

n_hidden = 512
seq_length = 25
learning_rate = 0.001
batch_size = 50
num_epochs = 500 
display_step = 10

x = tf.placeholder(tf.float32, shape=[None, seq_length, 1])
y = tf.placeholder(tf.float32, shape=[None, len(chars)])

weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, len(chars)]))
}

biases = {
	'out': tf.Variable(tf.random_normal([len(chars)]))
}

def RNN(x, weights, biases):
	x = tf.reshape(x, [-1, seq_length])
	x = tf.split(x, seq_length, 1)
	rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), 
		rnn.BasicLSTMCell(n_hidden)])
	outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	step = 0
	offset = random.randint(0, seq_length+1)
	end_offset = seq_length+1
	acc_total = 0
	loss_total = 0 

	while step < num_epochs:
		if offset > (len(data) - end_offset):
			offset = random.randint(0, seq_length+1)

		symbols_in_keys = [[char_indices[str(data[i])]] for i in range(offset, offset+seq_length)]
		symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, seq_length, 1])
		
		symbols_out_onehot = np.zeros([len(chars)], dtype=float)
		symbols_out_onehot[char_indices[str(data[offset+seq_length])]] = 1.0
		symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

		_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
			feed_dict={x: symbols_in_keys, y:symbols_out_onehot})

		loss_total += loss
		acc_total += acc

		if (step+1) % display_step == 0:
			print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))

			acc_total = 0
			loss_total = 0
			symbols_in = [data[i] for i in range(offset, offset + seq_length)]
			symbols_out = data[offset + seq_length]
			symbols_out_pred = indices_char[int(tf.argmax(onehot_pred, 1).eval())]
			print("%s - [%s] vs [%s]" % (''.join(symbols_in), symbols_out, symbols_out_pred))

		step += 1
		offset += (seq_length+1)