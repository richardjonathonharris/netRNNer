import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np
import math
import sys
import random
from datetime import datetime
from communicate import jobs_done

with open('card_text.txt', 'r') as f:
	data = f.read()

sentences = data.split('\n')

max_sentence_length = max([len(x) for x in sentences])

chars = set(data)
print('total chars', len(chars))
char_indices = dict((c, i+1) for i, c in enumerate(chars))
indices_char = dict((i+1, c) for i, c in enumerate(chars))
char_indices['~'] = 0
indices_char[0] = '~'

print(char_indices)
print(indices_char)

print(len(data), len(chars), len(sentences))

print(max_sentence_length)

# TF parameters

n_hidden = 64
seq_length = 50
learning_rate = 0.001
batch_size = 1
num_epochs = 1
display_step = 1
num_sentences_for_epoch = 25

num_batch_per_epoch = math.ceil(len(sentences) / batch_size)
num_offsets_per_sentence = math.ceil(max_sentence_length / seq_length)

final_length_of_sentences = (seq_length * num_offsets_per_sentence)

x = tf.placeholder(tf.float32, shape=[None, seq_length, 1])
y = tf.placeholder(tf.float32, shape=[None, len(chars)+1])

weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, len(chars)+1]))
}

biases = {
	'out': tf.Variable(tf.random_normal([len(chars)+1]))
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

def generate_chain(past_text, current_pred=''):
	gen_text = past_text+current_pred
	if len(gen_text) > seq_length:
		counter_start = len(gen_text) - seq_length
		return gen_text[counter_start:]
	elif len(gen_text) < seq_length:
		counter_start = seq_length - len(gen_text) 
		return '~'*counter_start + gen_text
	else:
		return gen_text

print(char_indices)
print(len(list(range(max_sentence_length))))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
	session.run(init)
	step = 0
	offset = 0
	end_offset = seq_length+1
	acc_total = 0
	loss_total = 0 
	current_sentence_start = 0
	counter = 0
	loss_across_iteration = 0
	acc_across_iteration = 0

	while step < num_epochs: # set up cycle of epochs
		print('starting epochs')

		random_num = [random.randint(0, len(sentences)) for x in range(num_sentences_for_epoch)]
		for mini_batch in random_num: # per epoch, cycle through each sentences
			sent = sentences[mini_batch]
			print('Using card %s: %s' % (counter, sent))
			time_then = datetime.now()
			if len(sent) < max_sentence_length: # if sentence isn't max length, pad with '~'
				num_to_fill = max_sentence_length - len(sent)
				full_fill = '~' * num_to_fill
				sent += full_fill
			offset = 0 # set offset to 0

			for sequence in list(range(max_sentence_length - seq_length)): # for character in the sentence we're looking at
				symbols_in_keys = [[char_indices[str(sent[i])]] for i in range(offset, offset+seq_length)]
				symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, seq_length, 1]) # set up input of seq_length index numbers
				symbols_out_onehot = np.zeros([len(chars)+1], dtype=float) # setup zeroes in shape of possible outputs 
				symbols_out_onehot[char_indices[str(sent[offset+seq_length])]] = 1.0 
				symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])

				_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
					feed_dict={x: symbols_in_keys, y:symbols_out_onehot})

				loss_total += loss
				acc_total += acc
				if (offset + seq_length) < max_sentence_length-1:
					offset += 1

			print('Average loss for sentence %s' % (loss_total / (len(list(range(max_sentence_length - seq_length))))))
			print('Average accuracy for sentence %s' % (acc_total / (len(list(range(max_sentence_length - seq_length))))))
			print('Time elapsed for card %s' % (datetime.now() - time_then))
			loss_across_iteration += loss_total / (len(list(range(max_sentence_length - seq_length))))
			acc_across_iteration += acc_total / (len(list(range(max_sentence_length - seq_length))))
			acc_total = 0
			loss_total = 0
			counter += 1

		if (step) % display_step == 0:
			print("Iter= " + str(step) + ", Average Loss= " + \
                  "{:.6f}".format((loss_across_iteration/(num_sentences_for_epoch))/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*(acc_across_iteration/num_sentences_for_epoch)/display_step))

			loss_across_iteration = 0
			acc_across_iteration = 0

		step += 1
		save_path = saver.save(session, 'model.ckpt')
		print('Model saved in file: %s' % save_path)

	pred_text = 'my new card is this one -'
	for counter in range(100):
		send_text = generate_chain(pred_text)
		symbols_in = [char for char in send_text]
		symbols_in = [[char_indices[str(symbols_in[i])]] for i in range(len(send_text))]
		symbols_in = np.reshape(symbols_in, [-1, seq_length, 1])
		onehot_pred = session.run(pred, feed_dict={x: symbols_in})
		symbols_out_pred = indices_char[int(tf.argmax(onehot_pred, 1).eval())]
		pred_text += symbols_out_pred

	print(pred_text)
