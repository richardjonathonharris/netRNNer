import tensorflow as tf
from tensorflow.contrib import rnn
import random
import numpy as np
import math
import sys
import random
from datetime import datetime
from communicate import jobs_done
from netRNNer.model import indice_transformer, generate_chain, random_sets, \
	return_padded_sentence, symbols_in_transformer, symbols_out_transformer

with open('card_text.txt', 'r') as f:
	data = f.read()

sentences = data.split('\n')

max_sentence_length = max([len(x) for x in sentences])

chars = set(data)

char_indices, indices_char = indice_transformer(chars)

# TF parameters

n_hidden = 48
seq_length = 50
learning_rate = 0.001
batch_size = 1
num_epochs = 1
display_step = 1
input_dropout = 0.90
output_dropout = 0.90
num_sentences_for_epoch = 250

num_batch_per_epoch = math.ceil(len(sentences) / batch_size)
num_offsets_per_sentence = math.ceil(max_sentence_length / seq_length)

final_length_of_sentences = (seq_length * num_offsets_per_sentence)

x = tf.placeholder(tf.float32, shape=[None, seq_length, 1], name='feed_x')
y = tf.placeholder(tf.float32, shape=[None, len(chars)+1], name='feed_y')

weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, len(chars)+1]), name='weights')
}

biases = {
	'out': tf.Variable(tf.random_normal([len(chars)+1]), name='biases')
}

def RNN(x, weights, biases):
	x = tf.reshape(x, [-1, seq_length])
	x = tf.split(x, seq_length, 1)
	rnn_cell = rnn.MultiRNNCell(
		[
		tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden), 
			input_keep_prob=input_dropout,
			output_keep_prob=output_dropout),
		tf.contrib.rnn.DropoutWrapper(rnn.BasicLSTMCell(n_hidden), 
			input_keep_prob=input_dropout,
			output_keep_prob=output_dropout)]
		)
	outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
	loss_across_iteration = []
	acc_across_iteration = []
	preds = []
	count_tildes = 0
	count_iterations = len(list(range(max_sentence_length - seq_length)))

	while step < num_epochs: # set up cycle of epochs
		tf.add_to_collection('predictor', pred)
		tf.add_to_collection('feed_x', x)

		for index in random_sets(sentences, num_sentences_for_epoch): # per epoch, cycle through each sentences
			sent = return_padded_sentence(counter, sentences, index, max_sentence_length)
			time_then = datetime.now()
			offset = 0 
			sequence_counter = 0

			for sequence in list(range(max_sentence_length - seq_length)): # for character in the sentence we're looking at
				symbols_in_keys = symbols_in_transformer(sent, offset, seq_length, char_indices)
				symbols_out_onehot = symbols_out_transformer(len(chars), 
					sent[offset+seq_length], char_indices)

				_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
					feed_dict={x: symbols_in_keys, y:symbols_out_onehot})
				# if counter % 100 == 0:
				# 	preds.append(indices_char[int(tf.argmax(onehot_pred, 1).eval())])
				if sent[offset+seq_length] != '~':
					loss_total += loss
					acc_total += acc
				else:
					count_tildes += 1
				if (offset + seq_length) < max_sentence_length-1:
					offset += 1
				sequence_counter+=1

			# Turn this on during actual run
			# if counter % 100 == 0:
			# 	print('Actual predictions across sentence %s' % ''.join(preds))
			# 	preds = []
			count_iterations_less_tildes = count_iterations - count_tildes
			print('Average loss for sentence %s' % (loss_total / count_iterations_less_tildes))
			print('Average accuracy for sentence %s' % (acc_total / count_iterations_less_tildes))
			print('Number of predictions for sentence that were not padding %s' % count_iterations_less_tildes)
			print('Time elapsed for card %s' % (datetime.now() - time_then))
			loss_across_iteration.append(loss_total / count_iterations_less_tildes)
			acc_across_iteration.append(acc_total / count_iterations_less_tildes)
			acc_total = 0
			loss_total = 0
			count_tildes = 0
			counter += 1

		if (step) % display_step == 0:
			print("\nIter= " + str(step) + ", Average Loss= " + \
                  "{:.6f}".format(sum(loss_across_iteration)/len(loss_across_iteration)) + ", Average Accuracy= " + \
                  "{:.2f}%\n".format(100*(sum(acc_across_iteration)/len(acc_across_iteration))))

		step += 1
		save_path = saver.save(session, 'model')
		print('Model saved in file: %s' % save_path)

	pred_text = 'my new card is this one -'
	symbols_out_pred = ' '
	for counter in range(100):
		send_text = generate_chain(pred_text, seq_length)
		print('Full text is %s, sending in is %s, Current next char pred is %s' % (pred_text, send_text, symbols_out_pred))
		symbols_in = symbols_in_transformer(send_text, 0, len(send_text), char_indices)
		onehot_pred = session.run(pred, feed_dict={x: symbols_in})
		symbols_out_pred = indices_char[int(tf.argmax(onehot_pred, 1).eval())]
		pred_text += symbols_out_pred

	print(pred_text)

