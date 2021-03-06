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
import logging

logging.basicConfig(
	filename='logfile.log',
	format='%(asctime)s %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S%z',
	level=logging.DEBUG
	)

with open('card_text.txt', 'r') as f:
	data = f.read()

sentences = data.split('\n')

max_sentence_length = max([len(x) for x in sentences])

chars = set(data)

char_indices, indices_char = indice_transformer(chars)

# TF parameters

n_hidden = 512
seq_length = 5
learning_rate = 0.001
batch_size = 1
num_epochs = 40
display_step = 1
input_dropout = 0.90
output_dropout = 0.90
num_sentences_for_epoch = len(sentences)

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
	loss_across_iteration = []
	acc_across_iteration = []
	preds = []
	count_tildes = 0
	count_iterations = len(list(range(max_sentence_length - seq_length)))
	tf.add_to_collection('predictor', pred)
	tf.add_to_collection('feed_x', x)

	while step < num_epochs: # set up cycle of epochs

		start_time = datetime.now()
		counter = 0

		for index in random_sets(sentences, num_sentences_for_epoch): # per epoch, cycle through each sentences
			sent = return_padded_sentence(sentences, index, max_sentence_length)
			time_then = datetime.now()
			offset = 0 
			sequence_counter = 0

			for sequence in list(range(max_sentence_length - seq_length)): # for character in the sentence we're looking at
				symbols_in_keys = symbols_in_transformer(sent, offset, seq_length, char_indices)
				symbols_out_onehot = symbols_out_transformer(len(chars), 
					sent[offset+seq_length], char_indices)

				_, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
					feed_dict={x: symbols_in_keys, y:symbols_out_onehot})
				if sent[offset+seq_length] != '~':
					loss_total += loss
					acc_total += acc
				else:
					count_tildes += 1
				if (offset + seq_length) < max_sentence_length-1:
					offset += 1
				sequence_counter+=1

			count_iterations_less_tildes = count_iterations - count_tildes
			average_loss_sentence = loss_total / count_iterations_less_tildes
			average_accuracy_sentence = acc_total / count_iterations_less_tildes
			# non_padding_predicts = count_iterations_less_tildes 
			time_elapsed = datetime.now() - time_then
			time_elapsed_all = datetime.now() - start_time


			loss_across_iteration.append(average_loss_sentence)
			acc_across_iteration.append(average_accuracy_sentence)
			acc_total = 0
			loss_total = 0
			count_tildes = 0
			counter += 1

			logging.debug(
					'(Epoch %s, Counter %s): Time Elapsed for Card %s, Average Loss Across Card %s, Average Accuracy Across Card %s, Time Elapsed Overall %s'  % (step, 
						counter, time_elapsed, 
						average_loss_sentence, average_accuracy_sentence,
						time_elapsed_all)
				)

		step += 1
		save_path = saver.save(session, 'model')
		jobs_done('Your job is finished')
