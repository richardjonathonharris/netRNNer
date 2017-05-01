import random
import numpy as np

def indice_transformer(chars):
	char_indices = dict((c, i+1) for i, c in enumerate(chars))
	indices_char = dict((i+1, c) for i, c in enumerate(chars))
	char_indices['~'] = 0
	indices_char[0] = '~'
	return char_indices, indices_char

def generate_chain(past_text, seq_length, current_pred=''):
	gen_text = past_text+current_pred
	if len(gen_text) > seq_length:
		counter_start = len(gen_text) - seq_length
		return gen_text[counter_start:]
	elif len(gen_text) < seq_length:
		counter_start = seq_length - len(gen_text) 
		return '~'*counter_start + gen_text
	else:
		return gen_text

def random_sets(sentences, number_to_choose):
	sent_indexes = [random.randint(0, len(sentences)-1) for x in range(number_to_choose)]
	print('Selecting these indices %s' % sent_indexes)
	return sent_indexes

def return_padded_sentence(counter, text, index, max_sentence_length):
	text = text[index]
	print('Counter %s, Card Index: %s, Card Text: %s' % (counter, index, text))
	if len(text) < max_sentence_length: # if sentence isn't max length, pad with '~'
		num_to_fill = max_sentence_length - len(text)
		full_fill = '~' * num_to_fill
		text += full_fill
	return text

def symbols_in_transformer(sentence, offset, seq_length, indices):
	symbols_in_keys = [[indices[str(sentence[i])]] for i in range(offset, offset+seq_length)]
	return np.reshape(np.array(symbols_in_keys), [-1, seq_length, 1]) 

def symbols_out_transformer(char_length, character, indices):
	symbols_out_onehot = np.zeros([char_length+1], dtype=float) 
	symbols_out_onehot[indices[str(character)]] = 1.0 
	return np.reshape(symbols_out_onehot, [1, -1])