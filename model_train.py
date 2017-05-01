import tensorflow as tf
from netRNNer.model import indice_transformer, generate_chain, random_sets, \
	return_padded_sentence, symbols_in_transformer, symbols_out_transformer



sess = tf.Session()
new_saver = tf.train.import_meta_graph('model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('variables')
for v in all_vars:
    # v_ = sess.run(v)
    print(v)
print(all_vars)
print(new_saver)

with open('card_text.txt', 'r') as f:
	data = f.read()

sentences = data.split('\n')

max_sentence_length = max([len(x) for x in sentences])

chars = set(data)

char_indices, indices_char = indice_transformer(chars)

pred_text = 'my new card is this one -'
symbols_out_pred = ' '
pred = tf.get_collection('predictor')[0]
x = tf.get_collection('feed_x')[0]
print(pred)
for counter in range(100):
	send_text = generate_chain(pred_text, 20)
	print('Full text is %s, sending in is %s, Current next char pred is %s' % (pred_text, send_text, symbols_out_pred))
	symbols_in = symbols_in_transformer(send_text, 0, len(send_text), char_indices)
	onehot_pred = sess.run(pred, feed_dict={x: symbols_in})
	symbols_out_pred = indices_char[int(tf.argmax(onehot_pred, 1).eval())]
	pred_text += symbols_out_pred

print(pred_text)