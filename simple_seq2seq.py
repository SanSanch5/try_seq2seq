# https://github.com/j-min/tf_tutorial_plus/tree/master/RNN_seq2seq
# To plot learning curve graph
import matplotlib.pyplot as plt

# for pretty print
from pprint import pprint

# for tokenizer
import re

# for word counter in vocabulary dictionary
from collections import Counter

# TensorFlow of Course :)
import tensorflow as tf

# The paths of RNNCell or rnn functions are too long.
from tensorflow.contrib.legacy_seq2seq.python.ops import *
from datetime import datetime
import os.path

from data.batches_handler import Dataset
from settings import *


def tokenizer(sentence):
    tokens = re.findall(r"[\w]+|[^\s\w]", sentence)
    return tokens


def build_vocab(sentences, is_target=False, max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    reverse_vocab = dict()

    for sentence in sentences:
        tokens = tokenizer(sentence)
        word_counter.update(tokens)

    if max_vocab_size is None:
        max_vocab_size = len(word_counter)

    if is_target:
        vocab['_GO'] = 0
        vocab['_PAD'] = 1
        vocab_idx = 2
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1
    else:
        vocab['_PAD'] = 0
        vocab_idx = 1
        for key, value in word_counter.most_common(max_vocab_size):
            vocab[key] = vocab_idx
            vocab_idx += 1

    for key, value in vocab.items():
        reverse_vocab[value] = key

    return vocab, reverse_vocab, max_vocab_size


def token2idx(word, vocab):
    return vocab[word]


dataset = Dataset("parsed.bak")

enc_vocab, enc_reverse_vocab, enc_vocab_size = build_vocab(dataset.data)
dec_vocab, dec_reverse_vocab, dec_vocab_size = build_vocab(dataset.labels, is_target=True)


def sent2idx(sent, vocab=enc_vocab, max_sentence_length=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length


def idx2token(idx, reverse_vocab):
    return reverse_vocab[idx]


def idx2sent(indices, reverse_vocab=dec_reverse_vocab):
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])


tf.reset_default_graph()

enc_inputs = tf.placeholder(
    tf.int32,
    shape=[None, enc_sentence_length],
    name='input_sentences')

sequence_lengths = tf.placeholder(
    tf.int32,
    shape=[None],
    name='sentences_length')

dec_inputs = tf.placeholder(
    tf.int32,
    shape=[None, dec_sentence_length + 1],
    name='output_sentences')

# batch major => time major
enc_inputs_t = tf.transpose(enc_inputs, perm=[1, 0])
dec_inputs_t = tf.transpose(dec_inputs, perm=[1, 0])

with tf.device('/cpu:0'):
    enc_Wemb = tf.get_variable('enc_word_emb',
                               initializer=tf.random_uniform([enc_vocab_size + 1, enc_emb_size]))
    dec_Wemb = tf.get_variable('dec_word_emb',
                               initializer=tf.random_uniform([dec_vocab_size + 2, dec_emb_size]))
    dec_out_W = tf.get_variable('dec_out_W',
                                initializer=tf.random_uniform([hidden_size, dec_vocab_size + 2]))
    dec_out_b = tf.get_variable('dec_out_b',
                                initializer=tf.random_uniform([dec_vocab_size + 2]))

with tf.variable_scope('encoder'):
    enc_emb_inputs = tf.nn.embedding_lookup(enc_Wemb, enc_inputs_t)

    # enc_emb_inputs:
    #     list(enc_sent_len) of tensor[batch_size x embedding_size]
    # Because `static_rnn` takes list inputs
    enc_emb_inputs = tf.unstack(enc_emb_inputs)

    enc_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    # enc_sent_len x batch_size x embedding_size
    enc_outputs, enc_last_state = tf.contrib.rnn.static_rnn(
        cell=enc_cell,
        inputs=enc_emb_inputs,
        sequence_length=sequence_lengths,
        dtype=tf.float32)

dec_outputs = []
dec_predictions = []
with tf.variable_scope('decoder') as scope:
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)

    for i in range(dec_sentence_length + 1):
        if i == 0:
            input_ = tf.nn.embedding_lookup(dec_Wemb, dec_inputs_t[i])
            state = enc_last_state
        else:
            scope.reuse_variables()
            input_ = tf.nn.embedding_lookup(dec_Wemb, dec_prediction)

        # dec_output: batch_size x dec_vocab_size+2
        # state:      batch_size x hidden_size
        dec_output, state = dec_cell(input_, state)
        dec_output = tf.nn.xw_plus_b(dec_output, dec_out_W, dec_out_b)

        # dec_prediction: batch_size x 1
        dec_prediction = tf.argmax(dec_output, axis=1)

        dec_outputs.append(dec_output)
        dec_predictions.append(dec_prediction)

# predictions: [batch_size x dec_sentence_lengths+1]
predictions = tf.transpose(tf.stack(dec_predictions), [1, 0])

# labels & logits: [dec_sentence_length+1 x batch_size x dec_vocab_size+2]
labels = tf.one_hot(dec_inputs_t, dec_vocab_size + 2)
logits = tf.stack(dec_outputs)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))

# training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    loss_history = []
    for epoch in range(n_epoch):
        all_preds = []
        epoch_loss = 0

        input_token_indices = []
        target_token_indices = []
        sentence_lengths = []
        for input_s, target_s in list(zip(*dataset.train.next_batch(batch_size))):
            input_sent, sent_len = sent2idx(input_s)
            target_sent = sent2idx(target_s,
                                   vocab=dec_vocab,
                                   max_sentence_length=dec_sentence_length,
                                   is_target=True)
            if sent_len > enc_sentence_length or len(target_sent) > dec_sentence_length + 1:
                print("The pair %s -> %s has a shape (%d, %d) bigger then (%d, %d) available" %
                      (input_s, target_s, sent_len, len(target_sent), enc_sentence_length, dec_sentence_length+1))
                continue

            input_token_indices.append(input_sent)
            sentence_lengths.append(sent_len)
            target_token_indices.append(target_sent)

        # Evaluate three operations in the graph
        # => predictions, loss, training_op(optimzier)
        batch_preds, batch_loss, _ = sess.run(
            [predictions, loss, training_op],
            feed_dict={
                enc_inputs: input_token_indices,
                sequence_lengths: sentence_lengths,
                dec_inputs: target_token_indices
            })
        loss_history.append(batch_loss)
        epoch_loss += batch_loss
        all_preds.append(batch_preds)

        # # Logging every 100 epochs
        # if epoch % 100 == 0:
        print('[%s] Epoch %i loss: %.2f' % (str(datetime.now()), epoch, epoch_loss))
        # print('\tepoch loss: {:.2f}\n'.format(epoch_loss))

    saver = tf.train.Saver()
    saver.save(sess, saved_model_file)
