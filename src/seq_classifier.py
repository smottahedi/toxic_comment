"""
Sequence classification model.
"""

import functools
import time
from attention import attention
from nn_layer import nn_layer, variable_summaries, column_loss
import tensorflow as tf

import config



def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SeqClassifier(object):
    """."""

    def __init__(self, mode, pre_train):
        self.mode = mode
        self.pre_train = pre_train
        self.embedding = None
        self.loss = None
        self.output = None
    
    @lazy_property
    def _create_placeholders(self):
        self._inputs = tf.placeholder(
            tf.int32, shape=[None, config.MAX_SEQ_LENGTH], name='input_placeholder')

        self._seq_length = tf.placeholder(tf.int32, shape=[None], name='seq_length_placeholder')
        
        self._target = tf.placeholder(
            tf.float32, shape=[config.BATCH_SIZE, config.NUM_CLASSES], name='target_placeholder')

        if self.pre_train:
            self._embedding_placeholder = tf.placeholder(
                tf.float32, shape=[config.VOCAB_SIZE, config.GLOVE_SIZE])
        else:
            self._embedding_placeholder = tf.placeholder(
                tf.float32, shape=[config.VOCAB_SIZE, config.EMBEDDING_DIMENSION])          

    @lazy_property
    def _create_embedding(self):
        with tf.name_scope('Embeddings'):
            if self.pre_train:
                self.embedding = tf.Variable(tf.constant(0.0,
                                                        shape=[config.VOCAB_SIZE, 
                                                        config.GLOVE_SIZE]), trainable=config.TRAINABLE_EMBEDDING)
                self.embedding_init = self.embedding.assign(self._embedding_placeholder)

            else:
                self.embedding = tf.Variable(tf.random_uniform(
                    [config.VOCAB_SIZE, config.EMBEDDING_DIMENSION], -1.0, 1.0))

    @lazy_property
    def _inference(self):

        embed = tf.nn.embedding_lookup(self.embedding, self._inputs)

        with tf.name_scope('Bi-GRU'):
            with tf.variable_scope('forward'):
                lstm_fw_cell = tf.contrib.rnn.GRUCell(
                    config.HIDDEN_LAYER_SIZE)
                if self.mode == 'train':
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=config.KEEP_PROB)

            with tf.variable_scope('backward'):
                lstm_bw_cell = tf.contrib.rnn.GRUCell(
                    config.HIDDEN_LAYER_SIZE)
                if self.mode == 'train':
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=config.KEEP_PROB)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                            cell_bw=lstm_bw_cell,
                                                            inputs=embed,
                                                            sequence_length=self._seq_length,
                                                            dtype=tf.float32,
                                                            scope='Bi-GRU')
        states = tf.concat(values=states, axis=1)
        with tf.name_scope('Attention'):
            attention_output, self.alphas = attention(outputs, config.ATTENTION_SIZE, return_alphas=config.RETURN_ALPHA)
            if self.mode == 'train':
                attention_output = tf.nn.dropout(attention_output, config.KEEP_PROB)


        layer_1 = nn_layer(attention_output, 2 * config.HIDDEN_LAYER_SIZE, config.HIDDEN_LAYER_SIZE // 2, 'layer_1', act=tf.nn.relu)
        self.output = nn_layer(layer_1, config.HIDDEN_LAYER_SIZE // 2, config.NUM_CLASSES, 'output')
        self.predictions = tf.nn.sigmoid(self.output)
        


    @lazy_property
    def _create_loss(self):

        with tf.name_scope('loss'):
            softmax = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output,
                                                        labels=self._target)
            # auc_score = tf.metrics.auc()
            self.losses = tf.reduce_mean(softmax)
        tf.summary.scalar('losses', self.losses)


    @lazy_property
    def _accuracy(self):
        self.accuracy = column_loss(label=self._target,
                                    pred=self.predictions, 
                                    func=tf.metrics.auc)


    @lazy_property
    def _create_optimizer(self):
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                           name='global_step')
            
            if self.mode == 'train':
                with tf.name_scope('learning_rate'):
                    self.learning_rate = tf.train.exponential_decay(config.LR, self.global_step,
                                            10000, 0.96, staircase=True)
                    tf.summary.scalar('learning_rate', self.learning_rate)
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainbales = tf.trainable_variables()
                start = time.time()
                clipped_grads, self.gradient_norms = tf.clip_by_global_norm(tf.gradients(self.losses, trainbales), config.MAX_GRAD_NORM)
                self.train_ops = self.optimizer.apply_gradients(zip(clipped_grads, trainbales), global_step=self.global_step)
                print('creating opt took {} seonds'.format(time.time() - start))


    def build_graph(self):
        self._create_placeholders
        self._create_embedding
        self._inference
        self._create_loss
        self._accuracy
        self._create_optimizer


if __name__ == '__main__':
    model = SeqClassifier('train', True)
    model.build_graph()
