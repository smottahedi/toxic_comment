import tensorflow as tf
from seq_classifier import SeqClassifier
import data
import random
import time
import os
import argparse
import config
import sys
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def _get_skip_step(iteration):
    """ How many steps should the model train before it saves all the weights. """
    if iteration < 100:
        return 30
    return 100

def _get_random_bucket(train_buckets_scale):
    """ Get a random bucket from which to choose a training sample """
    rand = random.random()
    return min([i for i in range(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters")


def run_step(sess, model, inputs, seq_length, mode, summary, targets=None):
    """ Run one step in training."""


    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}

    input_feed[model._inputs.name] = inputs
    if mode == 'train' or mode == 'val':
        input_feed[model._target] = targets
    input_feed[model._seq_length] = seq_length

    # output feed: depends on whether we do a backward step or not.
    if mode == 'train':
        model.mode = 'train'
        output_feed = [summary,
                       model.train_ops,  # update op that does SGD.
                       model.gradient_norms,  # gradient norm.
                       model.losses]  # loss for this batch.
    elif mode == 'test':
        model.mode = 'test'
        output_feed = [model.predictions]  # loss for this batch.

    elif mode == 'val' and config.RETURN_ALPHA:
        model.mode = 'val'
        output_feed = [model.losses, model.alphas]
    elif mode == 'val' and not config.RETURN_ALPHA:
        model.mode = 'val'
        output_feed = [model.losses, model.accuracy]




    outputs = sess.run(output_feed, input_feed)
    if mode == 'train':
        return outputs[0], outputs[2], outputs[3], None  # Gradient norm, loss, no outputs.
    elif mode == 'test':
        return outputs  # No gradient norm, loss, accuracy, outputs.
    elif mode == 'val' and config.RETURN_ALPHA:

        return outputs[0], None, outputs[1],
    elif mode == 'val' and not config.RETURN_ALPHA:
        return outputs[0], outputs[1], None

def early_stopping(loss, patience, delta=0.):
    monitor_op = lambda a, b: np.less(a, b - delta)



class EarlyStopping(object):

    def __init__(self,  min_delta, patience):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
        self.wait = 0
        self.best = np.inf


    def update(self, loss):
        current = loss
        if self.monitor_op(loss - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True

def train():
    (train_input, train_target,
     test_input, test_target) = data.load_data(os.path.join(config.PROCESSED_PATH, 'train_ids.txt'),
                                os.path.join(config.DATA_PATH, 'train.csv'), mode='train')
    model = SeqClassifier(mode='train', pre_train=config.PRE_TRAINED)
    model.build_graph()

    saver = tf.train.Saver()
    attention_vis = []

    with tf.Session() as sess:

        print('running session')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        _check_restore_parameters(sess, saver)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(config.LOG_DIR + '/train', graph=tf.get_default_graph())


        if config.PRE_TRAINED:
            if os.path.isfile(os.path.join(config.PROCESSED_PATH, 'glove_embeddings.pkl')):
                print('load embeddings')
                with open(os.path.join(config.PROCESSED_PATH, 'glove_embeddings.pkl'), 'rb') as f:
                    glove_embeddings = pickle.load(f)
            else:
                print('get Glove')
                glove_embeddings = data.get_glove(config.GLOVE_PATH, config.VOCAB_PATH)
            sess.run(model.embedding_init, feed_dict={model._embedding_placeholder:glove_embeddings})

        iteration = model.global_step.eval()
        total_loss = 0
        total_auc = 0
        total_val_loss = 0
        early_stopping = EarlyStopping(0.0, np.inf)
        best_loss = 0
        start = time.time()
        for _ in range(config.EPOCHS):
            skip_step = _get_skip_step(iteration)
            inputs, targets, seq_length = data.get_batch(train_input, train_target, batch_size=config.BATCH_SIZE)


            summary, step_grad_norm, step_loss, _ = run_step(sess, model, inputs, seq_length, 'train', merged, targets)

            train_writer.add_summary(summary)
            total_loss += step_loss
            iteration += 1

            inputs, targets, seq_length = data.get_batch(test_input, test_target, batch_size=config.BATCH_SIZE)
            val_loss, accuracy, alphas = run_step(sess, model, inputs, seq_length, mode='val', summary=None, targets=targets)

            total_auc += accuracy
            total_val_loss += val_loss

            if iteration % 50 == 0:
                print('Iter {}:  loss: {}, validation loss: {} auc: {} grad_norm: {}, time {}'.format(iteration,
                                                            total_loss / 50,
                                                            total_val_loss / 50,
                                                            total_auc / 50,
                                                            step_grad_norm,
                                                            time.time() - start))


                start = time.time()
                sys.stdout.flush()
                if early_stopping.update(step_loss):
                    print('early stopping')
                    break
                if total_auc / 50 > best_loss:
                    best_loss = total_auc / 50
                    saver.save(sess, os.path.join(config.CPT_PATH, 'SeqClassifier'), global_step=model.global_step)
                total_auc = 0
                total_val_loss = 0
                total_loss = 0


            if config.RETURN_ALPHA:
                attention_vis.append({str(a): b for a, b in zip(inputs[0], alphas[0])})

    if config.RETURN_ALPHA:
        with open('./data/attention_viz.pickle', 'wb') as f:
            from pickle import dump
            dump(attention_vis, f)



def predict():
    input_data, ids = data.load_data(os.path.join(config.PROCESSED_PATH, 'test_ids.txt'),
                                os.path.join(config.DATA_PATH, 'test.csv'), 'test')
    inputs, ids, inputs_length = data.get_test_data(input_data, ids)
    model = SeqClassifier(mode='test', pre_train=config.PRE_TRAINED)
    model.build_graph()

    saver = tf.train.Saver()
    outputs = []
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(config.LOG_DIR + '/test',graph=tf.get_default_graph())
        print('running test session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)


        num_chunks = 2000
        inputs_length_iterator = data.grouper_it(iterable=inputs_length, n=num_chunks)

        for input_chunk in tqdm(data.grouper_it(iterable=inputs, n=num_chunks)):

            input_length = list(next(inputs_length_iterator))
            output = run_step(sess, model, list(input_chunk), input_length, targets=None, mode='test', summary=None)
            outputs.append(output[0])
            # test_writer.add_summary(summary, i)


    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    outputs = np.concatenate(outputs)
    df = pd.DataFrame(outputs, columns=list_classes)
    df.insert(0, 'id', ids)
    num_submission = len(os.listdir(os.path.join(config.DATA_PATH, 'submissions')))
    df.to_csv(os.path.join(config.DATA_PATH, 'submissions',  'submit_' + str(num_submission) + '.csv'), index=False)



def main(args):
    if not os.path.isdir(config.PROCESSED_PATH):
        data.process_data()
    print('Data is ready!')

    data.make_dir(config.CPT_PATH)

    mode = args[-1]

    if mode == 'train':
        train()

    elif mode == 'test':
        predict()

tf.app.run()