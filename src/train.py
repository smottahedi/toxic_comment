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


def run_step(sess, model, inputs, seq_length, train, targets=None):
    """ Run one step in training."""
    

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    
    input_feed[model._inputs.name] = inputs
    if train:
        input_feed[model._target] = targets
    input_feed[model._seq_length] = seq_length

    # output feed: depends on whether we do a backward step or not.
    if train:
        output_feed = [model.train_ops,  # update op that does SGD.
                       model.gradient_norms,  # gradient norm.
                       model.losses,
                       model.accuracy]  # loss for this batch.
    else:
        output_feed = [model.predictions]  # loss for this batch.


    outputs = sess.run(output_feed, input_feed)
    if train:
        return outputs[1], outputs[2], outputs[3], None  # Gradient norm, loss, no outputs.
    else:
        return outputs  # No gradient norm, loss, accuracy, outputs.


def train():
    input_data = data.load_data(os.path.join(config.PROCESSED_PATH, 'train_ids.txt'))
    model = SeqClassifier(mode='train', pre_train=config.PRE_TRAINED)
    model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('running session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)

        if config.PRE_TRAINED:
            glove_embeddings = data.get_glove(config.GLOVE_PATH, config.VOCAB_PATH)
            sess.run(model.embedding_init, feed_dict={model._embedding_placeholder:glove_embeddings})    
        
        iteration = model.global_step.eval()
        total_loss = 0 
        total_accuracy = 0
        # for epoch in tqdm(range(config.EPOCHS)): 
        for _ in range(config.EPOCHS):
            skip_step = _get_skip_step(iteration)
            inputs, targets, seq_length = data.get_batch(input_data, 'train.csv', batch_size=config.BATCH_SIZE)
        
            start = time.time()
            step_grad_norm, step_loss, step_accuracy, _ = run_step(sess, model, inputs, seq_length, True, targets)

            total_loss += step_loss
            total_accuracy += step_accuracy
            iteration += 1
            

            if iteration % skip_step == 0:
                print('Iter {}:  loss {}, accuracy {}%, grad_norm {}, time {}'.format(iteration,
                                                         total_loss/skip_step,
                                                         total_accuracy/skip_step,
                                                         step_grad_norm,
                                                         time.time() - start))
                start = time.time()
                total_loss = 0
                total_accuracy = 0 
                saver.save(sess, os.path.join(config.CPT_PATH, 'SeqClassifier'), global_step=model.global_step)
                if iteration % (10 * skip_step) == 0:
                    # Run evals on development set and print their loss
                    # _eval_test_set(sess, model, test_buckets)
                    start = time.time()
                sys.stdout.flush()


def predict():
    input_data = data.load_data(os.path.join(config.PROCESSED_PATH, 'test_ids.txt'))
    inputs, ids, inputs_length = data.get_test_data(input_data, 'test.csv')
    model = SeqClassifier(mode='test', pre_train=config.PRE_TRAINED)
    model.build_graph()

    saver = tf.train.Saver()
    outputs = []
    with tf.Session() as sess:
        print('running test session')
        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver)
        for i in tqdm(range(len(inputs))):
            output = run_step(sess, model, [inputs[i]], [inputs_length[i]], targets=None, train=False)
            outputs.append(output[0])
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    outputs = np.asarray(outputs).reshape((-1, 6))
    df = pd.DataFrame(outputs, columns=list_classes)
    df.insert(0, 'id', ids)
    df.to_csv(os.path.join(config.DATA_PATH, 'submit.csv'), index=False)



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