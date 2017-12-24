"""
Sequence classification model.
"""

import numpy as np
import tensorflow as tf

import config
import data

class SeqClassifier(object):

    def __init__(self):
        pass

    def create_placeholders(self):
        _inputs = tf.placeholder(tf.int32, )