"""
Sequence classification model.
"""

import numpy as np
import tensorflow as tf

import config
import data

class SeqClassifier(object):

    def __init__(self):
        self._inputs = tf.placeholder(tf.float32, shape=[None, None])
        self.target = tf.placeholder(tf.float32, shape=[None, 5])
          
    
    def create_placeholders(self):
        _inputs = tf.placeholder(tf.int32, shape=[])