import collections

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#from .layers import batch_normalization
import numpy as np
import math as m

from . import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

class VariationalAutoencoder(object):

    def __init__(self, name, n_input=256, n_channels=3, n_filters=8, filter_size=8):
        
        self.n_input = n_input
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.filter_size = filter_size

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu

    def _compress(self,x):
        with tf.name_scope("VI_RNN"):
 
            # Reshape input to a 3D tensor - single channel
            X = tf.reshape(x, shape=[-1, self.n_input, 1, self.n_channels])
            weight_name = 'w_rnn_'
            bias_name = 'b_rnn_'
            conv_1 = tf.add(tf.nn.conv2d(X, self.weights['VI_rnn'][weight_name+'1'],strides=2,padding='SAME'),self.weights['VI_rnn'][bias_name+'1'])
            conv_post_1 = self.nonlinearity(conv_1)
            
            conv_2a = tf.add(tf.nn.conv2d(conv_post_1a, self.weights['VI_rnn'][weight_name+'2a'],strides=2,padding='SAME'),self.weights['VI_rnn'][bias_name+'2a'])
            conv_2b = tf.nn.conv2d(X, self.weights['VI_rnn'][weight_name+'2b'],strides=4,padding='SAME')
            conv_post_2 = self.nonlinearity(conv_2a+conv_2b)

            conv_3a = tf.add(tf.nn.conv2d(conv_post_2, self.weights['VI_rnn'][weight_name+'3a'],strides=2,padding='SAME'),self.weights['VI_rnn'][bias_name+'3a'])
            conv_3b = tf.nn.conv2d(conv_post_1, self.weights['VI_rnn'][weight_name+'3b'],strides=4,padding='SAME')
            conv_3b = tf.nn.conv2d(X, self.weights['VI_rnn'][weight_name+'3c'],strides=4,padding='SAME')
            conv_post_3 = self.nonlinearity(conv_3a+conv_3b+conv_3c)
 
            tf.summary.histogram('conv_post_3', conv_post_3)
            
            return tf.reshape(conv_post_3,(-1,self.n_input/8))

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VI_RNN"):            
            all_weights['VI_rnn'] = collections.OrderedDict()

            weight_name = 'w_rnn_'
            bias_name = 'b_rnn_'
            all_weights['VI_rnn'][weight_name+'1'] = tf.Variable(vae_utils.xavier_init(self.filter_size, 1 self.n_filters), dtype=tf.float32)
            all_weights['VI_rnn'][bias_name+'1'] = tf.Variable(tf.zeros(self.n_filters, dtype=tf.float32))

            all_weights['VI_rnn'][weight_name+'2a'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.n_filters, self.n_filters), dtype=tf.float32)
            all_weights['VI_rnn'][bias_name+'2a'] = tf.Variable(tf.zeros(self.n_filters, dtype=tf.float32))
            all_weights['VI_rnn'][weight_name+'2b'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.n_filters, self.n_filters), dtype=tf.float32)

            all_weights['VI_rnn'][weight_name+'3a'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.n_filters, 1), dtype=tf.float32)
            all_weights['VI_rnn'][bias_name+'3a'] = tf.Variable(tf.zeros(1, dtype=tf.float32))
            all_weights['VI_rnn'][weight_name+'3b'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.n_filters, 1), dtype=tf.float32)
            all_weights['VI_rnn'][weight_name+'3c'] = tf.Variable(vae_utils.xavier_init(self.filter_size, self.n_filters, 1), dtype=tf.float32)

            tf.summary.histogram(weight_name+'1', all_weights['VI_rnn'][weight_name+'1'])
            tf.summary.histogram(bias_name+'1', all_weights['VI_rnn'][bias_name+'1'])
            tf.summary.histogram(weight_name+'2a', all_weights['VI_rnn'][weight_name+'2a'])
            tf.summary.histogram(bias_name+'2q', all_weights['VI_rnn'][bias_name+'2a'])
            tf.summary.histogram(weight_name+'2b', all_weights['VI_rnn'][weight_name+'2b'])
            tf.summary.histogram(weight_name+'3a', all_weights['VI_rnn'][weight_name+'3a'])
            tf.summary.histogram(bias_name+'3a', all_weights['VI_rnn'][bias_name+'3a'])
            tf.summary.histogram(weight_name+'3b', all_weights['VI_rnn'][weight_name+'3b'])
            tf.summary.histogram(weight_name+'3c', all_weights['VI_rnn'][weight_name+'3c'])
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
