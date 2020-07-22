import collections

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import math as m

from . import vae_utils

# based on implementation here:
# https://github.com/tensorflow/models/blob/master/autoencoder/autoencoder_models/VariationalAutoencoder.py

class VariationalAutoencoder(object):

    def __init__(self, name, n_input1=3, n_input2=256, n_output=4, n_channels=3, n_weights=2048, drate=0.2, n_filters=8, filter_size=8, maxpool=4):
        
        self.n_input1 = n_input1
        self.n_input2 = n_input2
        self.n_output = n_output
        self.n_channels = n_channels
        self.n_weights = n_weights
        self.n_hlayers = len(n_weights)
        self.n_conv = len(n_filters)
        self.drate = drate
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.maxpool = maxpool

        network_weights = self._create_weights()
        self.weights = network_weights
        self.nonlinearity = tf.nn.relu

    def _calc_z_mean_and_sigma(self,x,y):
        with tf.name_scope("VI_encoder_q"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(y, shape=[-1, self.n_input2, 1, self.n_channels])
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    bnorm_name = 'b_conv_' + str(i)
                    conv_pre1 = tf.add(tf.nn.conv2d(conv_pool, self.weights['VI_encoder_q'][weight_name + '1'],strides=1,padding='SAME'),self.weights['VI_encoder_q'][bias_name + '1'])
                    #conv_pre2 = tf.add(tf.nn.conv1d(conv_pool, self.weights['VICI_VAE_encoder'][weight_name + '2'],stride=1,padding='SAME'),self.weights['VICI_VAE_encoder'][bias_name + '2'])
                    #conv_pre3 = tf.add(tf.nn.conv1d(conv_pool, self.weights['VICI_VAE_encoder'][weight_name + '3'],stride=1,padding='SAME'),self.weights['VICI_VAE_encoder'][bias_name + '3'])
                    #conv_post = self.nonlinearity(tf.concat([conv_pre1,conv_pre2,conv_pre3],axis=2))
                    conv_post = self.nonlinearity(conv_pre1)
                    conv_batchnorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([1,1,self.n_filters[i]], dtype=tf.float32)),tf.Variable(tf.ones([1,1,self.n_filters[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                    #conv_dropout = tf.nn.dropout(conv_batchnorm,rate=self.drate)
                    conv_pool = tf.nn.max_pool2d(conv_batchnorm,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')

                fc = tf.concat([x,tf.reshape(conv_pool, [-1, int(self.n_input2*self.n_filters[-1]/(np.prod(self.maxpool)))])],axis=1)

            else:
                fc = tf.concat([x,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_q'][weight_name]), self.weights['VI_encoder_q'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_batchnorm = tf.nn.batch_normalization(hidden_post,tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32)),tf.Variable(tf.ones([self.n_weights[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                hidden_dropout = tf.layers.dropout(hidden_batchnorm,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_q']['w_loc']), self.weights['VI_encoder_q']['b_loc'])
            scale = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_q']['w_scale']), self.weights['VI_encoder_q']['b_scale'])

            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale', scale)
            return loc, scale


    def _sample_from_gaussian_dist(self, num_rows, num_cols, mean, log_sigma_sq):
        with tf.name_scope("sample_in_z_space"):
            eps = tf.random.normal([num_rows, num_cols], 0, 1., dtype=tf.float32)
            sample = tf.add(mean, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
        return sample

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VI_ENC_q"):
            # Encoder
            all_weights['VI_encoder_q'] = collections.OrderedDict()
            
            if self.n_conv is not None:
                dummy = self.n_channels
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VI_encoder_q'][weight_name + '1'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_encoder_q'][bias_name + '1'] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    #all_weights['VICI_VAE_encoder'][weight_name + '2'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size*2, dummy*self.n_filters),[self.filter_size*2, dummy, self.n_filters]), dtype=tf.float32)
                    #all_weights['VICI_VAE_encoder'][bias_name + '2'] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    #all_weights['VICI_VAE_encoder'][weight_name + '3'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size*4, dummy*self.n_filters),[self.filter_size*4, dummy, self.n_filters]), dtype=tf.float32)
                    #all_weights['VICI_VAE_encoder'][bias_name + '3'] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    tf.summary.histogram(weight_name + '1', all_weights['VI_encoder_q'][weight_name + '1'])
                    tf.summary.histogram(bias_name + '1', all_weights['VI_encoder_q'][bias_name + '1'])
                    #tf.summary.histogram(weight_name + '2', all_weights['VICI_VAE_encoder'][weight_name + '2'])
                    #tf.summary.histogram(bias_name + '2', all_weights['VICI_VAE_encoder'][bias_name + '2'])
                    #tf.summary.histogram(weight_name + '3', all_weights['VICI_VAE_encoder'][weight_name + '3'])
                    #tf.summary.histogram(bias_name + '3', all_weights['VICI_VAE_encoder'][bias_name + '3'])
                    dummy = self.n_filters[i]

                fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters[-1]/(np.prod(self.maxpool)))
            else:
                fc_input_size = self.n_input1 + self.n_input2*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VI_encoder_q'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_encoder_q'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VI_encoder_q'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_encoder_q'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VI_encoder_q']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output),dtype=tf.float32)
            all_weights['VI_encoder_q']['b_loc'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VI_encoder_q']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_encoder_q']['b_loc'])           
            all_weights['VI_encoder_q']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output),dtype=tf.float32)
            all_weights['VI_encoder_q']['b_scale'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VI_encoder_q']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VI_encoder_q']['b_scale'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
