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

    def __init__(self, name, n_input=256, n_output=4, n_channels=3, n_weights=2048, n_modes=2, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_channels = n_channels
        self.n_weights = n_weights
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.n_hlayers = len(n_weights)
        self.n_conv = len(n_filters)
        self.n_modes = n_modes
        self.drate = drate
        self.maxpool = maxpool

        network_weights = self._create_weights()
        self.weights = network_weights

        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value

    def _calc_z_mean_and_sigma(self,x):
        with tf.name_scope("VI_encoder_r1"):
 
            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(x, shape=[-1, self.n_input, 1, self.n_channels])
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    bnorm_name = 'bn_conv_' + str(i)
                    conv_pre1 = tf.add(tf.nn.conv2d(conv_pool, self.weights['VI_encoder_r1'][weight_name + '1'],strides=1,padding='SAME'),self.weights['VI_encoder_r1'][bias_name + '1'])
                    conv_post = self.nonlinearity(conv_pre1)
                    conv_batchnorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([1,1,self.n_filters[i]], dtype=tf.float32)),tf.Variable(tf.ones([1,1,self.n_filters[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                    conv_pool = tf.nn.max_pool2d(conv_batchnorm,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')

                fc = tf.reshape(conv_pool, [-1, int(self.n_input*self.n_filters[-1]/(np.prod(self.maxpool)))])

            else:
                fc = tf.reshape(x,[-1,self.n_input*self.n_channels])
           
            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                bnorm_name = 'bn_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1'][weight_name]), self.weights['VI_encoder_r1'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_batchnorm = tf.nn.batch_normalization(hidden_post,tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32)),tf.Variable(tf.ones([self.n_weights[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                hidden_dropout = tf.layers.dropout(hidden_batchnorm,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_loc']), self.weights['VI_encoder_r1']['b_loc'])
            scale_diag = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_scale_diag']), self.weights['VI_encoder_r1']['b_scale_diag'])
            #scale_tri = tf.add(tf.matmul(hidden_dropout, self.weights['VICI_encoder']['w_scale_tri']), self.weights['VICI_encoder']['b_scale_tri'])
            weight = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_weight']), self.weights['VI_encoder_r1']['b_weight']) 

            # make +ve definite covariance matrix
            #A_rs = tf.reshape(scale_tri,(-1,int((self.n_output*(self.n_output+1))/2)))
            #A_posdiag = tf.linalg.set_diag(A_rs,SMALL_CONSTANT + tf.exp(tf.linalg.diag_part(A_rs)))
            #A = tfp.math.fill_triangular(A_posdiag, upper=False, name=None)
            #cov = tf.linalg.matmul(A, A, transpose_a=False, transpose_b=True)

            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale_diag', scale_diag)
            #tf.summary.histogram('scale_tri', scale_tri)
            tf.summary.histogram('weight', weight)
            return tf.reshape(loc,(-1,self.n_modes,self.n_output)), tf.reshape(scale_diag,(-1,self.n_modes,self.n_output)), tf.reshape(weight,(-1,self.n_modes))    
            #return tf.reshape(loc,(-1,self.n_modes,self.n_output)), tf.reshape(A,(-1,self.n_modes,self.n_output,self.n_output)), tf.reshape(weight,(-1,self.n_modes))

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VI_ENC_r1"):            
            all_weights['VI_encoder_r1'] = collections.OrderedDict()

            if self.n_conv is not None:
                dummy = self.n_channels
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VI_encoder_r1'][weight_name + '1'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_encoder_r1'][bias_name + '1'] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    #all_weights['VICI_encoder'][weight_name + '2'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size*2, dummy*self.n_filters),[self.filter_size*2, dummy, self.n_filters]), dtype=tf.float32)
                    #all_weights['VICI_encoder'][bias_name + '2'] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    #all_weights['VICI_encoder'][weight_name + '3'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size*4, dummy*self.n_filters),[self.filter_size*4, dummy, self.n_filters]), dtype=tf.float32)
                    #all_weights['VICI_encoder'][bias_name + '3'] = tf.Variable(tf.zeros([self.n_filters], dtype=tf.float32))
                    tf.summary.histogram(weight_name + '1', all_weights['VI_encoder_r1'][weight_name + '1'])
                    tf.summary.histogram(bias_name + '1', all_weights['VI_encoder_r1'][bias_name + '1'])
                    #tf.summary.histogram(weight_name + '2', all_weights['VICI_encoder'][weight_name + '2'])
                    #tf.summary.histogram(bias_name + '2', all_weights['VICI_encoder'][bias_name + '2'])
                    #tf.summary.histogram(weight_name + '3', all_weights['VICI_encoder'][weight_name + '3'])
                    #tf.summary.histogram(bias_name + '3', all_weights['VICI_encoder'][bias_name + '3'])
                    dummy = self.n_filters[i]

                fc_input_size = int(self.n_input*self.n_filters[-1]/(np.prod(self.maxpool)))
            else:
                fc_input_size = self.n_input*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VI_encoder_r1'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_encoder_r1'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VI_encoder_r1'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_encoder_r1'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VI_encoder_r1']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VI_encoder_r1']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_encoder_r1']['b_loc'])
            all_weights['VI_encoder_r1']['w_scale_diag'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_scale_diag'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VI_encoder_r1']['w_scale_diag'])
            tf.summary.histogram('b_scale', all_weights['VI_encoder_r1']['b_scale_diag'])
            #all_weights['VICI_encoder']['w_scale_tri'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], int((self.n_output*(self.n_output+1)*self.n_modes)/2)),dtype=tf.float32)
            #all_weights['VICI_encoder']['b_scale_tri'] = tf.Variable(tf.zeros([int((self.n_output*(self.n_output+1)*self.n_modes)/2)], dtype=tf.float32), dtype=tf.float32)
            #tf.summary.histogram('w_scale_tri', all_weights['VICI_encoder']['w_scale_tri'])
            #tf.summary.histogram('b_scale_tri', all_weights['VICI_encoder']['b_scale_tri'])
            all_weights['VI_encoder_r1']['w_weight'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_weight'] = tf.Variable(tf.zeros([self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_weight', all_weights['VI_encoder_r1']['w_weight'])
            tf.summary.histogram('b_weight', all_weights['VI_encoder_r1']['b_weight'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
