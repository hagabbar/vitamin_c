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

    def __init__(self, name, n_input=256, n_output=4, n_channels=3, n_weights=2048, n_modes=2, n_hlayers=2, drate=0.2, n_filters=8, filter_size=8, maxpool=4, n_conv=2, strides=1, dilations=1, batch_norm=True, twod_conv=False, parallel_conv=False):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_channels = n_channels
        self.n_weights = n_weights
        self.n_filters = n_filters
        self.strides = strides
        self.parallel_conv = parallel_conv
        self.dilations = dilations
        self.filter_size = filter_size
        self.n_hlayers = len(n_weights)
        self.n_conv = len(n_filters)
        self.n_modes = n_modes
        self.drate = drate
        self.maxpool = maxpool
        self.batch_norm = batch_norm
        self.nonlinearity = tf.nn.relu
        self.nonlinearity_mean = tf.clip_by_value
        self.twod_conv = twod_conv
        if self.twod_conv:
            self.conv_out_size_t = n_input - 2*int(self.filter_size[0]/2)
        else:
            self.conv_out_size_t = n_input # + 1
        for i in range(self.n_conv):
            self.conv_out_size_t = np.ceil(self.conv_out_size_t/strides[i])
            self.conv_out_size_t = np.ceil(self.conv_out_size_t/maxpool[i])       
        if self.parallel_conv:
            self.conv_out_size_t = int(self.conv_out_size_t*n_filters[-1])*self.n_channels
        else:
            self.conv_out_size_t = int(self.conv_out_size_t*n_filters[-1])
        if self.twod_conv:
            self.conv_out_size_t *= n_channels
        #self.conv_out_size_t = n_channels*int(self.conv_out_size_t*n_filters[-1]) 

        network_weights = self._create_weights()
        self.weights = network_weights


    def _calc_z_mean_and_sigma(self,x, training=True):
        with tf.name_scope("VI_encoder_r1"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv>0:
                if self.twod_conv:
                    conv_pool_t = tf.reshape(x, shape=[-1, self.n_input, self.n_channels,1])
                    conv_pool_t = tf.concat([tf.reshape(conv_pool_t[:,:,-1,:],[-1,self.n_input,1,1]),conv_pool_t,tf.reshape(conv_pool_t[:,:,0,:],[-1,self.n_input,1,1])],axis=2)
                    conv_padding = 'VALID'
                else:
                    conv_pool_t = tf.reshape(x, shape=[-1, self.n_input, 1, self.n_channels])
                    if self.parallel_conv:
                        conv_pool_t = tf.concat(tf.split(conv_pool_t,num_or_size_splits=self.n_channels,axis=3),axis=0)
                    conv_padding = 'SAME'
                #conv_pool_t0 = tf.reshape(conv_pool_t[:,:,0], shape=[-1,self.n_input,1])
                #conv_pool_t1 = tf.reshape(conv_pool_t[:,:,1], shape=[-1,self.n_input,1])
                #conv_pool_t2 = tf.reshape(conv_pool_t[:,:,2], shape=[-1,self.n_input,1])

                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    conv_pre_t = tf.add(tf.nn.conv2d(conv_pool_t, self.weights['VI_encoder_r1'][weight_name+'t'],strides=[self.strides[i],1],dilations=[self.dilations[i],1],padding=conv_padding),self.weights['VI_encoder_r1'][bias_name+'t'])
                    conv_post_t = self.nonlinearity(conv_pre_t)
                    conv_pool_t = tf.nn.max_pool2d(conv_post_t,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')
                    conv_padding = 'SAME'
                    #conv_pre_t0 = tf.add(tf.nn.conv1d(conv_pool_t0, self.weights['VI_encoder_r1'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_encoder_r1'][bias_name+'t']) 
                    #conv_pre_t1 = tf.add(tf.nn.conv1d(conv_pool_t1, self.weights['VI_encoder_r1'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_encoder_r1'][bias_name+'t'])
                    #conv_pre_t2 = tf.add(tf.nn.conv1d(conv_pool_t2, self.weights['VI_encoder_r1'][weight_name+'t'],stride=self.strides[i],dilations=self.dilations[i],padding='SAME'),self.weights['VI_encoder_r1'][bias_name+'t'])
                    #conv_post_t0 = self.nonlinearity(conv_pre_t0)
                    #conv_post_t1 = self.nonlinearity(conv_pre_t1)
                    #conv_post_t2 = self.nonlinearity(conv_pre_t2)
                    #conv_pool_t0 = tf.nn.max_pool1d(conv_post_t0,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')
                    #conv_pool_t1 = tf.nn.max_pool1d(conv_post_t1,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')
                    #conv_pool_t2 = tf.nn.max_pool1d(conv_post_t2,ksize=self.maxpool[i],strides=self.maxpool[i],padding='SAME')

                #conv_pool_t = tf.concat([conv_pool_t0,conv_pool_t1,conv_pool_t2],axis=-1)
                if self.parallel_conv:
                    conv_pool_t = tf.concat(tf.split(conv_pool_t,num_or_size_splits=self.n_channels,axis=0),axis=1)
                fc = tf.reshape(conv_pool_t, [-1, self.conv_out_size_t])

            else:
                fc = tf.reshape(x,[-1,self.n_input*self.n_channels])
            """
        with tf.name_scope("VI_encoder_r1"):
            # ResNet Approach
            conv_pool_t = tf.reshape(x, shape=[-1, self.n_input, 1, self.n_channels])
            weight_name = 'w_conv_' + str(0)
            bias_name = 'b_conv_' + str(0)
            conv_pool_t = tf.add(tf.nn.conv2d(conv_pool_t, self.weights['VI_encoder_r1'][weight_name+'t'],strides=[1,1],dilations=[1,1],padding='SAME'),self.weights['VI_encoder_r1'][bias_name+'t'])
            res_n = [2, 2, 2, 2]
            for block_idx,block in enumerate(res_n):
                for i in range(block):
                    conv_pool_t = vae_utils.resblock(conv_pool_t, channels=self.n_channels, is_training=training, downsample=False, scope='resblock%d_%d' % (block_idx,i))
            fc = tf.reshape(conv_pool_t, [-1, self.conv_out_size_t])          
            """ 

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                bn_name = 'VI_bn_hidden_' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1'][weight_name]), self.weights['VI_encoder_r1'][bias_name])
                if self.batch_norm:
                    hidden_batchnorm = tf.layers.batch_normalization(hidden_pre,axis=-1,center=False,scale=False,
                                   beta_initializer=tf.zeros_initializer(),
                                   gamma_initializer=tf.ones_initializer(),
                                   moving_mean_initializer=tf.zeros_initializer(),
                                   moving_variance_initializer=tf.ones_initializer(),   
                                   trainable=True,epsilon=1e-3,training=training)
                    hidden_post = self.nonlinearity(hidden_batchnorm)
                else:
                    hidden_post = self.nonlinearity(hidden_pre)
                hidden_dropout = tf.layers.dropout(hidden_post,rate=self.drate)
            loc = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_loc']), self.weights['VI_encoder_r1']['b_loc'])
            scale_diag = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_scale_diag']), self.weights['VI_encoder_r1']['b_scale_diag'])
            weight = tf.add(tf.matmul(hidden_dropout, self.weights['VI_encoder_r1']['w_weight']), self.weights['VI_encoder_r1']['b_weight']) 


            tf.summary.histogram('loc', loc)
            tf.summary.histogram('scale_diag', scale_diag)
            tf.summary.histogram('weight', weight)
            return tf.reshape(loc,(-1,self.n_modes,self.n_output)), tf.reshape(scale_diag,(-1,self.n_modes,self.n_output)), tf.reshape(weight,(-1,self.n_modes))    

    def _create_weights(self):
        all_weights = collections.OrderedDict()
        with tf.variable_scope("VI_ENC_r1"):            
            all_weights['VI_encoder_r1'] = collections.OrderedDict()

            if self.n_conv>0:
                if self.parallel_conv:
                    dummy_t = 1
                else:
                    dummy_t = self.n_channels
                #dummy_t = 1
                for i in range(self.n_conv):
                    #i = 0
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    if self.twod_conv:
                        all_weights['VI_encoder_r1'][weight_name+'t'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], self.n_channels*self.n_filters[i]),[self.filter_size[i], self.n_channels, 1, self.n_filters[i]]), dtype=tf.float32)
                    else:
                        all_weights['VI_encoder_r1'][weight_name+'t'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy_t*self.n_filters[i]),[self.filter_size[i], 1, dummy_t, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_encoder_r1'][bias_name+'t'] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name+'t', all_weights['VI_encoder_r1'][weight_name+'t'])
                    tf.summary.histogram(bias_name+'t', all_weights['VI_encoder_r1'][bias_name+'t'])
                    dummy_t = self.n_filters[i]
                
                fc_input_size = self.conv_out_size_t
            
            else:
                fc_input_size = self.n_input*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden_' + str(i)
                all_weights['VI_encoder_r1'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_encoder_r1'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VI_encoder_r1'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_encoder_r1'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VI_encoder_r1']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            #all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            #all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(tf.random.uniform([self.n_output*self.n_modes],minval=-2,maxval=2,dtype=tf.float32),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(tf.random.normal([self.n_output*self.n_modes], 0, 1., dtype=tf.float32),dtype=tf.float32)
            #all_weights['VI_encoder_r1']['b_loc'] = tf.Variable(2.0*tf.cast(tf.random.uniform([self.n_output*self.n_modes],minval=-2,maxval=3,dtype=tf.int32),dtype=tf.float32),dtype=tf.float32)
            tf.summary.histogram('w_loc', all_weights['VI_encoder_r1']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_encoder_r1']['b_loc'])
            all_weights['VI_encoder_r1']['w_scale_diag'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output*self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_scale_diag'] = tf.Variable(tf.zeros([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            #all_weights['VI_encoder_r1']['b_scale_diag'] = tf.Variable(-2.0*tf.ones([self.n_output*self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_scale', all_weights['VI_encoder_r1']['w_scale_diag'])
            tf.summary.histogram('b_scale', all_weights['VI_encoder_r1']['b_scale_diag'])
            all_weights['VI_encoder_r1']['w_weight'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_modes),dtype=tf.float32)
            all_weights['VI_encoder_r1']['b_weight'] = tf.Variable(tf.zeros([self.n_modes], dtype=tf.float32), dtype=tf.float32)
            tf.summary.histogram('w_weight', all_weights['VI_encoder_r1']['w_weight'])
            tf.summary.histogram('b_weight', all_weights['VI_encoder_r1']['b_weight'])

            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
