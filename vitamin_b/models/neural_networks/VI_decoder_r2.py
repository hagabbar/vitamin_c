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

    def __init__(self, name, wrap_mask, nowrap_mask, m1_mask, m2_mask, sky_mask, n_input1=4, n_input2=256, n_output=3, n_channels=3, n_weights=2048, drate=0.2, n_filters=8, filter_size=8, maxpool=4):
        
        self.n_input1 = n_input1                    # actually the output size
        self.n_input2 = n_input2                    # actually the output size
        self.n_output = n_output                    # the input data size
        self.n_channels = n_channels                # the number of channels/detectors
        self.n_weights = n_weights                  # the number of weights were layer
        self.n_hlayers = len(n_weights)             # the number of fully connected layers
        self.n_conv = len(n_filters)                # the number of convolutional layers
        self.n_filters = n_filters                  # the number of filters in each conv layer
        self.filter_size = filter_size              # the filter sizes in each conv layer
        self.maxpool = maxpool                      # the max pooling sizes in each conv layer
        self.name = name                            # the name of the network
        self.drate = drate                          # dropout rate
        self.wrap_mask = wrap_mask                  # mask identifying wrapped indices
        self.nowrap_mask = nowrap_mask              # mask identifying non-wrapped indices
        self.m1_mask = m1_mask                      # the mask identifying the m1 parameter
        self.m2_mask = m2_mask                      # the mask identifying the m2 parameter
        self.sky_mask = sky_mask                    # the mask identifying the sky (RA,dec) parameters
        self.nonlinear_loc_nowrap = tf.sigmoid      # activation for non-wrapped location params
        self.nonlinear_loc_wrap = tf.sigmoid        # activation for wrapped location params
        self.nonlinear_loc_m1 = tf.sigmoid          # activation for mass params
        self.nonlinear_loc_m2 = tf.sigmoid          # activation for mass params
        self.nonlinear_loc_sky = tf.identity        # activation for sky params
        self.nonlinear_scale_nowrap = tf.identity   # activation for non-wrapped scale params
        self.nonlinear_scale_wrap = tf.nn.relu      # activation for wrapped scale params
        self.nonlinear_scale_m1 = tf.nn.relu        # activation for mass params
        self.nonlinear_scale_m2 = tf.nn.relu        # activation for mass params  
        self.nonlinear_scale_sky = tf.nn.relu       # activation for sky params
        self.nonlinearity = tf.nn.relu              # activation between hidden layers

        network_weights = self._create_weights()
        self.weights = network_weights

    def calc_reconstruction(self, z, y):
        with tf.name_scope("VI_decoder_r2"):

            # Reshape input to a 3D tensor - single channel
            if self.n_conv is not None:
                conv_pool = tf.reshape(y, shape=[-1, self.n_input2, 1, self.n_channels])
                for i in range(self.n_conv):            
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    bnorm_name = 'bn_conv_' + str(i)
                    conv_pre1 = tf.add(tf.nn.conv2d(conv_pool, self.weights['VI_decoder_r2'][weight_name + '1'],strides=1,padding='SAME'),self.weights['VI_decoder_r2'][bias_name + '1'])
                    conv_post = self.nonlinearity(conv_pre1)
                    conv_batchnorm = tf.nn.batch_normalization(conv_post,tf.Variable(tf.zeros([1,1,self.n_filters[i]], dtype=tf.float32)),tf.Variable(tf.ones([1,1,self.n_filters[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                    conv_pool = tf.nn.max_pool2d(conv_batchnorm,ksize=[self.maxpool[i],1],strides=[self.maxpool[i],1],padding='SAME')

                fc = tf.concat([z,tf.reshape(conv_pool, [-1, int(self.n_input2*self.n_filters[-1]/(np.prod(self.maxpool)))])],axis=1)            

            else:
                fc = tf.concat([z,y],axis=1)

            hidden_dropout = fc
            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                bnorm_name = 'bn_hidden' + str(i)
                hidden_pre = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2'][weight_name]), self.weights['VI_decoder_r2'][bias_name])
                hidden_post = self.nonlinearity(hidden_pre)
                hidden_batchnorm = tf.nn.batch_normalization(hidden_post,tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32)),tf.Variable(tf.ones([self.n_weights[i]], dtype=tf.float32)),None,None,1e-6,name=bnorm_name)
                hidden_dropout = tf.layers.dropout(hidden_batchnorm,rate=self.drate)
            loc_all = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2']['w_loc']), self.weights['VI_decoder_r2']['b_loc'])
            scale_all = tf.add(tf.matmul(hidden_dropout, self.weights['VI_decoder_r2']['w_scale']), self.weights['VI_decoder_r2']['b_scale'])

            # split up the output into non-wrapped and wrapped params and apply appropriate activation
            loc_nowrap = self.nonlinear_loc_nowrap(tf.boolean_mask(loc_all,self.nowrap_mask + [False],axis=1))   # add an extra null element to the mask
            scale_nowrap = self.nonlinear_scale_nowrap(tf.boolean_mask(scale_all,self.nowrap_mask[:-1],axis=1))  # ignore last element because scale_all is 1 shorter
            loc_m1 = self.nonlinear_loc_m1(tf.boolean_mask(loc_all,self.m1_mask + [False],axis=1))             # add an extra null element to the mask
            scale_m1 = -1.0*self.nonlinear_scale_m1(tf.boolean_mask(scale_all,self.m1_mask[:-1],axis=1))      # ignore last element because scale_all is 1 shorter
            loc_m2 = self.nonlinear_loc_m2(tf.boolean_mask(loc_all,self.m2_mask + [False],axis=1))            # add an extra null element to the mask
            scale_m2 = -1.0*self.nonlinear_scale_m2(tf.boolean_mask(scale_all,self.m2_mask[:-1],axis=1))    # ignore last element because scale_all is 1 shorter
            loc_wrap = self.nonlinear_loc_wrap(tf.boolean_mask(loc_all,self.wrap_mask + [False],axis=1))    # add an extra null element to the mask 
            scale_wrap = -1.0*self.nonlinear_scale_wrap(tf.boolean_mask(scale_all,self.wrap_mask[:-1],axis=1))  # ignore last element because scale_all is 1 shorter
            loc_sky = self.nonlinear_loc_sky(tf.boolean_mask(loc_all,self.sky_mask + [True],axis=1))        # add an extra element to the mask for the 3rd sky parameter
            scale_sky = -1.0*self.nonlinear_scale_sky(tf.boolean_mask(scale_all,self.sky_mask[:-1],axis=1))    # ignore last element because scale_all is 1 shorter
            return loc_nowrap, scale_nowrap, loc_wrap, scale_wrap, loc_m1, scale_m1, loc_m2, scale_m2, loc_sky, scale_sky   

    def _create_weights(self):
        all_weights = collections.OrderedDict()

        # Decoder
        with tf.variable_scope("VI_DEC"):
            all_weights['VI_decoder_r2'] = collections.OrderedDict()
            
            if self.n_conv is not None:
                dummy = self.n_channels
                for i in range(self.n_conv):
                    weight_name = 'w_conv_' + str(i)
                    bias_name = 'b_conv_' + str(i)
                    all_weights['VI_decoder_r2'][weight_name + '1'] = tf.Variable(tf.reshape(vae_utils.xavier_init(self.filter_size[i], dummy*self.n_filters[i]),[self.filter_size[i], 1, dummy, self.n_filters[i]]), dtype=tf.float32)
                    all_weights['VI_decoder_r2'][bias_name + '1'] = tf.Variable(tf.zeros([self.n_filters[i]], dtype=tf.float32))
                    tf.summary.histogram(weight_name + '1', all_weights['VI_decoder_r2'][weight_name + '1'])
                    tf.summary.histogram(bias_name + '1', all_weights['VI_decoder_r2'][bias_name + '1'])
                    dummy = self.n_filters[i]

                fc_input_size = self.n_input1 + int(self.n_input2*self.n_filters[-1]/(np.prod(self.maxpool)))
            else:
                fc_input_size = self.n_input1 + self.n_input2*self.n_channels

            for i in range(self.n_hlayers):
                weight_name = 'w_hidden_' + str(i)
                bias_name = 'b_hidden' + str(i)
                all_weights['VI_decoder_r2'][weight_name] = tf.Variable(vae_utils.xavier_init(fc_input_size, self.n_weights[i]), dtype=tf.float32)
                all_weights['VI_decoder_r2'][bias_name] = tf.Variable(tf.zeros([self.n_weights[i]], dtype=tf.float32))
                tf.summary.histogram(weight_name, all_weights['VI_decoder_r2'][weight_name])
                tf.summary.histogram(bias_name, all_weights['VI_decoder_r2'][bias_name])
                fc_input_size = self.n_weights[i]
            all_weights['VI_decoder_r2']['w_loc'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output+1),dtype=tf.float32)  # +1 for extra sky param
            all_weights['VI_decoder_r2']['b_loc'] = tf.Variable(tf.zeros([self.n_output+1], dtype=tf.float32), dtype=tf.float32) # +1 for extra sky param
            tf.summary.histogram('w_loc', all_weights['VI_decoder_r2']['w_loc'])
            tf.summary.histogram('b_loc', all_weights['VI_decoder_r2']['b_loc'])
            all_weights['VI_decoder_r2']['w_scale'] = tf.Variable(vae_utils.xavier_init(self.n_weights[-1], self.n_output-1),dtype=tf.float32) # # -1 for common concentration par in VMF dist
            all_weights['VI_decoder_r2']['b_scale'] = tf.Variable(tf.zeros([self.n_output-1], dtype=tf.float32), dtype=tf.float32)  # -1 for common concentration par in VMF dist
            tf.summary.histogram('w_scale', all_weights['VI_decoder_r2']['w_scale'])
            tf.summary.histogram('b_scale', all_weights['VI_decoder_r2']['b_scale'])
            
            all_weights['prior_param'] = collections.OrderedDict()
        
        return all_weights
