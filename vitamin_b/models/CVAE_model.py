################################################################################################################
#
# --Variational Inference for gravitational wave parameter estimation--
# 
# This model takes as input measured signals and infers target images/objects.
#
################################################################################################################

import numpy as np
import time
import os, sys,io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import corner
import h5py

from .neural_networks import VI_decoder_r2
from .neural_networks import VI_encoder_r1
from .neural_networks import VI_encoder_q
from .neural_networks import batch_manager
try:
    from .. import gen_benchmark_pe
except:
    import gen_benchmark_pe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfd = tfp.distributions
SMALL_CONSTANT = 1e-12 # necessary to prevent the division by zero in many operations 
GAUSS_RANGE = 10.0     # Actual range of truncated gaussian when the ramp is 0

# NORMALISE DATASET FUNCTION
def tf_normalise_dataset(xp):
    
    Xs = tf.shape(xp)
    
    l2norm = tf.sqrt(tf.reduce_sum(tf.multiply(xp,xp),1))
    l2normr = tf.reshape(l2norm,[Xs[0],1])
    x_data = tf.divide(xp,l2normr)
  
    # comment this if you want to use normalise 
    x_data = xp 
    return x_data

# MULTIMODAL UPDATE: NORMALISE DATASET TO THE SUM FUNCTION
def tf_normalise_sum_dataset(xp):
    
    Xs = tf.shape(xp)
    
    log_norm = tf.math.reduce_logsumexp(xp,1)
    log_norm = tf.reshape(log_norm,[Xs[0],1])
    x_data = tf.add(xp,-log_norm)    

    return x_data

def get_wrap_index(params):

    # identify the indices of wrapped and non-wrapped parameters - clunky code
    wrap_mask, nowrap_mask = [], []
    idx_wrap, idx_nowrap = [], []
    
    # loop over inference params
    for i,p in enumerate(params['inf_pars']):

        # loop over wrapped params 
        flag = False
        for q in params['wrap_pars']:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag
        
        # record the true/false value for this inference param
        if flag==True:
            wrap_mask.append(True)
            nowrap_mask.append(False)
            idx_wrap.append(i)
        elif flag==False:
            wrap_mask.append(False)
            nowrap_mask.append(True)
            idx_nowrap.append(i)
     
    idx_mask = idx_nowrap + idx_wrap
    return wrap_mask, nowrap_mask, idx_mask

def load_chunk(input_dir,inf_pars,params,bounds,fixed_vals,load_condor=False):

    # load generated samples back in
    train_files = []
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])

    snrs = []
    for filename in filenames:
        try:
            train_files.append(filename)
        except OSError:
            print('Could not load requested file')
            continue

    train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/1000.0)]
    np.random.shuffle(train_files)
    train_files = np.array(train_files)[train_files_idx]
    for filename in train_files: 
            print('... Loading file -> ' + filename)
            data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
                  'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
                  'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
            data['x_data'].append(data_temp['x_data'])
            data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
            data['rand_pars'] = data_temp['rand_pars']


    # extract the prior bounds
    bounds = {}
    for k in data_temp['rand_pars']:
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        bounds[par_max] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_max][...].item()
        bounds[par_min] = h5py.File(dataLocations[0]+'/'+filename, 'r')[par_min][...].item()
        if par_min == 'psi_min':
            bounds[par_max] = np.pi
            bounds[par_min] = 0.0
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)


    # normalise the data parameters
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        if par_min == 'psi_min':
            data['x_data'][:,i] = np.remainder(data['x_data'][:,i],np.pi)
        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
    x_data = data['x_data']
    y_data = data['y_data_noisefree']

    # extract inference parameters
    idx = []
    for k in inf_pars:
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    x_data = x_data[:,idx]

    
    # reshape arrays for multi-detector
    y_data_train = y_data
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])

    # reshape y data into channels last format for convolutional approach
    if params['n_filters_r1'] != None:
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(fixed_vals['det'])))

        for i in range(y_data_train.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    return x_data, y_data_train

def get_param_index(all_pars,pars):

    # identify the indices of wrapped and non-wrapped parameters - clunky code
    mask = []
    idx = []
    
    # loop over inference params
    for i,p in enumerate(all_pars):

        # loop over wrapped params 
        flag = False
        for q in pars:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag
        
        # record the true/false value for this inference param
        if flag==True:
            mask.append(True)
            idx.append(i)
        elif flag==False:
            mask.append(False)
     
    return mask, idx, np.sum(mask)

def run(params, y_data_test, siz_x_data, y_normscale, load_dir):

    multi_modal = True

    # USEFUL SIZES
    xsh1 = siz_x_data
    if params['by_channel'] == True:
        ysh0 = np.shape(y_data_test)[0]
        ysh1 = np.shape(y_data_test)[1]
    else:
        ysh0 = np.shape(y_data_test)[1]
        ysh1 = np.shape(y_data_test)[2]
    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_hlayers_r1 = len(params['n_weights_r1'])
    n_hlayers_r2 = len(params['n_weights_r2'])
    n_hlayers_q = len(params['n_weights_q'])
    n_conv_r1 = len(params['n_filters_r1']) if params['n_filters_r1'] != None else None
    n_conv_r2 = len(params['n_filters_r2']) if params['n_filters_r2'] != None else None
    n_conv_q = len(params['n_filters_q'])   if params['n_filters_q'] != None else None
    n_filters_r1 = params['n_filters_r1']
    n_filters_r2 = params['n_filters_r2']
    n_filters_q = params['n_filters_q']
    filter_size_r1 = params['filter_size_r1']
    filter_size_r2 = params['filter_size_r2']
    filter_size_q = params['filter_size_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = ysh1
    ysh_conv_r2 = ysh1
    ysh_conv_q = ysh1
    drate = params['drate']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    if n_filters_r1 != None:
        if params['by_channel'] == True:
            num_det = np.shape(y_data_test)[2]
        else:
            num_det = ysh0
    else:
        num_det = None

    # identify the indices of different sets of physical parameters
    vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
    idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)
    masses_len = m1_len + m2_len

   
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-12

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")

        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh1, n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2)
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh1, n_input2=params['ndata'], n_output=z_dimension, 
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, 
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q)

        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)


        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))


        # DRAW FROM r1(z|y)
        r1_zy_samp = bimix_gauss.sample()


        # GET r2(x|z,y) from r1(z|y) samples
        reconstruction_xzy = r2_xzy.calc_reconstruction(r1_zy_samp,y_ph)

        # ugly but needed for now
        # extract the means and variances of the physical parameter distributions
        r2_xzy_mean_gauss = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_gauss = reconstruction_xzy[1]
        r2_xzy_mean_vonmise = reconstruction_xzy[2]
        r2_xzy_log_sig_sq_vonmise = reconstruction_xzy[3]
        r2_xzy_mean_m1 = reconstruction_xzy[4]
        r2_xzy_log_sig_sq_m1 = reconstruction_xzy[5]
        r2_xzy_mean_m2 = reconstruction_xzy[6]
        r2_xzy_log_sig_sq_m2 = reconstruction_xzy[7]
        r2_xzy_mean_sky = reconstruction_xzy[8]
        r2_xzy_log_sig_sq_sky = reconstruction_xzy[9]

        # draw from r2(x|z,y) - the masses
        temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the m1 variance
        temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)     # the m2 variance
        joint = tfd.JointDistributionSequential([
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),0,1,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0),  # m1
            lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),0,b0,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0)],    # m2
            validate_args=True)
        r2_xzy_samp_masses = tf.transpose(tf.reshape(joint.sample(),[2,-1]))  # sample from the m1.m2 space

        # draw from r2(x|z,y) - the truncated gaussian 
        temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
        @tf.function    # make this s a tensorflow function
        def truncnorm(idx,output):    # we set up a function that adds the log-likelihoods and also increments the counter
            loc = tf.slice(r2_xzy_mean_gauss,[0,idx],[-1,1])            # take each specific parameter mean using slice
            std = tf.sqrt(tf.slice(temp_var_r2_gauss,[0,idx],[-1,1]))   # take each specific parameter std using slice
            tn = tfd.TruncatedNormal(loc,std,0.0,1.0)                   # define the truncated Gaussian distribution
            return [idx+1, tf.concat([output,tf.reshape(tn.sample(),[bs_ph,1])],axis=1)] # return the updated index and new samples concattenated to the input 
        # we do the loop until we've hit all the truncated gaussian parameters - i starts at 0 and the samples starts with a set of zeros that we cut out later
        idx = tf.constant(0)              # initialise counter
        nsamp = params['n_samples']       # define the number of samples (MUST be a normal int NOT tensor so can't use bs_ph)
        output = tf.zeros([nsamp,1],dtype=tf.float32)    # initialise the output (we cut this first set of zeros out later
        condition = lambda i,output: i<gauss_len         # define the while loop stopping condition
        _,r2_xzy_samp_gauss = tf.while_loop(condition, truncnorm, loop_vars=[idx,output],shape_invariants=[idx.get_shape(), tf.TensorShape([nsamp,None])])
        r2_xzy_samp_gauss = tf.slice(tf.reshape(r2_xzy_samp_gauss,[-1,gauss_len+1]),[0,1],[-1,-1])   # cut out the actual samples - delete the initial vector of zeros

        # draw from r2(x|z,y) - the vonmises part
        temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
        con = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance
        von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*(r2_xzy_mean_vonmise-0.5), concentration=con)
        r2_xzy_samp_vonmise = tf.reshape(von_mises.sample()/(2.0*np.pi) + 0.5,[-1,vonmise_len])   # sample from the von mises distribution and shift and scale from -pi-pi to 0-1
        
        # draw from r2(x|z,y) - the von mises Fisher 
        temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
        con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
        von_mises_fisher = tfp.distributions.VonMisesFisher(
                          mean_direction=tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[bs_ph,3]),axis=1),
                          concentration=con)   # define p_vm(2*pi*mu,con=1/sig^2)
        xyz = tf.reshape(von_mises_fisher.sample(),[bs_ph,3])          # sample the distribution
        samp_ra = tf.math.floormod(tf.atan2(tf.slice(xyz,[0,1],[-1,1]),tf.slice(xyz,[0,0],[-1,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
        samp_dec = (tf.asin(tf.slice(xyz,[0,2],[-1,1])) + 0.5*np.pi)/np.pi                       # convert to the rescaled 0->1 dec from the unit vector
        r2_xzy_samp_sky = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[bs_ph,2])             # group the sky samples

        # combine the samples
        r2_xzy_samp = tf.concat([r2_xzy_samp_gauss,r2_xzy_samp_vonmise,r2_xzy_samp_masses,r2_xzy_samp_sky],axis=1)
        r2_xzy_samp = tf.gather(r2_xzy_samp,tf.constant(idx_mask),axis=1)

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VI")]

        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(var_list_VICI)
        saver_VICI.restore(session,load_dir)

    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = params['n_samples'] # number of samples to save per reconstruction 

    y_data_test_exp = np.tile(y_data_test,(ns,1))/y_normscale
    y_data_test_exp = y_data_test_exp.reshape(-1,params['ndata'],num_det)
    run_startt = time.time()
    xs, mode_weights = session.run([r2_xzy_samp,r1_weight],feed_dict={bs_ph:ns,y_ph:y_data_test_exp})
    run_endt = time.time()

#    run_startt = time.time()
#    xs, mode_weights = session.run([r2_xzy_samp,r1_weight],feed_dict={bs_ph:ns,y_ph:y_data_test_exp})
#    run_endt = time.time()

    return xs, (run_endt - run_startt), mode_weights

def train(params, x_data, y_data, x_data_test, y_data_test, y_data_test_noisefree, y_normscale, save_dir, truth_test, bounds, fixed_vals, posterior_truth_test,snrs_test=None):    

    # if True, do multi-modal
    multi_modal = True

    # USEFUL SIZES
    xsh = np.shape(x_data)
    ysh = np.shape(y_data)[1]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_hlayers_r1 = len(params['n_weights_r1'])
    n_hlayers_r2 = len(params['n_weights_r2'])
    n_hlayers_q = len(params['n_weights_q'])
    n_conv_r1 = len(params['n_filters_r1']) if params['n_filters_r1'] != None else None
    n_conv_r2 = len(params['n_filters_r2']) if params['n_filters_r2'] != None else None
    n_conv_q = len(params['n_filters_q'])   if params['n_filters_q'] != None else None
    n_filters_r1 = params['n_filters_r1']
    n_filters_r2 = params['n_filters_r2']
    n_filters_q = params['n_filters_q']
    filter_size_r1 = params['filter_size_r1']
    filter_size_r2 = params['filter_size_r2']
    filter_size_q = params['filter_size_q']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    batch_norm = params['batch_norm']
    ysh_conv_r1 = int(ysh)
    ysh_conv_r2 = int(ysh)
    ysh_conv_q = int(ysh)
    drate = params['drate']
    ramp_start = params['ramp_start']
    ramp_end = params['ramp_end']
    num_det = len(fixed_vals['det'])

    # identify the indices of different sets of physical parameters
    vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
    idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # PLACE HOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph") # params placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")
        ramp = tf.placeholder(dtype=tf.float32)    # the ramp to slowly increase the KL contribution

        # LOAD VICI NEURAL NETWORKS
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, 
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1)
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh[1], n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, 
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh[1], n_input2=params['ndata'], n_output=z_dimension, 
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, 
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q) 
        tf.set_random_seed(np.random.randint(0,10))

        # reduce the y data size
        y_conv = y_ph

        # GET r1(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)

        
        # define the r1(z|y) mixture model
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))


        # DRAW FROM r1(z|y) - given the Gaussian parameters generate z samples
        r1_zy_samp = bimix_gauss.sample()        
        
        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq = q_zxy._calc_z_mean_and_sigma(x_ph,y_conv)

        # DRAW FROM q(z|x,y)
        temp_var_q = SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)
        mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=q_zxy_mean,
                          scale_diag=tf.sqrt(temp_var_q))
        q_zxy_samp = mvn_q.sample()  
       
        # GET r2(x|z,y)
        eps = tf.random.normal([bs_ph, params['ndata'], num_det], 0, 1., dtype=tf.float32)
        y_ph_ramp = tf.add(tf.multiply(ramp,y_conv), tf.multiply((1.0-ramp), eps))
        reconstruction_xzy = r2_xzy.calc_reconstruction(q_zxy_samp,y_ph_ramp)

        # ugly but required for now - unpack the r2 output params
        r2_xzy_mean_gauss = reconstruction_xzy[0]           # truncated gaussian mean
        r2_xzy_log_sig_sq_gauss = reconstruction_xzy[1]     # truncated gaussian log var
        r2_xzy_mean_vonmise = reconstruction_xzy[2]         # vonmises means
        r2_xzy_log_sig_sq_vonmise = reconstruction_xzy[3]   # vonmises log var
        r2_xzy_mean_m1 = reconstruction_xzy[4]              # m1 mean
        r2_xzy_log_sig_sq_m1 = reconstruction_xzy[5]        # m1 var
        r2_xzy_mean_m2 = reconstruction_xzy[6]              # m2 mean (m2 will be conditional on m1)
        r2_xzy_log_sig_sq_m2 = reconstruction_xzy[7]        # m2 log var (m2 will be conditional on m1)
        r2_xzy_mean_sky = reconstruction_xzy[8]             # sky mean unit vector (3D)
        r2_xzy_log_sig_sq_sky = reconstruction_xzy[9]       # sky log var (1D)

        # COST FROM RECONSTRUCTION - the masses
        # this sets up a joint distribution on m1 and m2 with m2 being conditional on m1
        # the ramp eveolves the truncation boundaries from far away to 0->1 for m1 and 0->m1 for m2
        if m1_len>0 and m2_len>0:
            temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the safe r2 variance
            temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)
            joint = tfd.JointDistributionSequential([    # shrink the truncation with the ramp
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0),reinterpreted_batch_ndims=0),  # m1
                lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + ramp*b0),reinterpreted_batch_ndims=0)],    # m2
            )
            reconstr_loss_masses = joint.log_prob((tf.boolean_mask(x_ph,m1_mask,axis=1),tf.boolean_mask(x_ph,m2_mask,axis=1)))

        # COST FROM RECONSTRUCTION - Truncated Gaussian parts
        # this sets up a loop over uncorreltaed truncated Gaussians 
        # the ramp evolves the boundaries from far away to 0->1 
        if gauss_len>0:
            temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
            gauss_x = tf.boolean_mask(x_ph,gauss_mask,axis=1)
            @tf.function
            def truncnorm(i,lp):    # we set up a function that adds the log-likelihoods and also increments the counter
                loc = tf.slice(r2_xzy_mean_gauss,[0,i],[-1,1])
                std = tf.sqrt(tf.slice(temp_var_r2_gauss,[0,i],[-1,1]))
                pos = tf.slice(gauss_x,[0,i],[-1,1])  
                tn = tfd.TruncatedNormal(loc,std,-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)   # shrink the truncation with the ramp
                return [i+1, lp + tn.log_prob(pos)]
            # we do the loop until we've hit all the truncated gaussian parameters - i starts at 0 and the logprob starts at 0 
            _,reconstr_loss_gauss = tf.while_loop(lambda i,reconstr_loss_gauss: i<gauss_len, truncnorm, [0,tf.zeros([bs_ph],dtype=tf.dtypes.float32)])

        # COST FROM RECONSTRUCTION - Von Mises parts for single parameters that wrap over 2pi
        if vonmise_len>0:
            temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance - convert to concentration
            von_mises = tfp.distributions.VonMises(
                          loc=2.0*np.pi*(tf.reshape(r2_xzy_mean_vonmise,[-1,vonmise_len])-0.5),   # remap 0>1 mean onto -pi->pi range
                          concentration=con)
            reconstr_loss_vonmise = von_mises.log_prob(2.0*np.pi*(tf.reshape(tf.boolean_mask(x_ph,vonmise_mask,axis=1),[-1,vonmise_len]) - 0.5))   # 2pi is the von mises input range
            
            reconstr_loss_vonmise = reconstr_loss_vonmise[:,0] + reconstr_loss_vonmise[:,1]

            # computing Gaussian likelihood for von mises parameters to be faded away with the ramp
            gauss_vonmises = tfp.distributions.MultivariateNormalDiag(
                         loc=r2_xzy_mean_vonmise,
                         scale_diag=tf.sqrt(temp_var_r2_vonmise))
            reconstr_loss_gauss_vonmise = gauss_vonmises.log_prob(tf.boolean_mask(x_ph,vonmise_mask,axis=1))        
            reconstr_loss_vonmise = ramp*reconstr_loss_vonmise + (1.0-ramp)*reconstr_loss_gauss_vonmise    # start with a Gaussian model and fade in the true vonmises
        else:
            reconstr_loss_vonmise = 0.0

        # COST FROM RECONSTRUCTION - Von Mises Fisher (sky) parts
        if sky_len>0:
            temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
            loc_xyz = tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[-1,3]),axis=1)    # take the 3 output mean params from r2 and normalse so they are a unit vector
            von_mises_fisher = tfp.distributions.VonMisesFisher(
                          mean_direction=loc_xyz,
                          concentration=con)
            ra_sky = 2.0*np.pi*tf.reshape(tf.boolean_mask(x_ph,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
            dec_sky = np.pi*(tf.reshape(tf.boolean_mask(x_ph,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
            xyz_unit = tf.reshape(tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1),[-1,3])   # construct the true parameter unit vector
            reconstr_loss_sky = von_mises_fisher.log_prob(tf.math.l2_normalize(xyz_unit,axis=1))   # normalise it for safety (should already be normalised) and compute the logprob

            # computing Gaussian likelihood for von mises Fisher (sky) parameters to be faded away with the ramp
            mean_ra = tf.math.floormod(tf.atan2(tf.slice(loc_xyz,[0,1],[-1,1]),tf.slice(loc_xyz,[0,0],[-1,1])),2.0*np.pi)/(2.0*np.pi)    # convert the unit vector to scaled 0->1 RA 
            mean_dec = (tf.asin(tf.slice(loc_xyz,[0,2],[-1,1])) + 0.5*np.pi)/np.pi        # convert the unit vector to scaled 0->1 dec
            mean_sky = tf.reshape(tf.concat([mean_ra,mean_dec],axis=1),[bs_ph,2])        # package up the scaled RA and dec 
            gauss_sky = tfp.distributions.MultivariateNormalDiag(
                         loc=mean_sky,
                         scale_diag=tf.concat([tf.sqrt(temp_var_r2_sky),tf.sqrt(temp_var_r2_sky)],axis=1))   # use the same 1D concentration parameter for both RA and dec dimensions
            reconstr_loss_gauss_sky = gauss_sky.log_prob(tf.boolean_mask(x_ph,sky_mask,axis=1))     # compute the logprob at the true sky location
            reconstr_loss_sky = ramp*reconstr_loss_sky + (1.0-ramp)*reconstr_loss_gauss_sky   # start with a Gaussian model and fade in the true vonmises Fisher

        cost_R = -1.0*tf.reduce_mean(reconstr_loss_gauss + reconstr_loss_vonmise + reconstr_loss_masses + reconstr_loss_sky)
        r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_gauss,r2_xzy_mean_vonmise,r2_xzy_mean_m1,r2_xzy_mean_m2,r2_xzy_mean_sky],axis=1),tf.constant(idx_mask),axis=1)      # put the elements back in order
        r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_gauss,r2_xzy_log_sig_sq_vonmise,r2_xzy_log_sig_sq_m1,r2_xzy_log_sig_sq_m2,r2_xzy_log_sig_sq_sky],axis=1),tf.constant(idx_mask),axis=1)   # put the elements back in order
        
        log_q_q = mvn_q.log_prob(q_zxy_samp)
        log_r1_q = bimix_gauss.log_prob(q_zxy_samp)   # evaluate the log prob of r1 at the q samples
        KL = tf.reduce_mean(log_q_q - log_r1_q)      # average over batch

        # THE VICI COST FUNCTION
        COST = cost_R + ramp*KL #+ L1_weight_reg)

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables() if var.name.startswith("VI")]
        
        # DEFINE OPTIMISER (using ADAM here)
        optimizer = tf.train.AdamOptimizer(params['initial_training_rate']) 
#        optimizer = tf.train.RMSPropOptimizer(params['initial_training_rate'])
        minimize = optimizer.minimize(COST,var_list = var_list_VICI)
        
        # INITIALISE AND RUN SESSION
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver()

    print()
    print('... Training Inference Model')    
    print()
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    plotdata = []

    load_chunk_it = 1
    for i in range(params['num_iterations']):

        next_indices = indices_generator.next_indices()

        # if load chunks true, load in data by chunks
        if params['load_by_chunks'] == True and i == int(params['load_iteration']*load_chunk_it):
            x_data, y_data = load_chunk(params['train_set_dir'],params['inf_pars'],params,bounds,fixed_vals)
            load_chunk_it += 1

        # Make noise realizations and add to training data
        next_x_data = x_data[next_indices,:]
        if n_conv_r1 != None:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']),len(fixed_vals['det'])))
        else:
            next_y_data = y_data[next_indices,:] + np.random.normal(0,1,size=(params['batch_size'],int(params['ndata']*len(fixed_vals['det']))))
        next_y_data /= y_normscale  # required for fast convergence

        if params['by_channel'] == False:
            next_y_data_new = [] 
            for sig in next_y_data:
                next_y_data_new.append(sig.T)
            next_y_data = np.array(next_y_data_new)
            del next_y_data_new
      
        # restore session if wanted
        if params['resume_training'] == True and i == 0:
            print('... Loading previously trained model from -> ' + save_dir)
            saver.restore(session, save_dir)

        # compute the ramp value
        rmp = 0.0
        if params['ramp'] == True:
            if i>ramp_start:
                rmp = (np.log10(float(i)) - np.log10(ramp_start))/(np.log10(ramp_end) - np.log10(ramp_start))
            if i>ramp_end:
                rmp = 1.0  
        else:
            rmp = 1.0              

        # train the network 
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp}) 
 
        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:

            # get training loss
            cost, kl, AB_batch = session.run([cost_R, KL, r1_weight], feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp})

            # get validation loss on test set
            cost_val, kl_val = session.run([cost_R, KL], feed_dict={bs_ph:y_data_test.shape[0], x_ph:x_data_test, y_ph:y_data_test/y_normscale, ramp:rmp})
            plotdata.append([cost,kl,cost+kl,cost_val,kl_val,cost_val+kl_val])

           
            try:
                # Make loss plot
                plt.figure()
                xvec = params['report_interval']*np.arange(np.array(plotdata).shape[0])
                plt.semilogx(xvec,np.array(plotdata)[:,0],label='recon',color='blue',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,1],label='KL',color='orange',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,2],label='total',color='green',alpha=0.5)
                plt.semilogx(xvec,np.array(plotdata)[:,3],label='recon_val',color='blue',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,4],label='KL_val',color='orange',linestyle='dotted')
                plt.semilogx(xvec,np.array(plotdata)[:,5],label='total_val',color='green',linestyle='dotted')
                plt.ylim([-25,15])
                plt.xlabel('iteration')
                plt.ylabel('cost')
                plt.legend()
                plt.savefig('%s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving unzoomed cost to -> %s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
                plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving zoomed cost to -> %s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print()
                plt.close('all')
                
            except:
                pass

            if params['print_values']==True:
                print('--------------------------------------------------------------')
                print('Iteration:',i)
                print('Training -ELBO:',cost)
                print('Validation -ELBO:',cost_val)
                print('Training KL Divergence:',kl)
                print('Validation KL Divergence:',kl_val)
                print('Training Total cost:',kl + cost) 
                print('Validation Total cost:',kl_val + cost_val)
                print()

                # terminate training if vanishing gradient
                if np.isnan(kl+cost) == True or np.isnan(kl_val+cost_val) == True or kl+cost > int(1e5):
                    print()
                    print('Network is returning NaN values')
                    print('Terminating network training')
                    print()
                    if params['hyperparam_optim'] == True:
                        save_path = saver.save(session,save_dir)
                        return 5000.0, session, saver, save_dir
                    else:
                        exit()
                try:
                    # Save loss plot data
                    np.savetxt(save_dir.split('/')[0] + '/loss_data.txt', np.array(plotdata))
                except FileNotFoundError as err:
                    pass

        if i % params['save_interval'] == 0 and i > 0:

            if params['hyperparam_optim'] == False:
                # Save model 
                save_path = saver.save(session,save_dir)
            else:
                pass


        # stop hyperparam optim training it and return KL divergence as figure of merit
        if params['hyperparam_optim'] == True and i == params['hyperparam_optim_stop']:
            save_path = saver.save(session,save_dir)

            return np.array(plotdata)[-1,2], session, saver, save_dir

        if i % params['plot_interval'] == 0 and i>0:

            n_mode_weight_copy = 100 # must be a multiple of 50
            # just run the network on the test data
            for j in range(params['r']*params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                if params['n_filters_r1'] != None:
                    XS, dt, _  = run(params, y_data_test[j].reshape([1,y_data_test.shape[1],y_data_test.shape[2]]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                else:
                    XS, dt, _  = run(params, y_data_test[j].reshape([1,-1]), np.shape(x_data_test)[1],
                                                 y_normscale, 
                                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
                print()
                print('... Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            
                print()

                # Make corner plots
                # Get corner parnames to use in plotting labels
                parnames = []
                for k_idx,k in enumerate(params['rand_pars']):
                    if np.isin(k, params['inf_pars']):
                        parnames.append(params['cornercorner_parnames'][k_idx])

                defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

                figure = corner.corner(posterior_truth_test[j], **defaults_kwargs,labels=parnames,
                       color='tab:blue',truths=x_data_test[j,:],
                       show_titles=True)
                # compute weights, otherwise the 1d histograms will be different scales, could remove this
                corner.corner(XS,**defaults_kwargs,labels=parnames,
                       color='tab:red',
                       fill_contours=True,
                       show_titles=True, fig=figure)


                plt.savefig('%s/corner_plot_%s_%d-%d.png' % (params['plot_dir'],params['run_label'],i,j))
                print()
                print('... Saved corner plot iteration %d to -> %s/corner_plot_%s_%d-%d.png' % (i,params['plot_dir'],params['run_label'],i,j))
                plt.savefig('%s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                print('... Saved latest corner plot to -> %s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close('all')
                print('... Made corner plot %d' % j)
                print()

    return            

