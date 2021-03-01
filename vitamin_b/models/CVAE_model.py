################################################################################################################
#
# --Variational Inference for gravitational wave parameter estimation--
# 
# Our model takes as input measured signals and infers target images/objects.
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
from lal import GreenwichMeanSiderealTime
from universal_divergence import estimate

from .neural_networks import VI_decoder_r2
from .neural_networks import VI_encoder_r1
from .neural_networks import VI_encoder_q
from .neural_networks import batch_manager
try:
    from .. import gen_benchmark_pe
    from .neural_networks.vae_utils import convert_ra_to_hour_angle, convert_hour_angle_to_ra
except Exception as e:
    import gen_benchmark_pe
    from models.neural_networks.vae_utils import convert_ra_to_hour_angle, convert_hour_angle_to_ra

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tfd = tfp.distributions
SMALL_CONSTANT = 1e-6 # necessary to prevent the division by zero in many operations 
GAUSS_RANGE = 10.0     # Actual range of truncated gaussian when the ramp is 0

def load_chunk(input_dir,inf_pars,params,bounds,fixed_vals,load_condor=False):
    """ Function to load more training/testing data

    Parameters
    ----------
    input_dir: str
        Directory where training or testing files are stored
    inf_pars: list
        list of parameters to infer when training ML model
    params: dict
        Dictionary containing parameter values of run
    bounds: dict
        Dictionary containing the allowed bounds of source GW parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    load_condor: bool
        if True, load test samples rather than training samples

    Returns
    -------
    x_data: array_like
        data source parameter values
    y_data_train: array_like
        data time series 
    """

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

    train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/float(params['tset_split']))]
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


    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)

    if load_condor == False:
        # convert ra to hour angle if needed
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, params['rand_pars'])

    # normalise the data parameters
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
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
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(params['det'])))

        for i in range(y_data_train.shape[0]):
            for j in range(len(params['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    return x_data, y_data_train

def get_param_index(all_pars,pars):
    """ Get the list index of requested source parameter types 
  
    Parameters
    ----------
    all_pars: list
        list of infered parameters
    pars: list
        parameters to get index of

    Returns
    -------
    mask: list
        boolean array of parameter indices
    idx: list
        array of parameter indices
    np.sum(mask): float
        total number of parameter indices
    """
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

def run(params, x_data_test, y_data_test, load_dir, wrmp=None):
    """ Function to run a pre-trained tensorflow neural network
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of the run
    y_data_test: array_like
        test sample time series
    siz_x_data: float
        Number of source parameters to infer
    load_dir: str
        location of pre-trained model

    Returns
    -------
    xs: array_like
        predicted posterior samples
    (run_endt-run_startt): float
        total time to make predicted posterior samples
    mode_weights: array_like
        learned Gaussian Mixture Model modal weights
    """

    # USEFUL SIZES
    y_normscale = params['y_normscale']
    #xsh_train = len(params['rand_pars'])
    xsh_inf = len(params['inf_pars'])
    ysh0 = np.shape(y_data_test)[0]
    ysh1 = np.shape(y_data_test)[1]
    z_dimension = params['z_dimension']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_modes_q = params['n_modes_q']
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
    parallel_conv = params['parallel_conv']
    batch_norm = params['batch_norm']
    twod_conv = params['twod_conv']
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
    conv_dilations_r1 = params['conv_dilations_r1']
    conv_dilations_q = params['conv_dilations_q']
    conv_dilations_r2 = params['conv_dilations_r2']
    if n_filters_r1 != None:
        num_det = np.shape(y_data_test)[2]
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
    #inf_mask, inf_idx, _ = get_param_index(params['rand_pars'],params['inf_pars'])
   
    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():
        tf.set_random_seed(np.random.randint(0,10))
        SMALL_CONSTANT = 1e-12

        # PLACEHOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh_inf], name="x_ph")
        wramp = tf.placeholder(dtype=tf.float32, shape=[n_modes])

        # LOAD VICI NEURAL NETWORKS
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh_inf, n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, strides=conv_strides_r2, dilations=conv_dilations_r2,
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv)
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, strides=conv_strides_r1, dilations=conv_dilations_r1,
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh_inf, n_input2=params['ndata'], n_output=z_dimension, n_modes=n_modes_q,
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, strides=conv_strides_q, dilations=conv_dilations_q,
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv)

        # reduce the y data size
        y_conv = y_ph
        #x_inf = tf.boolean_mask(x_ph,inf_mask,axis=1))

        # GET r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv,training=False)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)

        # TESTING - cyclic latent space dimensions - HARDCODED FOR Z-DIM=7
        #bimix_gauss = tfd.Mixture(
        #    cat=tfd.Categorical(logits=r1_weight),
        #        components=[
        #            tfp.distributions.VonMises(
        #                  loc=2.0*np.pi*(tf.sigmoid(tf.slice(r1_loc,[0,0,0],[bs_ph,n_modes,1]))-0.5),   # remap 0>1 mean onto -pi->pi range
        #                  concentration=tf.math.reciprocal(SMALL_CONSTANT + tf.exp(-1.0*tf.nn.relu(tf.slice(r1_scale,[0,0,0],[bs_ph,n_modes,1]))))),
        #            tfp.distributions.VonMises(
        #                  loc=2.0*np.pi*(tf.sigmoid(tf.slice(r1_loc,[0,0,1],[bs_ph,n_modes,1]))-0.5),   # remap 0>1 mean onto -pi->pi range
        #                  concentration=tf.math.reciprocal(SMALL_CONSTANT + tf.exp(-1.0*tf.nn.relu(tf.slice(r1_scale,[0,0,1],[bs_ph,n_modes,1]))))),
        #            tfd.Normal(loc=tf.slice(r1_loc,[0,0,vonmise_len],[bs_ph,n_modes,1]), scale=tf.sqrt(tf.slice(temp_var_r1,[0,0,vonmise_len],[bs_ph,n_modes,1]))),
        #            tfd.Normal(loc=tf.slice(r1_loc,[0,0,vonmise_len+1],[bs_ph,n_modes,1]), scale=tf.sqrt(tf.slice(temp_var_r1,[0,0,vonmise_len+1],[bs_ph,n_modes,1]))),
        #            tfd.Normal(loc=tf.slice(r1_loc,[0,0,vonmise_len+2],[bs_ph,n_modes,1]), scale=tf.sqrt(tf.slice(temp_var_r1,[0,0,vonmise_len+2],[bs_ph,n_modes,1]))),
        #            tfd.Normal(loc=tf.slice(r1_loc,[0,0,vonmise_len+3],[bs_ph,n_modes,1]), scale=tf.sqrt(tf.slice(temp_var_r1,[0,0,vonmise_len+3],[bs_ph,n_modes,1]))),
        #            tfd.Normal(loc=tf.slice(r1_loc,[0,0,vonmise_len+4],[bs_ph,n_modes,1]), scale=tf.sqrt(tf.slice(temp_var_r1,[0,0,vonmise_len+4],[bs_ph,n_modes,1]))),
        #])

        # define the r1(z|y) mixture model
        r1_probs = r1_weight + tf.tile(tf.reshape(tf.log(wramp),[1,-1]),(bs_ph,1))  # apply weight ramps
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_probs),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))

        # DRAW FROM r1(z|y)
        r1_zy_samp = bimix_gauss.sample()

        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq, q_weight = q_zxy._calc_z_mean_and_sigma(x_ph,y_conv,training=False)

        # DRAW FROM q(z|x,y)
        temp_var_q = SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)
        mvn_q = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=q_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=q_zxy_mean,
                          scale_diag=tf.sqrt(temp_var_q)))
        #mvn_q = tfp.distributions.MultivariateNormalDiag(
        #                  loc=q_zxy_mean,
        #                  scale_diag=tf.sqrt(temp_var_q))
        q_zxy_samp = mvn_q.sample()

        # GET r2(x|z,y) from r1(z|y) samples
        reconstruction_xzy = r2_xzy.calc_reconstruction(r1_zy_samp,y_conv,training=False)

        # ugly but needed for now
        # extract the means and variances of the physical parameter distribution
        r2_xzy_mean_gauss = reconstruction_xzy[0]
        r2_xzy_log_sig_sq_gauss = reconstruction_xzy[1]
        r2_xzy_mean_vonmise = reconstruction_xzy[2]
        r2_xzy_log_sig_sq_vonmise = reconstruction_xzy[3]
        r2_xzy_mean_m1 = reconstruction_xzy[4]
        r2_xzy_log_sig_sq_m1 = reconstruction_xzy[5]
        r2_xzy_mean_m2 = reconstruction_xzy[6]
        r2_xzy_log_sig_sq_m2 = reconstruction_xzy[7]
        r2_xzy_mean_sky = reconstruction_xzy[8]
        r2_xzy_log_sig_sq_sky = tf.slice(reconstruction_xzy[9],[0,0],[bs_ph,1])      # sky log var (only take first dimension)

        # draw from r2(x|z,y) - the masses
        if m1_len>0 and m2_len>0:
            temp_var_r2_m1 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m1)     # the m1 variance
            temp_var_r2_m2 = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_m2)     # the m2 variance
            joint = tfd.JointDistributionSequential([
                       tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m1,tf.sqrt(temp_var_r2_m1),0,1,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0),  # m1
                lambda b0: tfd.Independent(tfd.TruncatedNormal(r2_xzy_mean_m2,tf.sqrt(temp_var_r2_m2),0,b0,validate_args=True,allow_nan_stats=True),reinterpreted_batch_ndims=0)],    # m2
                validate_args=True)
            r2_xzy_samp_masses = tf.transpose(tf.reshape(joint.sample(),[2,bs_ph]))  # sample from the m1.m2 space
        else:
            r2_xzy_samp_masses = tf.zeros([bs_ph,0], tf.float32)

        if gauss_len>0:
            # draw from r2(x|z,y) - the truncated gaussian 
            temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
            tn = tfd.TruncatedNormal(r2_xzy_mean_gauss,tf.sqrt(temp_var_r2_gauss),0.0,1.0)   # shrink the truncation with the ramp
            r2_xzy_samp_gauss = tn.sample()
        else:
            r2_xzy_samp_gauss = tf.zeros([bs_ph,0], tf.float32)

        if vonmise_len>0:
            # draw from r2(x|z,y) - the vonmises part
            temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance
            von_mises = tfp.distributions.VonMises(loc=2.0*np.pi*(r2_xzy_mean_vonmise-0.5), concentration=con)
            r2_xzy_samp_vonmise = tf.reshape(von_mises.sample()/(2.0*np.pi) + 0.5,[-1,vonmise_len])   # sample from the von mises distribution and shift and scale from -pi-pi to 0-1
        else:
            r2_xzy_samp_vonmise = tf.zeros([bs_ph,0], tf.float32)

        if sky_len>0:
            # draw from r2(x|z,y) - the von mises Fisher 
            temp_var_r2_sky = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_sky)
            con = tf.reshape(tf.math.reciprocal(temp_var_r2_sky),[bs_ph])   # modelling wrapped scale output as log variance - only 1 concentration parameter for all sky
            von_mises_fisher = tfp.distributions.VonMisesFisher(
                          mean_direction=tf.math.l2_normalize(tf.reshape(r2_xzy_mean_sky,[bs_ph,3]),axis=1),
                          concentration=con)   # define p_vm(2*pi*mu,con=1/sig^2)
            xyz = tf.reshape(von_mises_fisher.sample(),[bs_ph,3])          # sample the distribution
            samp_ra = tf.math.floormod(tf.atan2(tf.slice(xyz,[0,1],[bs_ph,1]),tf.slice(xyz,[0,0],[bs_ph,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
            samp_dec = (tf.asin(tf.slice(xyz,[0,2],[bs_ph,1])) + 0.5*np.pi)/np.pi                       # convert to the rescaled 0->1 dec from the unit vector
            r2_xzy_samp_sky = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[bs_ph,2])             # group the sky samples
        else:
            r2_xzy_samp_sky = tf.zeros([bs_ph,0], tf.float32)

        # combine the samples
        r2_xzy_samp = tf.concat([r2_xzy_samp_gauss,r2_xzy_samp_vonmise,r2_xzy_samp_masses,r2_xzy_samp_sky],axis=1)
        r2_xzy_samp = tf.gather(r2_xzy_samp,tf.constant(idx_mask),axis=1)

        # INITIALISE AND RUN SESSION
        init = tf.initialize_all_variables()
        session.run(init)
        saver_VICI = tf.train.Saver(tf.global_variables())
        saver_VICI.restore(session,load_dir)

    # ESTIMATE TEST SET RECONSTRUCTION PER-PIXEL APPROXIMATE MARGINAL LIKELIHOOD and draw from q(x|y)
    ns = params['n_samples'] # number of samples to save per reconstruction 

    y_data_test_exp = np.tile(y_data_test,(ns,1))
    x_data_test_exp = np.tile(x_data_test,(ns,1))
    y_data_test_exp = y_data_test_exp.reshape(-1,params['ndata'],num_det)/y_normscale
    run_startt = time.time()
    xs, mode_weights, zsamp, qzsamp, wgt, qwgt, zloc = session.run([r2_xzy_samp,r1_weight,r1_zy_samp,q_zxy_samp,r1_probs,q_weight,r1_loc],feed_dict={bs_ph:ns,y_ph:y_data_test_exp,x_ph:x_data_test_exp,wramp:wrmp})
    run_endt = time.time()

    return xs, (run_endt - run_startt), mode_weights, zsamp, qzsamp, wgt, qwgt, zloc

def train(params, x_data, y_data, x_data_val, y_data_val, x_data_test, y_data_test, y_data_test_noisefree, save_dir, truth_test, bounds, fixed_vals, posterior_truth_test,snrs_test=None):    
    """ Main function to train tensorflow model
    
    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of run
    x_data: array_like
        array containing training source parameter values
    x_data_test: array_like
        array containing testing source parameter values
    x_data_val: array_like
        array containing validation source parameter values
    x_data_val: array_like
        array containing validation source parameter values
    y_data_test: array_like
        array containing noisy testing time series
    y_data_test_noisefree: array_like
        array containing noisefree testing time series
    save_dir: str
        location to save trained tensorflow model
    truth_test: array_like TODO: this is redundant, must be removed ...
        array containing testing source parameter values
    bounds: dict
        allowed bounds of GW source parameters
    fixed_vals: dict
        fixed values of GW source parameters
    posterior_truth_test: array_like
        posterior from test Bayesian sampler analysis 
    """

    # skyweighting test
    skyweight=1.0

    # USEFUL SIZES
    y_normscale = params['y_normscale']
    xsh = np.shape(x_data)
    ysh = np.shape(y_data)[1]
    valsize = np.shape(x_data_val)[0]
    z_dimension = params['z_dimension']
    bs = params['batch_size']
    n_weights_r1 = params['n_weights_r1']
    n_weights_r2 = params['n_weights_r2']
    n_weights_q = params['n_weights_q']
    n_modes = params['n_modes']
    n_modes_q = params['n_modes_q']
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
    parallel_conv = params['parallel_conv']
    maxpool_r1 = params['maxpool_r1']
    maxpool_r2 = params['maxpool_r2']
    maxpool_q = params['maxpool_q']
    conv_strides_r1 = params['conv_strides_r1']
    conv_strides_r2 = params['conv_strides_r2']
    conv_strides_q = params['conv_strides_q']
    pool_strides_r1 = params['pool_strides_r1']
    pool_strides_r2 = params['pool_strides_r2']
    pool_strides_q = params['pool_strides_q']
    conv_dilations_r1 = params['conv_dilations_r1']
    conv_dilations_q = params['conv_dilations_q']
    conv_dilations_r2 = params['conv_dilations_r2']
    batch_norm = params['batch_norm']
    twod_conv = params['twod_conv']
    drate = params['drate']
    ramp_start = params['ramp_start']
    ramp_end = params['ramp_end']
    num_det = len(params['det'])
    n_KLzsamp = params['n_KLzsamp']
    if params['batch_norm']:
        print('...... applying batchnorm')

    # identify the indices of different sets of physical parameters
    vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
    gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
    sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
    ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
    dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
    m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
    m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
    idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask)

    inf_ol_mask, inf_ol_idx, inf_ol_len = get_param_index(params['inf_pars'],params['inf_pars'])
    bilby_ol_mask, bilby_ol_idx, bilby_ol_len = get_param_index(params['inf_pars'],params['inf_pars'])

    graph = tf.Graph()
    session = tf.Session(graph=graph)
    with graph.as_default():

        # PLACE HOLDERS
        bs_ph = tf.placeholder(dtype=tf.int64, name="bs_ph")                       # batch size placeholder
        x_ph = tf.placeholder(dtype=tf.float32, shape=[None, xsh[1]], name="x_ph") # params placeholder
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, params['ndata'], num_det], name="y_ph")
        ramp = tf.placeholder(dtype=tf.float32)    # the ramp to slowly increase the KL contribution
        wramp = tf.placeholder(dtype=tf.float32, shape=[n_modes]) # r1 weight ramp placeholder
        skyramp = tf.placeholder(dtype=tf.float32) # sky loss ramp placeholder
        extra_lr_decay_factor = tf.placeholder(dtype=tf.float32)    # the ramp to slowly increase the KL contribution

        # LOAD VICI NEURAL NETWORKS
        r1_zy = VI_encoder_r1.VariationalAutoencoder('VI_encoder_r1', n_input=params['ndata'], n_output=z_dimension, n_channels=num_det, n_weights=n_weights_r1,   # generates params for r1(z|y)
                                                    n_modes=n_modes, drate=drate, n_filters=n_filters_r1, strides=conv_strides_r1, dilations=conv_dilations_r1,
                                                    filter_size=filter_size_r1, maxpool=maxpool_r1, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv)
        r2_xzy = VI_decoder_r2.VariationalAutoencoder('VI_decoder_r2', vonmise_mask, gauss_mask, m1_mask, m2_mask, sky_mask, n_input1=z_dimension, 
                                                     n_input2=params['ndata'], n_output=xsh[1], n_channels=num_det, n_weights=n_weights_r2, 
                                                     drate=drate, n_filters=n_filters_r2, strides=conv_strides_r2, dilations=conv_dilations_r2,
                                                     filter_size=filter_size_r2, maxpool=maxpool_r2, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv)
        q_zxy = VI_encoder_q.VariationalAutoencoder('VI_encoder_q', n_input1=xsh[1], n_input2=params['ndata'], n_output=z_dimension, n_modes=n_modes_q,
                                                     n_channels=num_det, n_weights=n_weights_q, drate=drate, strides=conv_strides_q, dilations=conv_dilations_q,
                                                     n_filters=n_filters_q, filter_size=filter_size_q, maxpool=maxpool_q, batch_norm=batch_norm, twod_conv=twod_conv, parallel_conv=parallel_conv) 

        tf.set_random_seed(np.random.randint(0,10))

        # add noise to the data - MUST apply normscale to noise !!!!
        y_conv = y_ph + tf.random.normal([bs_ph,params['ndata'],num_det],0.0,1.0,dtype=tf.float32)/y_normscale

        # GET r1(z|y)
        # run inverse autoencoder to generate mean and logvar of z given y data - these are the parameters for r1(z|y)
        r1_loc, r1_scale, r1_weight = r1_zy._calc_z_mean_and_sigma(y_conv,training=True)
        temp_var_r1 = SMALL_CONSTANT + tf.exp(r1_scale)

        # TESTING - cyclic latent space dimensions
        #bimix_gauss = tfd.MixtureSameFamily(
        #    mixture_distribution=tfd.Categorical(logits=r1_weight),
        #bimix_gauss_test = tfd.JointDistributionSequential([
        #        tfp.distributions.VonMises(
        #                  loc=2.0*np.pi*(tf.sigmoid(tf.squeeze(tf.slice(r1_loc,[0,0,0],[bs_ph,n_modes,1]))-0.5)),   # remap 0>1 mean onto -pi->pi range
        #                  concentration=tf.math.reciprocal(SMALL_CONSTANT + tf.exp(-1.0*tf.nn.relu(tf.squeeze(tf.slice(r1_scale,[0,0,0],[bs_ph,n_modes,1])))))),
        #            tfp.distributions.VonMises(
        #                  loc=2.0*np.pi*(tf.sigmoid(tf.squeeze(tf.slice(r1_loc,[0,0,1],[bs_ph,n_modes,1])))-0.5),   # remap 0>1 mean onto -pi->pi range
        #                  concentration=tf.math.reciprocal(SMALL_CONSTANT + tf.exp(-1.0*tf.nn.relu(tf.squeeze(tf.slice(r1_scale,[0,0,1],[bs_ph,n_modes,1])))))),
        #            tfd.Normal(loc=tf.squeeze(tf.slice(r1_loc,[0,0,vonmise_len],[bs_ph,n_modes,1])), scale=tf.sqrt(tf.squeeze(tf.slice(temp_var_r1,[0,0,vonmise_len],[bs_ph,n_modes,1])))),
        #            tfd.Normal(loc=tf.squeeze(tf.slice(r1_loc,[0,0,vonmise_len+1],[bs_ph,n_modes,1])), scale=tf.sqrt(tf.squeeze(tf.slice(temp_var_r1,[0,0,vonmise_len+1],[bs_ph,n_modes,1])))),
        #            tfd.Normal(loc=tf.squeeze(tf.slice(r1_loc,[0,0,vonmise_len+2],[bs_ph,n_modes,1])), scale=tf.sqrt(tf.squeeze(tf.slice(temp_var_r1,[0,0,vonmise_len+2],[bs_ph,n_modes,1])))),
        #            tfd.Normal(loc=tf.squeeze(tf.slice(r1_loc,[0,0,vonmise_len+3],[bs_ph,n_modes,1])), scale=tf.sqrt(tf.squeeze(tf.slice(temp_var_r1,[0,0,vonmise_len+3],[bs_ph,n_modes,1])))),
        #            tfd.Normal(loc=tf.squeeze(tf.slice(r1_loc,[0,0,vonmise_len+4],[bs_ph,n_modes,1])), scale=tf.sqrt(tf.squeeze(tf.slice(temp_var_r1,[0,0,vonmise_len+4],[bs_ph,n_modes,1])))),
        #    ])
        #)

        # define the r1(z|y) mixture model
        r1_probs = r1_weight + tf.tile(tf.reshape(tf.log(wramp),[1,-1]),(bs_ph,1))  # apply weight ramps
        bimix_gauss = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=r1_probs),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=r1_loc,
                          scale_diag=tf.sqrt(temp_var_r1)))
        
        # GET q(z|x,y)
        q_zxy_mean, q_zxy_log_sig_sq, q_weight = q_zxy._calc_z_mean_and_sigma(x_ph,y_conv,training=True)

        # DRAW FROM q(z|x,y)
        temp_var_q = SMALL_CONSTANT + tf.exp(q_zxy_log_sig_sq)
        mvn_q = tfd.MixtureSameFamily(
                          mixture_distribution=tfd.Categorical(logits=q_weight),
                          components_distribution=tfd.MultivariateNormalDiag(
                          loc=q_zxy_mean,
                          scale_diag=tf.sqrt(temp_var_q)))
        q_zxy_samp = mvn_q.sample()  

        # GET r2(x|z,y)
        #eps = tf.random.normal([bs_ph, params['ndata'], num_det], 0, 1., dtype=tf.float32)
        y_ph_ramp = y_conv #tf.add(tf.multiply(ramp,y_conv), tf.multiply((1.0-ramp), eps))
        reconstruction_xzy = r2_xzy.calc_reconstruction(q_zxy_samp,y_ph_ramp,training=True)

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
        r2_xzy_log_sig_sq_sky = tf.slice(reconstruction_xzy[9],[0,0],[bs_ph,1])      # sky log var (only take first dimension)

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
        else:
            reconstr_loss_masses = 0.0

        # COST FROM RECONSTRUCTION - Truncated Gaussian parts
        # this sets up a loop over uncorreltaed truncated Gaussians 
        # the ramp evolves the boundaries from far away to 0->1 
        if gauss_len>0:
            temp_var_r2_gauss = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_gauss)
            gauss_x = tf.boolean_mask(x_ph,gauss_mask,axis=1)
            tn = tfd.TruncatedNormal(r2_xzy_mean_gauss,tf.sqrt(temp_var_r2_gauss),-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)   # shrink the truncation with the ramp
            reconstr_loss_gauss = tf.reduce_sum(tn.log_prob(gauss_x),axis=1)
        else:
            reconstr_loss_gauss = 0.0

        # COST FROM RECONSTRUCTION - Von Mises parts for single parameters that wrap over 2pi
        if vonmise_len>0:
            temp_var_r2_vonmise = SMALL_CONSTANT + tf.exp(r2_xzy_log_sig_sq_vonmise)
            con_vonmise = tf.reshape(tf.math.reciprocal(temp_var_r2_vonmise),[-1,vonmise_len])   # modelling wrapped scale output as log variance - convert to concentration
            von_mises = tfp.distributions.VonMises(
                          loc=2.0*np.pi*(tf.reshape(r2_xzy_mean_vonmise,[-1,vonmise_len])-0.5),   # remap 0>1 mean onto -pi->pi range
                          concentration=con_vonmise)
            reconstr_loss_vonmise = tf.reduce_sum(von_mises.log_prob(2.0*np.pi*(tf.reshape(tf.boolean_mask(x_ph,vonmise_mask,axis=1),[-1,vonmise_len]) - 0.5)),axis=1)   # 2pi is the von mises input range

            # computing Gaussian likelihood for von mises parameters to be faded away with the ramp
            gauss_vonmises = tfd.TruncatedNormal(r2_xzy_mean_vonmise,
                                                 tf.sqrt(temp_var_r2_vonmise),
                                                 -GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)
            reconstr_loss_gauss_vonmise = tf.reduce_sum(tf.reshape(gauss_vonmises.log_prob(tf.boolean_mask(x_ph,vonmise_mask,axis=1)),[-1,vonmise_len]),axis=1)        
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
            ra_sky = 2*np.pi*tf.reshape(tf.boolean_mask(x_ph,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
            dec_sky = np.pi*(tf.reshape(tf.boolean_mask(x_ph,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
            xyz_unit = tf.reshape(tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1),[-1,3])   # construct the true parameter unit vector
            reconstr_loss_sky = von_mises_fisher.log_prob(tf.math.l2_normalize(xyz_unit,axis=1))   # normalise it for safety (should already be normalised) and compute the logprob

            # computing Gaussian likelihood for von mises Fisher (sky) parameters to be faded away with the ramp
            mean_ra = tf.math.floormod(tf.atan2(tf.slice(loc_xyz,[0,1],[bs_ph,1]),tf.slice(loc_xyz,[0,0],[bs_ph,1])),2.0*np.pi)/(2.0*np.pi)    # convert the unit vector to scaled 0->1 RA 
            mean_dec = (tf.asin(tf.slice(loc_xyz,[0,2],[bs_ph,1])) + 0.5*np.pi)/np.pi        # convert the unit vector to scaled 0->1 dec
            mean_sky = tf.reshape(tf.concat([mean_ra,mean_dec],axis=1),[bs_ph,2])        # package up the scaled RA and dec 
            scale_sky = tf.concat([tf.sqrt(temp_var_r2_sky),tf.sqrt(temp_var_r2_sky)],axis=1)
            gauss_sky = tfd.TruncatedNormal(mean_sky,scale_sky,-GAUSS_RANGE*(1.0-ramp),GAUSS_RANGE*(1.0-ramp) + 1.0)
            reconstr_loss_gauss_sky = tf.reduce_sum(gauss_sky.log_prob(tf.boolean_mask(x_ph,sky_mask,axis=1)),axis=1)
            reconstr_loss_sky = ramp*reconstr_loss_sky + (1.0-ramp)*reconstr_loss_gauss_sky   # start with a Gaussian model and fade in the true vonmises Fisher
        else:
            reconstr_loss_sky = 0.0

        cost_R = -1.0*tf.reduce_mean(reconstr_loss_gauss + reconstr_loss_vonmise + reconstr_loss_masses + skyweight*skyramp*reconstr_loss_sky)
        #r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_gauss,r2_xzy_mean_vonmise,r2_xzy_mean_m1,r2_xzy_mean_m2,r2_xzy_mean_sky],axis=1),tf.constant(idx_mask),axis=1)      # put the elements back in order
        #r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_gauss,r2_xzy_log_sig_sq_vonmise,r2_xzy_log_sig_sq_m1,r2_xzy_log_sig_sq_m2,r2_xzy_log_sig_sq_sky],axis=1),tf.constant(idx_mask),axis=1)   # put the elements back in order
        #r2_xzy_mean = tf.gather(tf.concat([r2_xzy_mean_gauss,r2_xzy_mean_vonmise,r2_xzy_mean_m1,r2_xzy_mean_m2],axis=1),tf.constant(idx_mask),axis=1)      # put the elements back in order
        #r2_xzy_scale = tf.gather(tf.concat([r2_xzy_log_sig_sq_gauss,r2_xzy_log_sig_sq_vonmise,r2_xzy_log_sig_sq_m1,r2_xzy_log_sig_sq_m2],axis=1),tf.constant(idx_mask),axis=1)
       
        # fake larger batch averaging for KL calculation
        extra_q_zxy_samp = mvn_q.sample(n_KLzsamp)
        log_q_q = mvn_q.log_prob(extra_q_zxy_samp)
        log_r1_q = bimix_gauss.log_prob(extra_q_zxy_samp)   # evaluate the log prob of r1 at the q samples
        KL = tf.reduce_mean(log_q_q - log_r1_q)      # average over batch
        
        # THE VICI COST FUNCTION
        COST = cost_R + ramp*KL	

        # VARIABLES LISTS
        var_list_VICI = [var for var in tf.trainable_variables()] # if var.name.startswith("VI")]

        # DEFINE OPTIMISER (using ADAM here)
        global_step = tf.Variable(0, trainable=False)
        if params['extra_lr_decay_factor']: 
            starter_learning_rate = params['initial_training_rate']
            learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                               global_step, 10000, extra_lr_decay_factor, staircase=True)
        else:
            learning_rate = tf.convert_to_tensor(params['initial_training_rate'],dtype=tf.float32)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)                            
        with tf.control_dependencies(update_ops):                                                            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gvs = optimizer.compute_gradients(COST, var_list=var_list_VICI)
            capped_gvs = [(tf.clip_by_value(grad, -100, 100.0), var) for grad, var in gvs]
            minimize = optimizer.apply_gradients(capped_gvs, global_step=global_step)

        # INITIALISE AND RUN SESSION
        init = tf.global_variables_initializer()
        session.run(init)
        saver = tf.train.Saver()

    print('...... Training Inference Model')    
    # START OPTIMISATION OF OELBO
    indices_generator = batch_manager.SequentialIndexer(params['batch_size'], xsh[0])
    val_indices_generator = batch_manager.SequentialIndexer(params['batch_size'], valsize)
    plotdata = []
    kldata = []
    klxaxis = []

    # Convert right ascension to hour angle
    x_data_test_hour_angle = np.copy(x_data_test)
    x_data_test_hour_angle = convert_ra_to_hour_angle(x_data_test_hour_angle, params, params['inf_pars'])


    time_check = 0.0
    load_chunk_it = 1
    for i in range(params['num_iterations']):

        # restore session if wanted
        if params['resume_training'] == True and i == 0:
            print('... Loading previously trained model from -> ' + save_dir)
            saver.restore(session, save_dir)

        # compute the ramp value
        rmp = 0.0
        if params['ramp'] == True:
            ramp_epoch = np.log10(float(i))
            #ramp_duration = (np.log10(ramp_end) - np.log10(ramp_start))/5.0
            if i>ramp_start:
                #rmp = 0.0
                rmp = (ramp_epoch - np.log10(ramp_start))/(np.log10(ramp_end) - np.log10(ramp_start))
                #rmp = 1.0 - np.exp(-(float(i)-ramp_start)/ramp_end)*np.cos(2.0*np.pi*(float(i)-ramp_start)/ramp_end)

                #rmp = np.remainder(ramp_epoch-np.log10(ramp_start),ramp_duration)/ramp_duration
            if i>ramp_end:
                rmp = 1.0
        else:
            rmp = 1.0

        # TESTING r1 weight ramp
        wrmp = np.ones(n_modes)
        #wrmp = np.zeros(n_modes)
        #wrmp[0] = 1.0
        #for s in range(1,n_modes):
        #    wstart = ramp_end + (s-1)*0.25*ramp_end
        #    if i>wstart:
        #       wrmp[s] = min((i-wstart)/(0.25*ramp_end),1.0)

        # TESTING sky ramp
        srmp = 0.0
        if params['ramp'] == True:
            ramp_epoch = np.log10(float(i))
            if i>ramp_end:
                srmp = (ramp_epoch - np.log10(ramp_end))/(np.log10(2.0*ramp_end) - np.log10(ramp_end))
            if i>2*ramp_end:
                srmp = 1.0
        else:
            srmp = 1.0


        next_indices = indices_generator.next_indices()
        # if load chunks true, load in data by chunks
        if params['load_by_chunks'] == True and i == int(params['load_iteration']*load_chunk_it):
            x_data, y_data = load_chunk(params['train_set_dir'],params['inf_pars'],params,bounds,fixed_vals)
            load_chunk_it += 1

        # get next batch and normalise
        next_x_data = x_data[next_indices,:]
        next_y_data = y_data[next_indices,:]
        next_y_data /= y_normscale

        # restore session if wanted
        if params['resume_training'] == True and i == 0:
            print('... Loading previously trained model from -> ' + save_dir)
            saver.restore(session, save_dir)

        # set the lr decay to start after the ramp 
        edf = 1.0
        if params['extra_lr_decay_factor'] and params['ramp'] and i>ramp_end :
            edf = 0.95

        # train the network
        t0 = time.time()
        session.run(minimize, feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp, wramp:wrmp, skyramp:srmp, extra_lr_decay_factor:edf}) 
        time_check += (time.time() - t0)

        # if we are in a report iteration extract cost function values
        if i % params['report_interval'] == 0 and i > 0:

            # get training loss
            cost, kl, AB_batch, lr, test = session.run([cost_R, KL, r1_probs, learning_rate, r1_probs], feed_dict={bs_ph:bs, x_ph:next_x_data, y_ph:next_y_data, ramp:rmp, wramp:wrmp, skyramp:srmp, extra_lr_decay_factor:edf})

            # Make noise realizations and add to training data
            val_next_indices = val_indices_generator.next_indices()
            next_x_data_val = x_data_val[val_next_indices,:]
            next_y_data_val = y_data_val[val_next_indices,:] #.reshape(valsize,int(params['ndata']),len(params['det']))[val_next_indices,:]
            next_y_data_val /= y_normscale

            # Get validation cost
            cost_val, kl_val = session.run([cost_R, KL], feed_dict={bs_ph:bs, x_ph:next_x_data_val, y_ph:next_y_data_val, ramp:rmp, wramp:wrmp, skyramp:srmp, extra_lr_decay_factor:edf})
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
                plt.grid()
                plt.legend()
                plt.savefig('%s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving unzoomed cost to -> %s/latest_%s/cost_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
                plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                print('... Saving zoomed cost to -> %s/latest_%s/cost_zoom_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
                plt.close('all')
                
            except:
                pass

            if params['print_values']==True:
                print('--------------------------------------------------------------',time.asctime())
                print('Iteration:',i)
                print('Training -ELBO:',cost)
                print('Validation -ELBO:',cost_val)
                print('Training KL Divergence:',kl)
                print('Validation KL Divergence:',kl_val)
                print('Training Total cost:',kl + cost) 
                print('Validation Total cost:',kl_val + cost_val)
                print('Learning rate:',lr)
                print('ramp:',rmp)
                print('sky ramp:',srmp)
                print('r1 weight ramp:',wrmp)
                print('train-time:',time_check)
                time_check = 0.0

                print()
              
                # terminate training if vanishing gradient
                if np.isnan(kl+cost) == True:
                    print()
                    print('Network is returning NaN values')
                    print('Terminating network training')
                    print()
                    exit()
                try:
                    # Save loss plot data
                    np.savetxt(save_dir.split('/')[0] + '/loss_data.txt', np.array(plotdata))
                except FileNotFoundError as err:
                    pass

        # Save model every save interval
        if i % params['save_interval'] == 0 and i > 0:
            save_path = saver.save(session,save_dir)

        # generate corner plots every plot interval
        if i % params['plot_interval'] == 0 and i>0:

            # just run the network on the test data
            kltemp = []
            for j in range(params['r']):

                # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                if params['n_filters_r1'] != None:
                    XS, dt, _, zsamp, qzsamp, wgt, qwgt, zloc  = run(params, x_data_test[j], y_data_test[j].reshape([1,y_data_test.shape[1],y_data_test.shape[2]]), save_dir,wrmp=wrmp)
                else:
                    XS, dt, _, zsamp, qzsamp, wgt, qwgt, zloc  = run(params, x_data_test[j], y_data_test[j].reshape([1,-1]),save_dir,wrmp=wrmp)
                print('...... Runtime to generate {} samples = {} sec'.format(params['n_samples'],dt))            

                # plot the r1 weights
                plt.figure()
                norm_wgt = np.exp(wgt[0,:]-np.max(wgt[0,:]))
                norm_wgt /= np.sum(norm_wgt)
                plt.bar(np.arange(params['n_modes']),norm_wgt,width=1.0)
                plt.ylim([0,1])
                plt.xlabel('mode')
                plt.ylabel('mode weight')
                plt.savefig('%s/r1_weight_plot_%s_%d-%d.png' % (params['plot_dir'],params['run_label'],i,j))
                print('...... Saved r1 weight plot iteration %d to -> %s/r1_weight_plot_%s_%d-%d.png' % (i,params['plot_dir'],params['run_label'],i,j))
                plt.savefig('%s/latest_%s/r1_weight_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                print('...... Saved latest r1 weight plot to -> %s/latest_%s/r1_weight_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close('all')

                # plot the q weights
                plt.figure()
                norm_qwgt = np.exp(qwgt[0,:]-np.max(qwgt[0,:]))
                norm_qwgt /= np.sum(norm_qwgt)
                plt.bar(np.arange(params['n_modes_q']),norm_qwgt,width=1.0)
                plt.ylim([0,1])
                plt.xlabel('mode')
                plt.ylabel('mode weight')
                plt.savefig('%s/q_weight_plot_%s_%d-%d.png' % (params['plot_dir'],params['run_label'],i,j))
                print('...... Saved q weight plot iteration %d to -> %s/r1_weight_plot_%s_%d-%d.png' % (i,params['plot_dir'],params['run_label'],i,j))
                plt.savefig('%s/latest_%s/q_weight_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                print('...... Saved latest q weight plot to -> %s/latest_%s/q_weight_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],j))
                plt.close('all')

                # unnormalize and get gps time
                extra = 'test'
                #true_post = np.copy(posterior_truth_test[j])
                #true_x = np.copy(x_data_test[j,:])
                true_post = np.zeros([XS.shape[0],bilby_ol_len])
                true_x = np.zeros(inf_ol_len)
                true_XS = np.zeros([XS.shape[0],inf_ol_len])
                ol_pars = []
                cnt = 0
                for inf_idx,bilby_idx in zip(inf_ol_idx,bilby_ol_idx):
                    inf_par = params['inf_pars'][inf_idx]
                    bilby_par = params['inf_pars'][bilby_idx]
                    print(inf_par,bilby_par)
                    true_XS[:,cnt] = (XS[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                    true_post[:,cnt] = (posterior_truth_test[j,:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
                    true_x[cnt] = (x_data_test[j,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
                    ol_pars.append(inf_par)
                    cnt += 1

                # un-normalise full inference parameters
                full_true_x = np.zeros(len(params['inf_pars']))
                for inf_par_idx,inf_par in enumerate(params['inf_pars']):
                    XS[:,inf_par_idx] = (XS[:,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                    full_true_x[inf_par_idx] = (x_data_test[j,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']

                # convert to RA
                XS = convert_hour_angle_to_ra(XS,params,params['inf_pars'])
                true_XS = convert_hour_angle_to_ra(true_XS,params,ol_pars)
                print(true_x,true_x.shape)
                full_true_x = convert_hour_angle_to_ra(np.reshape(full_true_x,[1,XS.shape[1]]),params,params['inf_pars']).flatten()
                true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten() 
                print(true_x,true_x.shape)

                # compute KL estimate
                idx1 = np.random.randint(0,true_XS.shape[0],1000)
                idx2 = np.random.randint(0,true_post.shape[0],1000)
                print('...... computing the KL divergence')
                run_startt = time.time()
                KLest = estimate(true_XS[idx1,:],true_post[idx2,:])
                run_endt = time.time()
                kltemp.append(KLest)
                print('...... the KL is {}'.format(KLest))
                print('...... KL calculation time = {}'.format(run_endt-run_startt))

                KLbs = []
                run_startt = time.time()
                for k in range(25):
                    idx1 = np.random.randint(0,true_XS.shape[0],100)
                    idx2 = np.random.randint(0,true_post.shape[0],100)
                    KLbs.append(estimate(true_XS[idx1,:],true_post[idx2,:]))
                KLbs_mu = np.mean(KLbs)
                KLbs_sig = np.std(KLbs)
                run_endt = time.time()
                print('...... the boot-strap KL calculation gives {} +/- {}'.format(KLbs_mu,KLbs_sig))
                print('...... KL boot-strap calculation time = {}'.format(run_endt-run_startt))   

                # Make corner plots
                # Get corner parnames to use in plotting labels
                parnames = []
                for k_idx,k in enumerate(params['rand_pars']):
                    if np.isin(k, params['inf_pars']):
                        parnames.append(params['corner_labels'][k])
                ol_parnames = []
                for k_idx,k in enumerate(params['rand_pars']):
                    if np.isin(k, ol_pars):
                        ol_parnames.append(params['corner_labels'][k])

                # define general plotting arguments
                defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

                # 1-d hist kwargs for normalisation
                hist_kwargs_A = dict(density=True,color='tab:blue')
                hist_kwargs_B = dict(density=True,color='tab:red')
 
                #try:
                # make corner plot
                if params['pe_dir']==None:
                    figure = corner.corner(true_XS,**defaults_kwargs,labels=ol_parnames,
                           color='tab:red',
                           fill_contours=True, truths=true_x,
                           show_titles=True, hist_kwargs=hist_kwargs_B)
                else:
                    figure = corner.corner(true_post, **defaults_kwargs,labels=ol_parnames,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_A)
                    corner.corner(true_XS,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True, truths=true_x,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_B)
                plt.annotate('KL = {:.3f} ({:.3f} +/- {:.3f})'.format(KLest,KLbs_mu,KLbs_sig),(0.65,0.85),xycoords='figure fraction')

                # save plot to file and output info
                plt.savefig('%s/corner_plot_%s_%d-%d_%s.png' % (params['plot_dir'],params['run_label'],i,j,extra))
                print('...... Saved corner plot iteration %d to -> %s/corner_plot_%s_%d-%d_%s.png' % (i,params['plot_dir'],params['run_label'],i,j,extra))
                plt.savefig('%s/latest_%s/corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                print('...... Saved latest corner plot to -> %s/latest_%s/corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                plt.close('all')
                print('...... Made corner plot %d' % j)

                figure = corner.corner(XS,**defaults_kwargs, labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=full_true_x,
                           show_titles=True, hist_kwargs=hist_kwargs_B)

                # save plot to file and output info
                plt.savefig('%s/full_corner_plot_%s_%d-%d_%s.png' % (params['plot_dir'],params['run_label'],i,j,extra))
                print('...... Saved full corner plot iteration %d to -> %s/full_corner_plot_%s_%d-%d_%s.png' % (i,params['plot_dir'],params['run_label'],i,j,extra))
                plt.savefig('%s/latest_%s/full_corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                print('...... Saved latest full corner plot to -> %s/latest_%s/full_corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                plt.close('all')
                print('...... Made full corner plot %d' % j)

                # plot zsamples distribution
                try:
                    figure = corner.corner(qzsamp,**defaults_kwargs,
                           color='tab:blue',
                           hist_kwargs=hist_kwargs_A)
                    corner.corner(zsamp,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True,
                           fig=figure, hist_kwargs=hist_kwargs_B)
                    # Extract the axes
                    axes = np.array(figure.axes).reshape((z_dimension, z_dimension))

                    # Loop over the histograms
                    for yi in range(z_dimension):
                        for xi in range(yi):
                            ax = axes[yi, xi]
                            ax.plot(zloc[0,:,xi], zloc[0,:,yi], "sg")
                
                    plt.savefig('%s/z_corner_plot_%s_%d-%d_%s.png' % (params['plot_dir'],params['run_label'],i,j,extra))
                    print('...... Saved z_corner plot iteration %d to -> %s/z_corner_plot_%s_%d-%d_%s.png' % (i,params['plot_dir'],params['run_label'],i,j,extra))
                    plt.savefig('%s/latest_%s/z_corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                    print('...... Saved latest z_corner plot to -> %s/latest_%s/z_corner_plot_%s_%d_%s.png' % (params['plot_dir'],params['run_label'],params['run_label'],j,extra))
                    plt.close('all')
                    print('...... Made z corner plot %d' % j)
                except:
                    pass

            # plot KL data
            kldata.append(kltemp)
            klxaxis.append(i)
            plt.figure()
            for j in range(params['r']): 
                plt.plot(klxaxis,np.array(kldata).reshape(-1,params['r'])[:,j])
            plt.ylim([-0.2,1.0])
            plt.xlabel('iteration')
            plt.ylabel('KL')
            plt.savefig('%s/latest_%s/kl_%s.png' % (params['plot_dir'],params['run_label'],params['run_label']))
            print('...... Saved kl plot %d to -> %s/latest_%s/kl_%s.png' % (i,params['plot_dir'],params['run_label'],params['run_label']))
            plt.close('all')


    return            

