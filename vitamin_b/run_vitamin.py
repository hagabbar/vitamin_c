######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import scipy.io as sio
import h5py
import sys
from sys import exit
import shutil
import bilby
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from time import strftime
import corner
import glob
from matplotlib.lines import Line2D
import pandas as pd
import logging.config
from contextlib import contextmanager
import json

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

try:
    from .models import CVAE_model
    from .gen_benchmark_pe import run
    from . import plotting
    from .plotting import prune_samples
except ModuleNotFoundError:
    from models import CVAE_model
    from gen_benchmark_pe import run
    import plotting
    from plotting import prune_samples

# Check for optional basemap installation
try:
    from mpl_toolkits.basemap import Basemap
    print("module 'basemap' is installed")
except (ModuleNotFoundError, ImportError):
    print("module 'basemap' is not installed")
    print("Skyplotting functionality is automatically disabled.")
    skyplotting_usage = False
else:
    skyplotting_usage = True
    try:
        from .skyplotting import plot_sky
    except:
        from skyplotting import plot_sky

""" Script has 4 main functions:
1.) Generate training data
2.) Generate testing data
3.) Train model
4.) Test model
5.) Generate samples only given model and timeseries
"""

parser = argparse.ArgumentParser(description='VItamin: A user friendly Bayesian inference machine learning library.')
parser.add_argument("--gen_train", default=False, help="generate the training data")
parser.add_argument("--gen_test", default=False, help="generate the testing data")
parser.add_argument("--train", default=False, help="train the network")
parser.add_argument("--resume_training", default=False, help="resume training of network")
parser.add_argument("--test", default=False, help="test the network")
parser.add_argument("--params_file", default=None, type=str, help="dictionary containing parameters of run")
parser.add_argument("--bounds_file", default=None, type=str, help="dictionary containing source parameter bounds")
parser.add_argument("--fixed_vals_file", default=None, type=str, help="dictionary containing source parameter values when fixed")
parser.add_argument("--pretrained_loc", default=None, type=str, help="location of a pretrained network (i.e. .ckpt file)")
parser.add_argument("--test_set_loc", default=None, type=str, help="directory containing test set waveforms")
parser.add_argument("--gen_samples", default=False, help="If True, generate samples only (no plotting)")
parser.add_argument("--num_samples", type=int, default=None, help="number of posterior samples to generate")
parser.add_argument("--use_gpu", default=False, help="if True, use gpu")
args = parser.parse_args()

global params; global bounds; global fixed_vals

# Define default location of the parameters files
params = os.path.join(os.getcwd(), 'params_files', 'params.json')
bounds = os.path.join(os.getcwd(), 'params_files', 'bounds.json')
fixed_vals = os.path.join(os.getcwd(), 'params_files', 'fixed_vals.json')

# Load parameters files
if args.params_file != None:
    params = args.params_file
if args.bounds_file != None:
    bounds = args.bounds_file
if args.fixed_vals_file != None:
    fixed_vals = args.fixed_vals_file

# Ranges over which hyperparameter optimization parameters are allowed to vary
kernel_1 = Integer(low=3, high=12, name='kernel_1')
strides_1 = Integer(low=1, high=2, name='strides_1')
pool_1 = Integer(low=1, high=2, name='pool_1')
kernel_2 = Integer(low=3, high=12, name='kernel_2')
strides_2 = Integer(low=1, high=2, name='strides_2')
pool_2 = Integer(low=1, high=2, name='pool_2')
kernel_3 = Integer(low=3, high=12, name='kernel_3')
strides_3 = Integer(low=1, high=2, name='strides_3')
pool_3 = Integer(low=1, high=2, name='pool_3')
kernel_4 = Integer(low=3, high=12, name='kernel_4')
strides_4 = Integer(low=1, high=2, name='strides_4')
pool_4 = Integer(low=1, high=2, name='pool_4')

z_dimension = Integer(low=7, high=100, name='z_dimension')
n_modes = Integer(low=7, high=12, name='n_modes')
n_filters_1 = Integer(low=32, high=33, name='n_filters_1')
n_filters_2 = Integer(low=32, high=33, name='n_filters_2')
n_filters_3 = Integer(low=32, high=33, name='n_filters_3')
n_filters_4 = Integer(low=32, high=33, name='n_filters_4')
batch_size = Integer(low=511, high=512, name='batch_size')
n_weights_fc_1 = Integer(low=2047, high=2048, name='n_weights_fc_1')
n_weights_fc_2 = Integer(low=2047, high=2048, name='n_weights_fc_2')
n_weights_fc_3 = Integer(low=2047, high=2048, name='n_weights_fc_3')

# putting defined hyperparameter optimization ranges into a list
dimensions = [kernel_1, 
              strides_1,
              pool_1,
              kernel_2, 
              strides_2,
              pool_2,
              kernel_3,
              strides_3,
              pool_3,
              kernel_4,
              strides_4,
              pool_4,
              z_dimension,
              n_modes,
              n_filters_1,
              n_filters_2,
              n_filters_3,
              n_filters_4,
              batch_size,
              n_weights_fc_1,
              n_weights_fc_2,
              n_weights_fc_3]

"""
# list of initial default hyperparameters to use for GP hyperparameter optimization
default_hyperparams = [params['filter_size_r1'][0],
                       params['conv_strides_r1'][0],
                       params['maxpool_r1'][0],
                       params['filter_size_r1'][1],
                       params['conv_strides_r1'][1],
                       params['maxpool_r1'][1],
                       params['filter_size_r1'][2],
                       params['conv_strides_r1'][2],
                       params['maxpool_r1'][2],
                       params['filter_size_r1'][3],
                       params['conv_strides_r1'][3],
                       params['maxpool_r1'][3],
                       params['z_dimension'],
                       params['n_modes'],
                       params['n_filters_r1'][0],
                       params['n_filters_r1'][1],
                       params['n_filters_r1'][2],
                       params['n_filters_r1'][3],
                       params['batch_size'],
                       params['n_weights_r1'][0],
                       params['n_weights_r1'][1],
                       params['n_weights_r1'][2],
                      ]
"""

# dummy value for initial hyperparameter best KL (to be minimized). Doesn't need to be changed.
best_loss = int(1e6)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def load_data(params,bounds,fixed_vals,input_dir,inf_pars,load_condor=False):
    """ Function to load either training or testing data.

    PARAMETERS:
        input_dir:
            Directory where training or testing files are stored
        inf_pars:
            list of parameters to infer when training ML model
        load_condor:
            if True, load test samples generated using a condor cluster

    RETURNS:
        x_data, y_data, y_data_noisy, y_normscale, snrs
        x_data:
            array containing training/testing source parameter values
        y_data:
            array containing training/testing noise-free times series
        y_data_noisy:
            array containing training/testing noisy time series	
        y_normscale:
            value by which to normalize all time series to be between zero and one
        snrs:
            array containing optimal snr values for all training/testing time series

    """

    ########################
    # load generated samples
    ########################
    train_files = []
    
    # Get list of all training/testing files and define dictionary to store values in files
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': []}

    # Sort files from first generated to last generated
    if load_condor == True:
        filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    else:
        filenames = os.listdir(dataLocations[0])

    # Append training/testing filenames to list. Ignore those that can't be loaded
    snrs = []
    for filename in filenames:
        try:
            train_files.append(filename)

        except OSError:
            print('Could not load requested file')
            continue

    # If loading by chunks, randomly shuffle list of training/testing filenames
    if params['load_by_chunks'] == True and load_condor == False:
        train_files_idx = np.arange(len(train_files))[:int(params['load_chunk_size']/1000.0)]
        np.random.shuffle(train_files)
        train_files = np.array(train_files)[train_files_idx]

    # Iterate over all training/testing files and store source parameters, time series and SNR info in dictionary
    for filename in train_files:
        try:
            data_temp={'x_data': h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:],
                  'y_data_noisefree': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:],
                  'y_data_noisy': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:],
                  'rand_pars': h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]}
            data['y_data_noisefree'].append(np.expand_dims(data_temp['y_data_noisefree'], axis=0))
            snrs.append(h5py.File(dataLocations[0]+'/'+filename, 'r')['snrs'][:])
            data['x_data'].append(data_temp['x_data'])
            data['y_data_noisy'].append(np.expand_dims(data_temp['y_data_noisy'], axis=0))
            data['rand_pars'] = data_temp['rand_pars']
        except OSError:
            print('Could not load requested file')
            continue
    snrs = np.array(snrs)

    # Extract the prior bounds from training/testing files
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)
    data['y_data_noisy'] = np.concatenate(np.array(data['y_data_noisy']), axis=0)
    

    # Normalise the source parameters np.remainder(blah,np.pi)
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        if par_min == 'psi_min':
            data['x_data'][:,i]=np.remainder(data['x_data'][:,i],np.pi)
        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
    x_data = data['x_data']
    y_data = data['y_data_noisefree']
    y_data_noisy = data['y_data_noisy']

    # Define time series normalization factor to use on test samples. We consistantly use the same normscale value if loading by chunks
    y_normscale = params['y_normscale']   
 
    # extract inference parameters from all source parameters loaded earlier
    idx = []
    print()
    for k in inf_pars:
        if load_condor == False:
            print('... ' + k + ' will be inferred')
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    x_data = x_data[:,idx]

    return x_data, y_data, y_data_noisy, y_normscale, snrs

@use_named_args(dimensions=dimensions)
def hyperparam_fitness(kernel_1, strides_1, pool_1,
                       kernel_2, strides_2, pool_2,
                       kernel_3, strides_3, pool_3,
                       kernel_4, strides_4, pool_4,
                       z_dimension,n_modes,
                       n_filters_1,n_filters_2,n_filters_3,n_filters_4,
                       batch_size,
                       n_weights_fc_1,n_weights_fc_2,n_weights_fc_3):
    """ Fitness function used in Gaussian Process hyperparameter optimization 
    Returns a value to be minimized (in this case, the total loss of the 
    neural network during training.

    PARAMETERS:
        hyperparameters to be tuned
   
    RETURNS:
        KL divergence (scalar value)

    """

    # set tunable hyper-parameters
    params['filter_size_r1'] = [kernel_1,kernel_2,kernel_3,kernel_4]
    params['filter_size_r2'] = [kernel_1,kernel_2,kernel_3,kernel_4]
    params['filter_size_q'] = [kernel_1,kernel_2,kernel_3,kernel_4]
    params['n_filters_r1'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4]
    params['n_filters_r2'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4]
    params['n_filters_q'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4]

    # number of filters has to be odd for some reason (this ensures that this is the case)
    for filt_idx in range(len(params['n_filters_r1'])):
        if (params['n_filters_r1'][filt_idx] % 3) != 0:

            # keep adding 1 until filter size is divisible by 3
            while (params['n_filters_r1'][filt_idx] % 3) != 0:
                params['n_filters_r1'][filt_idx] += 1
                params['n_filters_r2'][filt_idx] += 1
                params['n_filters_q'][filt_idx] += 1
    params['conv_strides_r1'] = [strides_1,strides_2,strides_3,strides_4]
    params['conv_strides_r2'] = [strides_1,strides_2,strides_3,strides_4] 
    params['conv_strides_q'] = [strides_1,strides_2,strides_3,strides_4] 
    params['maxpool_r1'] = [pool_1,pool_2,pool_3,pool_4]
    params['maxpool_r2'] = [pool_1,pool_2,pool_3,pool_4]
    params['maxpool_q'] = [pool_1,pool_2,pool_3,pool_4]
    params['pool_strides_r1'] = [pool_1,pool_2,pool_3,pool_4]
    params['pool_strides_r2'] = [pool_1,pool_2,pool_3,pool_4]
    params['pool_strides_q'] = [pool_1,pool_2,pool_3,pool_4]
    params['z_dimension'] = z_dimension
    params['n_modes'] = n_modes
    params['batch_size'] = batch_size
    params['n_weights_r1'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]
    params['n_weights_r2'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]
    params['n_weights_q'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3]

    # Print the hyper-parameters.
    print('kernel_1: {}'.format(kernel_1))
    print('strides_1: {}'.format(strides_1))
    print('pool_1: {}'.format(pool_1))
    print('kernel_2: {}'.format(kernel_2))
    print('strides_2: {}'.format(strides_2))
    print('pool_2: {}'.format(pool_2))
    print('kernel_3: {}'.format(kernel_3))
    print('strides_3: {}'.format(strides_3))
    print('pool_3: {}'.format(pool_3))
    print('kernel_4: {}'.format(kernel_4))
    print('strides_4: {}'.format(strides_4))
    print('pool_4: {}'.format(pool_4))
    print('z_dimension: {}'.format(z_dimension))
    print('n_modes: {}'.format(n_modes))
    print('n_filters_1: {}'.format(params['n_filters_r1'][0]))
    print('n_filters_2: {}'.format(params['n_filters_r1'][1]))
    print('n_filters_3: {}'.format(params['n_filters_r1'][2]))
    print('n_filters_4: {}'.format(params['n_filters_r1'][3]))
    print('batch_size: {}'.format(batch_size))
    print('n_weights_r1_1: {}'.format(n_weights_fc_1))
    print('n_weights_r1_2: {}'.format(n_weights_fc_2))
    print('n_weights_r1_3: {}'.format(n_weights_fc_3))
    print()

    start_time = time.time()
    print('start time: {}'.format(strftime('%X %x %Z'))) 
    # Train model with given hyperparameters
    VICI_loss, VICI_session, VICI_saver, VICI_savedir = VICI_inverse_model.train(params, x_data_train, y_data_train,
                             x_data_test, y_data_test, y_data_test_noisefree,
                             y_normscale,
                             "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                             x_data_test, bounds, fixed_vals,
                             XS_all)

    end_time = time.time()
    print('Run time : {} h'.format((end_time-start_time)/3600))

    # Print the loss.
    print()
    print("Total loss: {0:.2}".format(VICI_loss))
    print()

    # update variable outside of this function using global keyword
    global best_loss

    # save model if new best model
    if VICI_loss < best_loss:

        # Save model 
        save_path = VICI_saver.save(VICI_session,VICI_savedir)

        # save hyperparameters
        converged_hyperpar_dict = dict(filter_size = params['filter_size_r1'],
                                       conv_strides = params['conv_strides_r1'],
                                       maxpool = params['maxpool_r1'],
                                       pool_strides = params['pool_strides_r1'],
                                       z_dimension = params['z_dimension'],
                                       n_modes = params['n_modes'],
                                       n_filters = params['n_filters_r1'],
                                       batch_size = params['batch_size'],
                                       n_weights_fc = params['n_weights_r1'],
                                       best_loss = best_loss)
                                       #n_weights_r2 = params['n_weights_r2'],
                                       #n_weights_q = params['n_weights_q'])

        f = open("inverse_model_dir_%s/converged_hyperparams.txt" % params['run_label'],"w")
        f.write( str(converged_hyperpar_dict) )
        f.close()

        # update the best loss
        best_loss = VICI_loss
        
        # Print the loss.
        print()
        print("New best loss: {0:.2}".format(best_loss))
        print()

    # clear tensorflow session
    VICI_session.close()

    return VICI_loss

#######################
# Make training samples
#######################
def gen_train(params=params,bounds=bounds,fixed_vals=fixed_vals):

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make training set directory
    os.system('mkdir -p %s' % params['train_set_dir'])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    print()
    print('... Making training set')
    print()

    # Iterate over number of requested training samples
    for i in range(0,params['tot_dataset_size'],params['tset_split']):

        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': True,
        })
        with suppress_stdout():
            # generate training sample source parameter, waveform and snr
            _, signal_train, signal_train_pars,snrs = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=params['tset_split'],
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          seed=params['training_data_seed']+i,
                                                          label=params['run_label'],
                                                          training=True)
        logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        })
        print("Generated: %s/data_%d-%d.h5py ..." % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size']))

        # store training sample information in hdf5 format
        hf = h5py.File('%s/data_%d-%d.h5py' % (params['train_set_dir'],(i+params['tset_split']),params['tot_dataset_size']), 'w')
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v)
            except:
                pass
        hf.create_dataset('x_data', data=signal_train_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisy', data=signal_train)
        hf.create_dataset('y_data_noisefree', data=signal_train)
        hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
        hf.create_dataset('snrs', data=snrs)
        hf.close()
    return

############################
# Make test samples
############################
def gen_test(params=params,bounds=bounds,fixed_vals=fixed_vals):

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # Make testing set directory
    os.system('mkdir -p %s' % params['test_set_dir'])

    # Make testing samples
    for i in range(params['r']*params['r']):
        temp_noisy, temp_noisefree, temp_pars, temp_snr = run(sampling_frequency=params['ndata']/params['duration'],
                                                      duration=params['duration'],
                                                      N_gen=1,
                                                      ref_geocent_time=params['ref_geocent_time'],
                                                      bounds=bounds,
                                                      fixed_vals=fixed_vals,
                                                      rand_pars=params['rand_pars'],
                                                      inf_pars=params['inf_pars'],
                                                      label=params['bilby_results_label'] + '_' + str(i),
                                                      out_dir=params['pe_dir'],
                                                      samplers=params['samplers'],
                                                      training=False,
                                                      seed=params['testing_data_seed']+i,
                                                      do_pe=params['doPE'])

        signal_test_noisy = temp_noisy
        signal_test_noisefree = temp_noisefree
        signal_test_pars = temp_pars
        signal_test_snr = temp_snr

        print("Generated: %s/data_%s.h5py ..." % (params['test_set_dir'],params['run_label']))

        # Save generated testing samples in h5py format
        hf = h5py.File('%s/data_%d.h5py' % (params['test_set_dir'],i),'w')
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v)
            except:
                pass
        hf.create_dataset('x_data', data=signal_test_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisefree', data=signal_test_noisefree)
        hf.create_dataset('y_data_noisy', data=signal_test_noisy)
        hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
        hf.create_dataset('snrs', data=signal_test_snr)
        hf.close()
    return

####################################
# Train neural network
####################################
def train(params=params,bounds=bounds,fixed_vals=fixed_vals,resume_training=False):

    # define which gpu to use during training
    gpu_num = str(0)                                            # first GPU used by default
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    # Let GPU consumption grow as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    # If resuming training, set KL ramp off
    if resume_training:
        params['resume_training'] = True
        params['ramp'] = False

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale, snrs_train = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'])

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test,_,snrs_test = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],load_condor=True)

    # reshape time series arrays for single channel ( N_samples,fs*duration,n_detectors -> (N_samples,fs*duration*n_detectors) )
    y_data_train = y_data_train.reshape(y_data_train.shape[0]*y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
#    y_data_train = y_data_train.reshape(y_data_train.shape[1],y_data_train.shape[2]*y_data_train.shape[3])
    y_data_test = y_data_test.reshape(y_data_test.shape[0],y_data_test.shape[1]*y_data_test.shape[2])
    y_data_test_noisefree = y_data_test_noisefree.reshape(y_data_test_noisefree.shape[0],y_data_test_noisefree.shape[1]*y_data_test_noisefree.shape[2])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # Save configuration file to public_html directory
    f = open('%s/latest_%s/params_%s.txt' % (params['plot_dir'],params['run_label'],params['run_label']),"w")
    f.write( str(params) )
    f.close()

    # load up the posterior samples (if they exist)
    # load generated samples back in
    post_files = []
    #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py

    # first identify directory with lowest number of total finished posteriors
    num_finished_post = int(1e8)
    for i in params['samplers']:
        if i == 'vitamin':
            continue
        for j in range(1):
            input_dir = '%s_%s%d/' % (params['pe_dir'],i,j+1)
            if type("%s" % input_dir) is str:
                dataLocations = ["%s" % input_dir]

            filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))      
            if len(filenames) < num_finished_post:
                sampler_loc = i + str(j+1)
                num_finished_post = len(filenames)

    dataLocations_try = '%s_%s' % (params['pe_dir'],sampler_loc)
    dataLocations = '%s_%s1' % (params['pe_dir'],params['samplers'][1])

    #for i,filename in enumerate(glob.glob(dataLocations[0])):
    i_idx = 0
    i = 0
    i_idx_use = []

    # Iterate over requested number of testing samples to use
    while i_idx < params['r']*params['r']:
        filename_try = '%s/%s_%d.h5py' % (dataLocations_try,params['bilby_results_label'],i)
        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

        # If file does not exist, skip to next file
        try:
            h5py.File(filename_try, 'r')
        except Exception as e:
            i+=1
            print(e)
            continue

        print()
        print('... Loading test sample -> ' + filename)
        post_files.append(filename)
        data_temp = {} 
        n = 0
       
        # Retrieve all source parameters to do inference on
        for q in params['inf_pars']:
             p = q + '_post'
             par_min = q + '_min'
             par_max = q + '_max'
             data_temp[p] = h5py.File(filename, 'r')[p][:]
             if p == 'psi_post':
                 data_temp[p] = np.remainder(data_temp[p],np.pi)
             if p == 'geocent_time_post':
                 data_temp[p] = data_temp[p] - params['ref_geocent_time']
             data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
             Nsamp = data_temp[p].shape[0]
             n = n + 1
        XS = np.zeros((Nsamp,n))
        j = 0

        # place retrieved source parameters in numpy array rather than dictionary
        for p,d in data_temp.items():
            XS[:,j] = d
            j += 1

        # Append test sample posteriors to existing array of other test sample posteriors
        if i_idx == 0:
            XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
        else:
            XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))


        # add index to mark progress through while loop
        i_idx_use.append(i_idx)
        i+=1
        i_idx+=1


    # Identify test samples that are present accross all Bayesian PE samplers
    y_data_test = y_data_test[i_idx_use,:]
    y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]
    x_data_test = x_data_test[i_idx_use,:]

    # reshape y data into channels last format for convolutional approach (if requested)
    if params['n_filters_r1'] != None:
        y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(fixed_vals['det'])))
        y_data_test_noisefree_copy = np.zeros((y_data_test_noisefree.shape[0],params['ndata'],len(fixed_vals['det'])))
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(fixed_vals['det'])))
        for i in range(y_data_test.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
                y_data_test_noisefree_copy[i,:,j] = y_data_test_noisefree[i,idx_range]
        y_data_test = y_data_test_copy
        y_data_noisefree_test = y_data_test_noisefree_copy

        for i in range(y_data_train.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    # run hyperparameter optimization
    if params['hyperparam_optim'] == True:

        # Run optimization
        search_result = gp_minimize(func=hyperparam_fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Negative Expected Improvement.
                            n_calls=params['hyperparam_n_call'],
                            x0=default_hyperparams)

        from skopt import dump
        dump(search_result, 'search_result_store')

        # plot best loss as a function of optimization step
        plt.close('all')
        plot_convergence(search_result)
        plt.savefig('%s/latest_%s/hyperpar_convergence.png' % (params['plot_dir'],params['run_label']))
        print('... Saved hyperparameter convergence loss to -> %s/latest_%s/hyperpar_convergence.png' % (params['plot_dir'],params['run_label']))
        print('... Did a hyperparameter search') 

    # train using user defined params
    else:
        CVAE_model.train(params, x_data_train, y_data_train,
                                 x_data_test, y_data_test, y_data_test_noisefree,
                                 y_normscale,
                                 "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                                 x_data_test, bounds, fixed_vals,
                                 XS_all,snrs_test) 

    return

# if we are now testing the network
def test(params=params,bounds=bounds,fixed_vals=fixed_vals,use_gpu=False):

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    if use_gpu == True:
        print("... GPU found")
        os.environ["CUDA_VISIBLE_DEVICES"]=str(params['gpu_num'])
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # Let GPU consumption grow as needed
        session = tf.compat.v1.Session(config=config)
    else:
        print("... Using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"]=''
        config = tf.compat.v1.ConfigProto()
        session = tf.compat.v1.Session(config=config)

    y_normscale = params['y_normscale']

    # load the testing data time series and source parameter truths
    x_data_test, y_data_test_noisefree, y_data_test,_,snrs_test = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],load_condor=True)

    # Make directory to store plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # reshape arrays for single channel network (this will be overwritten if channels last is requested by user)
    y_data_test = y_data_test.reshape(y_data_test.shape[0],y_data_test.shape[1]*y_data_test.shape[2])
    y_data_test_noisefree = y_data_test_noisefree.reshape(y_data_test_noisefree.shape[0],y_data_test_noisefree.shape[1]*y_data_test_noisefree.shape[2])

    # Make directory for plots
    os.system('mkdir -p %s/latest_%s' % (params['plot_dir'],params['run_label']))

    # load up the posterior samples (if they exist)
    # load generated samples back in
    post_files = []
    #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py

    # Identify directory with lowest number of total finished posteriors
    num_finished_post = int(1e8)
    for i in params['samplers']:
        if i == 'vitamin':# or i == 'emcee':
            continue
        for j in range(1):
            input_dir = '%s_%s%d/' % (params['pe_dir'],i,j+1)
            if type("%s" % input_dir) is str:
                dataLocations = ["%s" % input_dir]

            filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
            if len(filenames) < num_finished_post:
                sampler_loc = i + str(j+1)
                num_finished_post = len(filenames)

    samp_posteriors = {}
    # Iterate over all Bayesian PE samplers
    for samp_idx in params['samplers'][1:]:
        dataLocations_try = '%s_%s' % (params['pe_dir'],sampler_loc)
        dataLocations = '%s_%s' % (params['pe_dir'],samp_idx+'1')
        i_idx = 0
        i = 0
        i_idx_use = []
        x_data_test_unnorm = np.copy(x_data_test)


        # Iterate over all requested testing samples
        while i_idx < params['r']*params['r']:


            filename_try = '%s/%s_%d.h5py' % (dataLocations_try,params['bilby_results_label'],i)
            filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

            # If file does not exist, skip to next file
            try:
                h5py.File(filename_try, 'r')
            except Exception as e:
                i+=1
                print(e)
                continue

            print('... Loading test sample file -> ' + filename)
            post_files.append(filename)
            
            # Prune emcee samples for bad likelihood chains
            if samp_idx == 'emcee':
                emcee_pruned_samples = prune_samples(filename,params)

            data_temp = {}
            n = 0
            for q_idx,q in enumerate(params['inf_pars']):
                 p = q + '_post'
                 par_min = q + '_min'
                 par_max = q + '_max'
                 if p == 'psi_post':
                     data_temp[p] = np.remainder(data_temp[p],np.pi)

                 if samp_idx == 'emcee':
                     data_temp[p] = emcee_pruned_samples[:,q_idx]
                 else:
                     data_temp[p] = np.float64(h5py.File(filename, 'r')[p][:])

                 if p == 'geocent_time_post' or p == 'geocent_time_post_with_cut':
                     data_temp[p] = np.subtract(np.float64(data_temp[p]),np.float64(params['ref_geocent_time'])) 

                 Nsamp = data_temp[p].shape[0]
                 n = n + 1

            XS = np.zeros((Nsamp,n))
            j = 0

            # store posteriors in numpy array rather than dictionary
            for p,d in data_temp.items():
                XS[:,j] = d
                j += 1

            rand_idx_posterior = np.random.choice(np.linspace(0,XS.shape[0]-1,dtype=np.int),params['n_samples'])
            # Append test sample posterior to existing array of test sample posteriors
            if i_idx == 0:
                #XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
                XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
            else:
                try:
                    #XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))
                    XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))
                except ValueError as error: # If not enough posterior samples, exit with ValueError
                    print('Not enough samples from the posterior generated')
                    print(error)
                    exit()

            # Get unnormalized array with source parameter truths
            for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'

                x_data_test_unnorm[i_idx,q_idx] = (x_data_test_unnorm[i_idx,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

            # Add to index in order to progress through while loop iterating over testing samples
            i_idx_use.append(i_idx)
            i+=1
            i_idx+=1

        # Add all testing samples for current Bayesian PE sampler to dictionary of all other Bayesian PE sampler test samples
        samp_posteriors[samp_idx+'1'] = XS_all

    # Ensure no failed test sample Bayesian PE runs are used
    x_data_test = x_data_test[i_idx_use,:]
    x_data_test_unnorm = x_data_test_unnorm[i_idx_use,:]
    y_data_test = y_data_test[i_idx_use,:]
    y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]

    # reshape y data into channels last format for convolutional approach
    y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(fixed_vals['det'])))
    if params['n_filters_r1'] != None:
        for i in range(y_data_test.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
        y_data_test = y_data_test_copy
    

    # Reshape time series  array to right format for 1-channel configuration
    if params['by_channel'] == False:
        y_data_test_new = []
        for sig in y_data_test:
            y_data_test_new.append(sig.T)
        y_data_test = np.array(y_data_test_new)
        del y_data_test_new

    # check is basemap is installed
    if not skyplotting_usage:
        params['Make_sky_plot'] = False

    VI_pred_all = []
    # Iterate over total number of testing samples
    for i in range(params['r']*params['r']):

        # If True, continue through and make corner plots
        if params['make_corner_plots'] == False:
            break
        
        # Generate ML posteriors using pre-trained model
        if params['n_filters_r1'] != None: # for convolutional approach
             VI_pred, dt, _  = CVAE_model.run(params, np.expand_dims(y_data_test[i],axis=0), np.shape(x_data_test)[1],
                                                         y_normscale,
                                                         "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])
        else:                                                          # for fully-connected approach
            VI_pred, dt, _  = CVAE_model.run(params, y_data_test[i].reshape([1,-1]), np.shape(x_data_test)[1],
                                                         y_normscale,
                                                         "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'])

        # Make corner corner plots
        bins=50
       
        # Define default corner plot arguments
        defaults_kwargs = dict(
                    bins=bins, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16), show_titles=False,
                    truth_color='black', quantiles=None,#[0.16, 0.84],
                    levels=(0.50,0.90), density=True, stacked=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

        matplotlib.rc('text', usetex=True)                
        parnames = []
    
        # Get infered parameter latex labels for corner plot
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, params['inf_pars']):
                parnames.append(params['cornercorner_parnames'][k_idx])

        # unnormalize the predictions from VICI (comment out if not wanted)
        color_cycle=['tab:blue','tab:green','tab:purple','tab:orange']
        legend_color_cycle=['blue','green','purple','orange']
        for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'
                VI_pred[:,q_idx] = (VI_pred[:,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]


        # Iterate over all Bayesian PE samplers and plot results
        custom_lines = []
        truths = x_data_test_unnorm[i,:]
        for samp_idx,samp in enumerate(params['samplers'][1:]):

            bilby_pred = samp_posteriors[samp+'1'][i]

            # compute weights, otherwise the 1d histograms will be different scales, could remove this
            weights = np.ones(len(VI_pred)) * (len(samp_posteriors[samp+'1'][i]) / len(VI_pred))
            if samp_idx == 0:
                figure = corner.corner(bilby_pred,**defaults_kwargs,labels=parnames,
                               color=color_cycle[samp_idx],
                               truths=truths
                               )
            else:
                figure = corner.corner(bilby_pred,**defaults_kwargs,labels=parnames,
                               color=color_cycle[samp_idx],
                               truths=truths,
                               fig=figure)
            custom_lines.append(Line2D([0], [0], color=legend_color_cycle[samp_idx], lw=4))

        # plot predicted ML results
        corner.corner(VI_pred, **defaults_kwargs, labels=parnames,
                           color='tab:red', fill_contours=True,
                           fig=figure)
        custom_lines.append(Line2D([0], [0], color='red', lw=4))

        if params['Make_sky_plot'] == True:
            # Compute skyplot
#            left, bottom, width, height = [0.55, 0.47, 0.5, 0.39] # orthographic representation
#            left, bottom, width, height = [0.525, 0.47, 0.45, 0.44] # mollweide representation
            print()
            print('... Generating sky plot')
            print()
            left, bottom, width, height = [0.46, 0.6, 0.5, 0.5] # switch with waveform positioning
            ax_sky = figure.add_axes([left, bottom, width, height]) 

            sky_color_cycle=['blue','green','purple','orange']
            sky_color_map_cycle=['Blues','Greens','Purples','Oranges']
            for samp_idx,samp in enumerate(params['samplers'][1:]):
                bilby_pred = samp_posteriors[samp+'1'][i]
                if samp_idx == 0:
                    ax_sky = plot_sky(bilby_pred[:,-2:],filled=False,cmap=sky_color_map_cycle[samp_idx],col=sky_color_cycle[samp_idx],trueloc=truths[-2:])
                else:
                    ax_sky = plot_sky(bilby_pred[:,-2:],filled=False,cmap=sky_color_map_cycle[samp_idx],col=sky_color_cycle[samp_idx], trueloc=truths[-2:], ax=ax_sky)
            ax_sky = plot_sky(VI_pred[:,-2:],filled=True,trueloc=truths[-2:],ax=ax_sky)


        #left, bottom, width, height = [0.34, 0.82, 0.3, 0.17] # standard positioning
        left, bottom, width, height = [0.67, 0.48, 0.3, 0.2] # swtiched with skymap positioning
        ax2 = figure.add_axes([left, bottom, width, height])
        # plot waveform in upper-right hand corner
        ax2.plot(np.linspace(0,1,params['ndata']),y_data_test_noisefree[i,:params['ndata']],color='cyan',zorder=50)
        snr = round(snrs_test[i,0],2)
        if params['n_filters_r1'] != None:
            if params['by_channel'] == False:
                 ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,0,:params['ndata']],color='darkblue')#,label='SNR: '+str(snr))
            else:
                ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,:params['ndata'],0],color='darkblue')#,label='SNR: '+str(snr))
        else:
            ax2.plot(np.linspace(0,1,params['ndata']),y_data_test[i,:params['ndata']],color='darkblue')#,label='SNR: '+str(snr))
        ax2.set_xlabel(r"$\textrm{time (seconds)}$",fontsize=16)
        ax2.yaxis.set_visible(False)
        ax2.tick_params(axis="x", labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)
        ax2.set_ylim([-6,6])
        ax2.grid(False)
        ax2.margins(x=0,y=0)

        # Save corner plot to latest public_html directory
        figure.legend(handles=custom_lines, labels=['Dynesty', 'ptemcee', 'VItamin'],
                      loc=(0.86,0.22), fontsize=20)
        plt.savefig('%s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        plt.close()
        del figure
        print()
        print('... Made corner plot: %s' % str(i+1))
        print('... Saved corner plot to -> %s/latest_%s/corner_plot_%s_%d.png' % (params['plot_dir'],params['run_label'],params['run_label'],i))
        print()

        # Store ML predictions for later plotting use
        VI_pred_all.append(VI_pred)

    VI_pred_all = np.array(VI_pred_all)

    # Define pp and KL plotting class
#    XS_all = None; x_data_test = None; y_data_test = None; y_normscale = None; snrs_test = None
    plotter = plotting.make_plots(params,XS_all,VI_pred_all,x_data_test)

    if params['make_kl_plot'] == True:    
        # Make KL plots
        plotter.gen_kl_plots(CVAE_model,y_data_test,x_data_test,y_normscale,bounds,snrs_test)

    # Make bilby pp plot
#    plotter.plot_bilby_pp(VICI_inverse_model,y_data_test,x_data_test,0,y_normscale,x_data_test,bounds)
#    exit()

    if params['make_pp_plot'] == True:
        # Make pp plot
        plotter.plot_pp(CVAE_model,y_data_test,x_data_test,0,y_normscale,x_data_test,bounds)

    if params['make_loss_plot'] == True:
        plotter.plot_loss()

    return

def gen_samples(params='params.json',bounds='bounds.json',fixed_vals='fixed_vals.json',model_loc='model-ex/model.ckpt',test_set='test-ex/',num_samples=None,plot_corner=True,use_gpu=False):
    """ Function to generate VItamin samples given a trained model
    """

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    # Load parameters files
    with open(params, 'r') as fp:
        params = json.load(fp)
    with open(bounds, 'r') as fp:
        bounds = json.load(fp)
    with open(fixed_vals, 'r') as fp:
        fixed_vals = json.load(fp)

    if use_gpu == True:
        print('GPU found')
        os.environ["CUDA_VISIBLE_DEVICES"]=str(params['gpu_num'])
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # Let GPU consumption grow as needed
    else:
        print("No GPU found")
        os.environ["CUDA_VISIBLE_DEVICES"]=''
        config = tf.compat.v1.ConfigProto() 

    session = tf.compat.v1.Session(config=config)

    if num_samples != None:
        params['n_samples'] = num_samples

    ########################
    # load generated samples
    ########################
    files = []
   
    # Get list of all training/testing files and define dictionary to store values in files
    if type("%s" % test_set) is str:
        dataLocations = ["%s" % test_set]
        data={'y_data_noisy': []}
    # Sort files from first generated to last generated
    filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))

    # Append training/testing filenames to list. Ignore those that can't be loaded
    for filename in filenames:
        try:
            files.append(filename)
        except OSError:
            print('Could not load requested file')
            continue

    # Iterate over all training/testing files and store source parameters, time series and SNR info in dictionary
    for filename in files:
        try:
            data_temp={'y_data_noisy': h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:]}
            data['y_data_noisy'].append(np.expand_dims(data_temp['y_data_noisy'], axis=0))
        except OSError:
            print('Could not load requested file')
            continue

    # Extract the prior bounds from training/testing files
    data['y_data_noisy'] = np.concatenate(np.array(data['y_data_noisy']), axis=0)
    

    y_data_test = data['y_data_noisy']

    # Define time series normalization factor to use on test samples. We consistantly use the same normscale value if loading by chunks
    y_normscale = params['y_normscale']   
 
    y_data_test = y_data_test.reshape(y_data_test.shape[0],y_data_test.shape[1]*y_data_test.shape[2])
    # reshape y data into channels last format for convolutional approach
    y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(fixed_vals['det'])))
    if params['n_filters_r1'] != None:
        for i in range(y_data_test.shape[0]):
            for j in range(len(fixed_vals['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)       
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
        y_data_test = y_data_test_copy
    num_timeseries=y_data_test.shape[0]
    samples = np.zeros((num_timeseries,num_samples,len(params['inf_pars'])))
    for i in range(num_timeseries):
        samples[i,:], dt, _  = CVAE_model.run(params, np.expand_dims(y_data_test[i],axis=0), len(params['inf_pars']),
                                                              params['y_normscale'],
                                                              model_loc)
        print('... Runtime to generate samples is: ' + str(dt))

        # unnormalize predictions
        for q_idx,q in enumerate(params['inf_pars']):
            par_min = q + '_min'
            par_max = q + '_max'
            samples[i,:,q_idx] = (samples[i,:,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

        # plot results
        if plot_corner==True:
            # Get infered parameter latex labels for corner plot
            parnames=[]
            for k_idx,k in enumerate(params['rand_pars']):
                if np.isin(k, params['inf_pars']):
                    parnames.append(params['cornercorner_parnames'][k_idx])
            figure = corner.corner(samples[i,:,:],labels=parnames)
            plt.savefig('./vitamin_corner_timeseries-%d.png' % i)
            plt.close()
            print('... Saved corner plot to -> ./vitamin_corner_timeseries-%d.png' % i)
            print()

    print('... All posterior samples generated for all waveforms in test sample directory!')
    return samples

# If running module from command line
if args.gen_train:
    gen_train(params,bounds,fixed_vals)
if args.gen_test:
    gen_test(params,bounds,fixed_vals)
if args.train:
    train(params,bounds,fixed_vals)
if args.test:
    test(params,bounds,fixed_vals,use_gpu=bool(args.use_gpu))
if args.gen_samples:
    gen_samples(params,bounds,fixed_vals,model_loc=args.pretrained_loc,
                test_set=args.test_set_loc,num_samples=args.num_samples,use_gpu=bool(args.use_gpu))

