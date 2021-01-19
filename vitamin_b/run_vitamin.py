######################################################################################################################

# -- Variational Inference for Gravitational wave Parameter Estimation --


#######################################################################################################################

import warnings
warnings.filterwarnings("ignore")
import os
from os import path
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
from lal import GreenwichMeanSiderealTime

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
    from .models.neural_networks.vae_utils import convert_ra_to_hour_angle
    from .gen_benchmark_pe import importance_sampling
except (ModuleNotFoundError, ImportError):
    from models import CVAE_model
    from gen_benchmark_pe import run
    import plotting
    from plotting import prune_samples
    from models.neural_networks.vae_utils import convert_ra_to_hour_angle
    from gen_benchmark_pe import importance_sampling

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

""" Script has several main functions:
1.) Generate training data
2.) Generate testing data
3.) Train model
4.) Test model
5.) Generate samples only given model and timeseries
6.) Apply importance sampling to VItamin results
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
parser.add_argument("--num_samples", type=int, default=10000, help="number of posterior samples to generate")
parser.add_argument("--use_gpu", default=False, help="if True, use gpu")
parser.add_argument("--importance_sampling", default=False, help="Apply importance sampling to VItamin posterior samples")
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
kernel_1 = Integer(low=3, high=11, name='kernel_1')
strides_1 = Integer(low=1, high=2, name='strides_1')
pool_1 = Integer(low=1, high=2, name='pool_1')
kernel_2 = Integer(low=3, high=11, name='kernel_2')
strides_2 = Integer(low=1, high=2, name='strides_2')
pool_2 = Integer(low=1, high=2, name='pool_2')
kernel_3 = Integer(low=3, high=11, name='kernel_3')
strides_3 = Integer(low=1, high=2, name='strides_3')
pool_3 = Integer(low=1, high=2, name='pool_3')
kernel_4 = Integer(low=3, high=11, name='kernel_4')
strides_4 = Integer(low=1, high=2, name='strides_4')
pool_4 = Integer(low=1, high=2, name='pool_4')
kernel_5 = Integer(low=3, high=11, name='kernel_5')
strides_5 = Integer(low=1, high=2, name='strides_5')
pool_5 = Integer(low=1, high=2, name='pool_5')

z_dimension = Integer(low=16, high=100, name='z_dimension')
n_modes = Integer(low=2, high=50, name='n_modes')
n_filters_1 = Integer(low=3, high=33, name='n_filters_1')
n_filters_2 = Integer(low=3, high=33, name='n_filters_2')
n_filters_3 = Integer(low=3, high=33, name='n_filters_3')
n_filters_4 = Integer(low=3, high=33, name='n_filters_4')
n_filters_5 = Integer(low=3, high=33, name='n_filters_5')
batch_size = Integer(low=32, high=33, name='batch_size')
n_weights_fc_1 = Integer(low=8, high=2048, name='n_weights_fc_1')
n_weights_fc_2 = Integer(low=8, high=2048, name='n_weights_fc_2')
n_weights_fc_3 = Integer(low=8, high=2048, name='n_weights_fc_3')

num_r1_conv = Integer(low=1, high=5, name='num_r1_conv')
num_r2_conv = Integer(low=1, high=5, name='num_r2_conv')
num_q_conv = Integer(low=1, high=5, name='num_q_conv')
num_r1_hidden = Integer(low=1, high=3, name='num_r1_hidden')
num_r2_hidden = Integer(low=1, high=3, name='num_r2_hidden')
num_q_hidden = Integer(low=1, high=3, name='num_q_hidden')

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
              kernel_5,
              strides_5,
              pool_5,
              z_dimension,
              n_modes,
              n_filters_1,
              n_filters_2,
              n_filters_3,
              n_filters_4,
              n_filters_5,
              batch_size,
              n_weights_fc_1,
              n_weights_fc_2,
              n_weights_fc_3,
              num_r1_conv,
              num_r2_conv,
              num_q_conv,
              num_r1_hidden,
              num_r2_hidden,
              num_q_hidden]

# dummy value for initial hyperparameter best KL (to be minimized). Doesn't need to be changed.
best_loss = int(1994)

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

    Parameters
    ----------
    params: dict
        Dictionary containing parameter values of run
    bounds: dict
        Dictionary containing the allowed bounds of source GW parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    input_dir: str
        Directory where training or testing files are stored
    inf_pars: list
        list of parameters to infer when training ML model
    load_condor: bool
        if True, load test samples rather than training samples

    Returns
    -------
    x_data: array_like
        array containing training/testing source parameter values
    y_data: array_like
        array containing training/testing noise-free times series
    y_data_noisy: array_like
        array containing training/testing noisy time series	
    y_normscale: float
        value by which to normalize all time series to be between zero and one
    snrs: array_like
        array containing optimal snr values for all training/testing time series
    """

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
    file_IDs = []
    for filename in train_files:

        if load_condor == True:
            # Don't load files which are not consistent between samplers
            for samp_idx_inner in params['samplers'][1:]:
                inner_file_existance = True
                dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
                filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],int(filename.split('_')[-1].split('.')[0]))
                # If file does not exist, skip to next file
                inner_file_existance = os.path.isfile(filename_inner) 
                print(filename_inner)
                if inner_file_existance == False:
                    break

            if inner_file_existance == False:
                print('File not consistent beetween samplers')
                continue

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
            if load_condor==True:
                file_IDs.append(int(filename.split('_')[-1].split('.')[0]) + int(params['testing_data_seed'])) # Save file number
            print('... Loaded file ' + filename)
        except OSError:
            print('Could not load requested file')
            continue
    snrs = np.array(snrs)

    # Extract the prior bounds from training/testing files
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.concatenate(np.array(data['y_data_noisefree']), axis=0)
    data['y_data_noisy'] = np.concatenate(np.array(data['y_data_noisy']), axis=0)
    
    # expand dimensions if only using one test sample
    if data['x_data'].ndim == 1:
        data['x_data'] = np.expand_dims(data['x_data'],axis=0)

    # Save all unnormalized values prior to RA conversion
    all_par = np.copy(data['x_data'])

    # convert ra to hour angle for training data only
    if load_condor==False:
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, rand_pars=True)


    # Normalise the source parameters np.remainder(blah,np.pi)
    for i,k in enumerate(data_temp['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'

        # normalize by bounds
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

    # Order all source parameter values
    idx = []
    print()
    for k in params['rand_pars']:
        if load_condor == False:
            print('... ' + k + ' will be inferred')
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    all_par = all_par[:,idx]

    importance_sampling_info = dict(all_par = all_par,
                                    file_IDs = file_IDs
                                   )

    return x_data, y_data, y_data_noisy, y_normscale, snrs, importance_sampling_info

@use_named_args(dimensions=dimensions)
def hyperparam_fitness(kernel_1, strides_1, pool_1,
                       kernel_2, strides_2, pool_2,
                       kernel_3, strides_3, pool_3,
                       kernel_4, strides_4, pool_4,
                       kernel_5, strides_5, pool_5,
                       z_dimension,n_modes,
                       n_filters_1,n_filters_2,n_filters_3,n_filters_4,n_filters_5,
                       batch_size,
                       n_weights_fc_1,n_weights_fc_2,n_weights_fc_3,
                       num_r1_conv,num_r2_conv,num_q_conv,
                       num_r1_hidden,num_r2_hidden,num_q_hidden):

    """ Fitness function used in Gaussian Process hyperparameter optimization 
    Returns a value to be minimized (in this case, the total loss of the 
    neural network during training.

    Parameters
    ----------
    kernel_1: skopt function
        Range over which kernel size in first CNN layer is allowed to vary
    kernel_2: skopt function
        Range over which kernel size in second CNN layer is allowed to vary
    kernel_3: skopt function
        Range over which kernel size in third CNN layer is allowed to vary  
    kernel_4: skopt function
        Range over which kernel size in fourth CNN layer is allowed to vary
    strides_1: skopt function
        Range over which stride size in first CNN layer is allowed to vary 
    strides_2: skopt function
        Range over which stride size in second CNN layer is allowed to vary
    strides_3: skopt function
        Range over which stride size in third CNN layer is allowed to vary
    strides_4: skopt function
        Range over which stride size in fourth CNN layer is allowed to vary
    pool_1: skopt function
        Range over which pool size in first CNN layer is allowed to vary
    pool_2: skopt function
        Range over which pool size in second CNN layer is allowed to vary
    pool_3: skopt function
        Range over which pool size in third CNN layer is allowed to vary
    pool_4: skopt function
        Range over which pool size in fourth CNN layer is allowed to vary
    z_dimension: skopt function
        Range over which latent space is allowed to vary
    n_modes: skopt function
        Range over which the number of gaussian modes in latent space is allowed to vary
    n_filters_1: skopt function
        Range over which the number of filters in first CNN layer is allowed to vary
    n_filters_2: skopt function
        Range over which the number of filters in second CNN layer is allowed to vary
    n_filters_3: skopt function
        Range over which the number of filters in third CNN layer is allowed to vary
    n_filters_4: skopt function
        Range over which the number of filters in fourth CNN layer is allowed to vary
    batch_size: skopt function
        Range over which the batch size is allowed to vary
    n_weights_fc_1: skopt function
        Range over which the number of neurons in the first hidden layer is allowed to vary
    n_weights_fc_2: skopt function
        Range over which the number of neurons in the second hidden layer is allowed to vary
    n_weights_fc_3: skopt function
        Range over which the number of neurons in the third hidden layer is allowed to vary

    Returns
    -------
    VICI_loss: float
        Total loss of the current optimized network
    """

    global params, bounds, fixed_vals
    global x_data_train, y_data_train, x_data_test, y_data_test, y_normscale, y_data_test_noisefree, XS_all
    global cur_hyperparam_iter

    # Check for requried parameters files
    if params == None or bounds == None or fixed_vals == None:
        print('Missing either params file, bounds file or fixed vals file')
        exit()

    try :
        # Load parameters files
        with open(params, 'r') as fp:
            params = json.load(fp)
        with open(bounds, 'r') as fp:
            bounds = json.load(fp)
        with open(fixed_vals, 'r') as fp:
            fixed_vals = json.load(fp)
    except TypeError:
        print('Already loaded params files. Skipping ...')

    # set tunable hyper-parameters
    params['filter_size_r1'] = [kernel_1,kernel_2,kernel_3,kernel_4,kernel_5][:num_r1_conv]
    params['filter_size_r2'] = [kernel_1,kernel_2,kernel_3,kernel_4,kernel_5][:num_r2_conv]
    params['filter_size_q'] = [kernel_1,kernel_2,kernel_3,kernel_4,kernel_5][:num_q_conv]
    params['n_filters_r1'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4,n_filters_5][:num_r1_conv]
    params['n_filters_r2'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4,n_filters_5][:num_r2_conv]
    params['n_filters_q'] = [n_filters_1,n_filters_2,n_filters_3,n_filters_4,n_filters_5][:num_q_conv]
    params['conv_strides_r1'] = [strides_1,strides_2,strides_3,strides_4,strides_5][:num_r1_conv]
    params['conv_strides_r2'] = [strides_1,strides_2,strides_3,strides_4,strides_5][:num_r2_conv] 
    params['conv_strides_q'] = [strides_1,strides_2,strides_3,strides_4,strides_5][:num_q_conv] 
    params['maxpool_r1'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_r1_conv]
    params['maxpool_r2'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_r2_conv]
    params['maxpool_q'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_q_conv]
    params['pool_strides_r1'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_r1_conv]
    params['pool_strides_r2'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_r2_conv]
    params['pool_strides_q'] = [pool_1,pool_2,pool_3,pool_4,pool_5][:num_q_conv]
    params['z_dimension'] = z_dimension
    params['n_modes'] = n_modes
    params['batch_size'] = batch_size
    params['n_weights_r1'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3][:num_r1_hidden]
    params['n_weights_r2'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3][:num_r2_hidden]
    params['n_weights_q'] = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3][:num_q_hidden]

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
    print('n_filters: {}'.format(params['n_filters_r1']))
    print('batch_size: {}'.format(batch_size))
    print('n_weights_r1_1: {}'.format(n_weights_fc_1))
    print('n_weights_r1_2: {}'.format(n_weights_fc_2))
    print('n_weights_r1_3: {}'.format(n_weights_fc_3))
    print('num_r1_conv: {}'.format(num_r1_conv))
    print('num_r2_conv: {}'.format(num_r2_conv))
    print('num_q_conv: {}'.format(num_q_conv))
    print('num_r1_hidden: {}'.format(num_r1_hidden))
    print('num_r2_hidden: {}'.format(num_r2_hidden))
    print('num_q_hidden: {}'.format(num_q_hidden))
    print()

    # Update load iteration accoring to new batch size
    params['load_iteration'] = int((params['load_chunk_size'] * 25)/params['batch_size'])
    params['plot_interval'] = 5e6 

    start_time = time.time()
    print('start time: {}'.format(strftime('%X %x %Z'))) 

     # Make best run directory if it does not already exist
    if not path.exists("inverse_model_dir_%s/best_model" % (params['run_label'])):
        os.mkdir("inverse_model_dir_%s" % (params['run_label']))
        os.mkdir("inverse_model_dir_%s/best_model" % (params['run_label']))
    else:
        pass

    # Make run directory
    if not path.exists("inverse_model_dir_%s/run_%d/" % (params['run_label'],cur_hyperparam_iter)):
        os.mkdir("inverse_model_dir_%s/run_%d/" % (params['run_label'],cur_hyperparam_iter))
    else:
        pass

    # Perform training
    try:
        CVAE_loss, CVAE_session, CVAE_saver, CVAE_savedir, nan_flag, plotdata = CVAE_model.train(params, x_data_train, y_data_train,
                                 x_data_test, y_data_test, y_data_test_noisefree,
                                 y_normscale,
                                 "inverse_model_dir_%s/run_%d/inverse_model.ckpt" % (params['run_label'],cur_hyperparam_iter),
                                 x_data_test, bounds, fixed_vals,
                                 XS_all)
    except Exception as e:

        if e == 'KeyboardInterrupt':
            exit()

        print('There was an exception. Returning large loss.')
        print()
        print(e)
        CVAE_loss = 1994

        # Add 1 to hyperparam iteration counter
        cur_hyperparam_iter += 1

        return CVAE_loss

    end_time = time.time()
    print('Run time : {} h'.format((end_time-start_time)/3600))

    # Print the loss.
    print()
    print("Total loss: {0:.2}".format(CVAE_loss))
    print()

    # update variable outside of this function using global keyword
    global best_loss

    if nan_flag == False:
        # Save model 
        save_path = CVAE_saver.save(CVAE_session,CVAE_savedir)

    # save hyperparameters
    converged_hyperpar_dict = dict(filter_size = [kernel_1,kernel_2,kernel_3,kernel_4,kernel_5],
                                   conv_strides = [strides_1,strides_2,strides_3,strides_4,strides_5],
                                   maxpool = [pool_1,pool_2,pool_3,pool_4,pool_5],
                                   pool_strides = [pool_1,pool_2,pool_3,pool_4,pool_5],
                                   z_dimension = params['z_dimension'],
                                   n_modes = params['n_modes'],
                                   n_filters = [n_filters_1,n_filters_2,n_filters_3,n_filters_4,n_filters_5],
                                   batch_size = params['batch_size'],
                                   n_weights_fc = [n_weights_fc_1,n_weights_fc_2,n_weights_fc_3],
                                   loss = CVAE_loss,
                                   runtime = (end_time-start_time)/3600,
                                   num_r1_conv = num_r1_conv,
                                   num_r2_conv = num_r2_conv,
                                   num_q_conv = num_q_conv,
                                   num_r1_hidden = num_r1_hidden,
                                   num_r2_hidden = num_r2_hidden,
                                   num_q_hidden = num_q_hidden,
                                   run_ID = cur_hyperparam_iter)

    # Save current run hyperparameters
    f = open("inverse_model_dir_%s/run_%d/hyperparams.txt" % (params['run_label'],cur_hyperparam_iter),"w")
    f.write( str(converged_hyperpar_dict) )
    f.close()

    # Save loss plot data
    try:
        np.savetxt('inverse_model_dir_%s/run_%d/loss_data.txt' % (params['run_label'],cur_hyperparam_iter), np.array(plotdata))
    except FileNotFoundError as err:
        pass 

    # save best model to seperate best model directory
    if CVAE_loss < best_loss:

        # update the best loss
        best_loss = CVAE_loss
        
        # Save best model
        best_model_savedir = "inverse_model_dir_%s/best_model/inverse_model.ckpt" % (params['run_label']) 
        try:
            shutil.copytree("inverse_model_dir_%s/run_%d/" % (params['run_label'],cur_hyperparam_iter),
                            "inverse_model_dir_%s/best_model/" % (params['run_label']))
        except FileExistsError:
            shutil.rmtree("inverse_model_dir_%s/best_model/" % (params['run_label']))
            shutil.copytree("inverse_model_dir_%s/run_%d/" % (params['run_label'],cur_hyperparam_iter),
                            "inverse_model_dir_%s/best_model/" % (params['run_label']))
        if nan_flag == False:
            save_path = CVAE_saver.save(CVAE_session,best_model_savedir)

        # Print best loss.
        print()
        print("New best loss: {0:.2}".format(best_loss))
        print()

    # clear tensorflow session
    CVAE_session.close()

    # Add 1 to hyperparam iteration counter
    cur_hyperparam_iter += 1

    return CVAE_loss

def gen_train(params=params,bounds=bounds,fixed_vals=fixed_vals):
    """ Generate training samples

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
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
            signal_train, signal_train_pars,snrs = run(sampling_frequency=params['ndata']/params['duration'],
                                                          duration=params['duration'],
                                                          N_gen=params['tset_split'],
                                                          ref_geocent_time=params['ref_geocent_time'],
                                                          bounds=bounds,
                                                          fixed_vals=fixed_vals,
                                                          rand_pars=params['rand_pars'],
                                                          seed=params['training_data_seed']+i,
                                                          label=params['run_label'],
                                                          training=True,det=params['det'],
                                                          psd_files=params['psd_files'],
                                                          use_real_det_noise=params['use_real_det_noise'])
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
                hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
            except:
                pass
        hf.create_dataset('x_data', data=signal_train_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisy', data=np.array([]))
        hf.create_dataset('y_data_noisefree', data=signal_train)
        hf.create_dataset('snrs', data=snrs)
        hf.close()
    return

def gen_test(params=params,bounds=bounds,fixed_vals=fixed_vals):
    """ Generate testing sample time series and posteriors using Bayesian inference (bilby)

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
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

    # Make testing set directory
    os.system('mkdir -p %s' % params['test_set_dir'])

    # Add 1 to sampler names
    for i in range(len(params['samplers'])):
        params['samplers'][i] = params['samplers'][i]+'1'

    # Make testing samples
    for i in range(params['r']):
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
                                                      do_pe=params['doPE'],det=params['det'],
                                                      psd_files=params['psd_files'],
                                                      use_real_det_noise=params['use_real_det_noise'],
                                                      use_real_events=params['use_real_events'],
                                                      samp_idx=i)

        signal_test_noisy = temp_noisy
        signal_test_noisefree = temp_noisefree
        signal_test_pars = temp_pars
        signal_test_snr = temp_snr

        print("Generated: %s/%s_%s.h5py ..." % (params['test_set_dir'],params['bilby_results_label'],params['run_label']))

        # Save generated testing samples in h5py format
        hf = h5py.File('%s/%s_%d.h5py' % (params['test_set_dir'],params['bilby_results_label'],i),'w')
        for k, v in params.items():
            try:
                hf.create_dataset(k,data=v) 
                hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
            except:
                pass
        hf.create_dataset('x_data', data=signal_test_pars)
        for k, v in bounds.items():
            hf.create_dataset(k,data=v)
        hf.create_dataset('y_data_noisefree', data=signal_test_noisefree)
        hf.create_dataset('y_data_noisy', data=signal_test_noisy)
        hf.create_dataset('snrs', data=signal_test_snr)
        hf.close()
    return

def train(params=params,bounds=bounds,fixed_vals=fixed_vals,resume_training=False):
    """ Train neural network given pre-made training/testing samples

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    resume_training: bool
        If True, continue training a pre-trained model.
    """
    
    global x_data_train, y_data_train, x_data_test, y_data_test, y_normscale, y_data_test_noisefree, XS_all

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

    # if doing hour angle, use hour angle bounds on RA
    if params['convert_to_hour_angle']:
        bounds['ra_min'] = params['hour_angle_range'][0]
        bounds['ra_max'] = params['hour_angle_range'][1]

    # define which gpu to use during training
    gpu_num = str(params['gpu_num'])                                            # first GPU used by default
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

    # Let GPU consumption grow as needed
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # If resuming training, set KL ramp off
    if resume_training or params['resume_training']:
        params['resume_training'] = True
        params['ramp'] = False

    global x_data_train

    # load the noisefree training data back in
    x_data_train, y_data_train, _, y_normscale, snrs_train, _ = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'], load_condor=False)

    # load the noisy testing data back in
    x_data_test, y_data_test_noisefree, y_data_test, _, snrs_test, _ = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'], load_condor=True)

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
    if params['pe_dir'] != None:
        # load generated samples back in
        post_files = []
        #~/bilby_outputs/bilby_output_dynesty1/multi-modal3_0.h5py

        # first identify directory with lowest number of total finished posteriors
        num_finished_post = int(1e8)
        for i in params['samplers']:
            if i == 'vitamin':
                continue

            # remove any remaining resume files
            resume_files=glob.glob('%s_%s1/*.resume*' % (params['pe_dir'],i))
            filelist = [resume_files]
            for file_idx,file_type in enumerate(filelist):
                for file in file_type:
                    os.remove(file) 

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

        i_idx = 0
        i = 0
        i_idx_use = []

        # Iterate over requested number of testing samples to use
        while i_idx < params['r']:

#            filename_try = '%s/%s_%d.h5py' % (dataLocations_try,params['bilby_results_label'],i)
            filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

            # Assert user has the minimum number of test samples generated
            number_of_files_in_dir = len(os.listdir(dataLocations))
            try:
                assert number_of_files_in_dir >= params['r']
            except Exception as e:
                print(e)
                print('You are requesting to use more test GW time series than you have made.')
                print('... Exiting program now')
                exit()

            # Check files are present accross all samplers
            for samp_idx_inner in params['samplers'][1:]:
                inner_file_existance = True
                if samp_idx_inner == params['samplers'][1]:
                    inner_file_existance = os.path.isfile(filename)
                    if inner_file_existance == False:
                        break
                    else:
                        continue

                dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
                filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
                # If file does not exist, skip to next file
                inner_file_existance = os.path.isfile(filename_inner)
                if inner_file_existance == False:
                    break

            if inner_file_existance == False:
                i+=1
#                print('File does not exist for one of the samplers')
                continue
 
           # If file does not exist, skip to next file
 #           try:
 #               h5py.File(filename_try, 'r')
 #           except Exception as e:
 #               i+=1
 #               print(e)
 #               continue

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
            rand_idx_posterior = np.linspace(0,XS.shape[0]-1,num=params['n_samples'],dtype=np.int)
            np.random.shuffle(rand_idx_posterior)
            rand_idx_posterior = rand_idx_posterior[:params['n_samples']]
            if i_idx == 0:
                #XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
                XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
            else:
                #XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))
                XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))

            # add index to mark progress through while loop
            i_idx_use.append(i_idx)
            i+=1
            i_idx+=1


        # Identify test samples that are present accross all Bayesian PE samplers
#        y_data_test = y_data_test[i_idx_use,:]
#        y_data_test_noisefree = y_data_test_noisefree[i_idx_use,:]
#        x_data_test = x_data_test[i_idx_use,:]

    # Set posterior samples to None if posteriors don't exist
    elif params['pe_dir'] == None:
        XS_all = None

    # reshape y data into channels last format for convolutional approach (if requested)
    if params['n_filters_r1'] != None:
        y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(params['det'])))
        y_data_test_noisefree_copy = np.zeros((y_data_test_noisefree.shape[0],params['ndata'],len(params['det'])))
        y_data_train_copy = np.zeros((y_data_train.shape[0],params['ndata'],len(params['det'])))
        for i in range(y_data_test.shape[0]):
            for j in range(len(params['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
                y_data_test_noisefree_copy[i,:,j] = y_data_test_noisefree[i,idx_range]
        y_data_test = y_data_test_copy
        y_data_noisefree_test = y_data_test_noisefree_copy

        for i in range(y_data_train.shape[0]):
            for j in range(len(params['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)
                y_data_train_copy[i,:,j] = y_data_train[i,idx_range]
        y_data_train = y_data_train_copy

    # run hyperparameter optimization
    if params['hyperparam_optim'] == True:

        # list of initial default hyperparameters to use for GP hyperparameter optimization
        default_hyperparams = [5,
                       1,
                       1,
                       8,
                       1,
                       2,
                       11,
                       1,
                       1,
                       10,
                       1,
                       2,
                       10,
                       1,
                       1,
                       16,
                       16,
                       33,
                       33,
                       33,
                       33,
                       33,
                       32,
                       2048,
                       2048,
                       2048,
                       5,
                       3,
                       3,
                       3,
                       2,
                       2,
                      ]

        # Define hyperparam iteration counter
        global cur_hyperparam_iter
        cur_hyperparam_iter = 0

        params['plot_interval'] = 50000

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
        exit()

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
    """ Test a pre-trained neural network. There are several metrics by 
    which the user may test the efficiency of the model (e.g. KL divergence, 
    pp plots, corner plots).

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    use_gpu: bool
        If True, use a GPU to generate samples from the posterior of a pretrained neural network
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

    # if doing hour angle, use hour angle bounds on RA
    if params['convert_to_hour_angle']:
        bounds['ra_min'] = params['hour_angle_range'][0]
        bounds['ra_max'] = params['hour_angle_range'][1]

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
    x_data_test, y_data_test_noisefree, y_data_test,_,snrs_test,imp_info = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],load_condor=True)

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

        # remove any remaining resume files
        resume_files=glob.glob('%s_%s1/*.resume*' % (params['pe_dir'],i))
        filelist = [resume_files]
        for file_idx,file_type in enumerate(filelist):
            for file in file_type:
                os.remove(file)

        for j in range(1):
            input_dir = '%s_%s%d/' % (params['pe_dir'],i,j+1)
            if type("%s" % input_dir) is str:
                dataLocations = ["%s" % input_dir]

            filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))
            if len(filenames) < num_finished_post:
                sampler_loc = i + str(j+1)
                num_finished_post = len(filenames)


    # Assert user has the minimum number of test samples generated
    number_of_files_in_dir = len(os.listdir(dataLocations[0]))
    try:
        assert number_of_files_in_dir >= params['r']
    except Exception as e:
        print(e)
        print('You are requesting to use more GW time series than you have made.')
        exit()

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
        while i_idx < params['r']:

            filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)

            for samp_idx_inner in params['samplers'][1:]:
                inner_file_existance = True
                if samp_idx_inner == samp_idx:
                    inner_file_existance = os.path.isfile(filename)
                    if inner_file_existance == False:
                        break
                    else:
                        continue

                dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
                filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
                # If file does not exist, skip to next file
                inner_file_existance = os.path.isfile(filename_inner)                
                if inner_file_existance == False:
                    break

            if inner_file_existance == False:
                i+=1
#                print('File does not exist for one of the samplers')
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

            rand_idx_posterior = np.linspace(0,XS.shape[0]-1,num=params['n_samples'],dtype=np.int)
            np.random.shuffle(rand_idx_posterior)
            rand_idx_posterior = rand_idx_posterior[:params['n_samples']]
            # Append test sample posterior to existing array of test sample posteriors
            if i_idx == 0:
                XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
                #XS_all = np.expand_dims(XS[:params['n_samples'],:], axis=0)
            else:
                try:
                    XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))
                    #XS_all = np.vstack((XS_all,np.expand_dims(XS[:params['n_samples'],:], axis=0)))
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

    # reshape y data into channels last format for convolutional approach
    y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(params['det'])))
    if params['n_filters_r1'] != None:
        for i in range(y_data_test.shape[0]):
            for j in range(len(params['det'])):
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
    for i in range(params['r']):

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
                    levels=(0.50,0.90), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

        matplotlib.rc('text', usetex=True)                
        parnames = []
    
        # Get infered parameter latex labels for corner plot
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, params['inf_pars']):
                parnames.append(params['corner_labels'][k])

        # unnormalize the predictions from VICI (comment out if not wanted)
        color_cycle=['tab:blue','tab:green','tab:purple','tab:orange']
        legend_color_cycle=['blue','green','purple','orange']
        for q_idx,q in enumerate(params['inf_pars']):
                par_min = q + '_min'
                par_max = q + '_max'

                VI_pred[:,q_idx] = (VI_pred[:,q_idx] * (bounds[par_max] - bounds[par_min])) + bounds[par_min]

        # Convert hour angle back to RA
        VI_pred = convert_ra_to_hour_angle(VI_pred, params, rand_pars=False)

        # Apply importance sampling if wanted by user
        if args.importance_sampling:
            VI_pred = importance_sampling(fixed_vals, params, VI_pred, imp_info['all_par'][i], imp_info['file_IDs'][i])

        # Get allowed range values for all posterior samples
        max_min_samplers = dict(max = np.zeros((len(params['samplers']),len(params['inf_pars']))),
                                min = np.zeros((len(params['samplers']),len(params['inf_pars'])))
                                )
        for samp_idx,samp in enumerate(params['samplers'][1:]):
            for par_idx in range(len(params['inf_pars'])):
                max_min_samplers['max'][samp_idx+1,par_idx] = np.max(samp_posteriors[samp+'1'][i][:,par_idx])
                max_min_samplers['min'][samp_idx+1,par_idx] = np.min(samp_posteriors[samp+'1'][i][:,par_idx])
                if samp_idx == 0:
                    max_min_samplers['max'][0,par_idx] = np.min(VI_pred[:,par_idx])
                    max_min_samplers['min'][0,par_idx] = np.max(VI_pred[:,par_idx])
        corner_range = [ (np.min(max_min_samplers['min'][:,par_idx]), np.max(max_min_samplers['max'][:,par_idx]) ) for par_idx in range(len(params['inf_pars']))]

        # Iterate over all Bayesian PE samplers and plot results
        custom_lines = []
        truths = x_data_test_unnorm[i,:]
        for samp_idx,samp in enumerate(params['samplers'][1:]):

            bilby_pred = samp_posteriors[samp+'1'][i]

            # compute weights, otherwise the 1d histograms will be different scales, could remove this
#            weights = np.ones(len(VI_pred)) * (len(samp_posteriors[samp+'1'][i]) / len(VI_pred))
            weights = np.ones(len(VI_pred)) / len(VI_pred)
            if samp_idx == 0:
                figure = corner.corner(bilby_pred,**defaults_kwargs,labels=parnames,
                               color=color_cycle[samp_idx],
                               truths=truths, range=corner_range
                               )
            else:
                figure = corner.corner(bilby_pred,**defaults_kwargs,labels=parnames,
                               color=color_cycle[samp_idx],
                               truths=truths,
                               fig=figure, range=corner_range)
            custom_lines.append(Line2D([0], [0], color=legend_color_cycle[samp_idx], lw=4))

        # plot predicted ML results
        corner.corner(VI_pred, **defaults_kwargs, labels=parnames,
                           color='tab:red', fill_contours=True,
                           fig=figure, range=corner_range)
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
        corner_names = params['figure_sampler_names'][1:] + ['VItamin']
        figure.legend(handles=custom_lines, labels=corner_names,
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

    # Create dataset to save KL divergence results for later plotting
    try:
        os.mkdir('plotting_data_%s' % params['run_label'])
    except:
        print()
        print('... Plotting directory already exists')
        print()

    if params['make_loss_plot'] == True:
        plotter.plot_loss()

    if params['make_pp_plot'] == True:
        # Make pp plot
        plotter.plot_pp(CVAE_model,y_data_test,x_data_test,y_normscale,bounds)

    if params['make_kl_plot'] == True:    
        # Make KL plots
        plotter.gen_kl_plots(CVAE_model,y_data_test,x_data_test,y_normscale,bounds,snrs_test)

    # Make bilby pp plot
#    plotter.plot_bilby_pp(VICI_inverse_model,y_data_test,x_data_test,0,y_normscale,x_data_test,bounds)
#    exit()

    return

def gen_samples(params=params,bounds=bounds,fixed_vals=fixed_vals,model_loc='model_ex/model.ckpt',test_set='test_waveforms/',num_samples=None,plot_corner=True,use_gpu=False):
    """ Function to generate VItamin samples given a trained model

    Parameters
    ----------
    params: dict
        Dictionary containing run parameters
    bounds: dict
        Dictionary containing allowed bounds of GW source parameters
    fixed_vals: dict
        Dictionary containing the fixed values of GW source parameters
    model_loc: str
        location of pre-trained model (i.e. file with .ckpt)
    test_set: str
        dictionary location of test sample time series
    num_samples: float
        number of posterior samples to generate using neural network
    plot_corner: bool
        if true, make corner plots of generated posterior samples
    use_gpu: bool
        if true, use gpu to make posterior samples

    Returns
    -------
    samples: array_like
        posterior samples generated by neural network
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

    # if doing hour angle, use hour angle bounds on RA
    if params['convert_to_hour_angle']:
        bounds['ra_min'] = params['hour_angle_range'][0]
        bounds['ra_max'] = params['hour_angle_range'][1]

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

    # load generated samples
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
    y_data_test_copy = np.zeros((y_data_test.shape[0],params['ndata'],len(params['det'])))
    if params['n_filters_r1'] != None:
        for i in range(y_data_test.shape[0]):
            for j in range(len(params['det'])):
                idx_range = np.linspace(int(j*params['ndata']),int((j+1)*params['ndata'])-1,num=params['ndata'],dtype=int)       
                y_data_test_copy[i,:,j] = y_data_test[i,idx_range]
        y_data_test = y_data_test_copy
    num_timeseries=y_data_test.shape[0]
    

    """
    # Daniel Williams waveforms
    num_timeseries = 1
    f = open('daniel_williams_BNU_waveforms/hunter-mdc/0.json')
    data = json.load(f)
    y_data_test = np.zeros((1,params['ndata'],len(params['det'])))
    for det_idx,det in enumerate(params['det']):
        y_data_test[0,-(np.array(data['data'][det]).shape[0]):,det_idx] = np.array(data['data'][det])

    # Get x_data_test
    x_data_test = []
    for idx, i in enumerate(params['inf_pars']):
        if i == data['meta'][i]:
            x_data_test.append(data['meta'][i])
    x_data_test = np.array(x_data_test)
    print(x_data_test)
    exit()

    # whiten Daniel waveforms
    # Set up interferometers. These default to their design
    # sensitivity

    # define the start time of the timeseries
    start_time = params['ref_geocent_time']-params['duration']/2.0

    ifos = bilby.gw.detector.InterferometerList(params['det'])
    # set noise to be colored Gaussian noise
    ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=params['ndata'], duration=params['duration'],
    start_time=start_time)

    for det_idx,det in enumerate(params['det']):
        fft_waveform = np.fft.rfft(y_data_test[0,:,det_idx])
        whitened_signal = fft_waveform / ifos[det_idx].amplitude_spectral_density_array
        y_data_test[0,:,det_idx] = np.fft.irfft(whitened_signal) + np.random.normal(loc=0,scale=1.0,size=(params['ndata']))
    """

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

        # Convert hour angle back to RA
        samples[i,:] = convert_ra_to_hour_angle(samples[i,:], params, rand_pars=False)

        # plot results
        if plot_corner==True:
            # Get infered parameter latex labels for corner plot
            parnames=[]
            for k_idx,k in enumerate(params['rand_pars']):
                if np.isin(k, params['inf_pars']):
                    parnames.append(params['corner_labels'][k])
            # Define default corner plot arguments
            defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16), show_titles=False,
                    truth_color='black', quantiles=None,#[0.16, 0.84],
                    levels=(0.50,0.90), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)
            figure = corner.corner(samples[i,:,:],**defaults_kwargs,labels=parnames)
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
