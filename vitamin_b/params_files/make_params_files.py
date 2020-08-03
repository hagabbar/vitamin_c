import numpy as np
import os
import json

# Source parameter values to use if chosen to be fixed
fixed_vals = {'mass_1':50.0,
        'mass_2':50.0,
        'mc':None,
        'geocent_time':0.0,
        'phase':0.0,
        'ra':1.375,
        'dec':-1.2108,
        'psi':0.0,
        'theta_jn':0.0,
        'luminosity_distance':2000.0,
        'a_1':0.0,
        'a_2':0.0,
	'tilt_1':0.0,
	'tilt_2':0.0,
        'phi_12':0.0,
        'phi_jl':0.0,
        'det':['H1','L1','V1']}                                                 # feel free to edit this if more or less detectors wanted


# Prior bounds on source parameters
bounds = {'mass_1_min':35.0, 'mass_1_max':80.0,
        'mass_2_min':35.0, 'mass_2_max':80.0,
        'M_min':70.0, 'M_max':160.0,
        'geocent_time_min':0.15,'geocent_time_max':0.35,
        'phase_min':0.0, 'phase_max':2.0*np.pi,
        'ra_min':0.0, 'ra_max':2.0*np.pi,
        'dec_min':-0.5*np.pi, 'dec_max':0.5*np.pi,
        'psi_min':0.0, 'psi_max':np.pi,
        'theta_jn_min':0.0, 'theta_jn_max':np.pi,
        'a_1_min':0.0, 'a_1_max':0.0,
        'a_2_min':0.0, 'a_2_max':0.0,
        'tilt_1_min':0.0, 'tilt_1_max':0.0,
        'tilt_2_min':0.0, 'tilt_2_max':0.0,
        'phi_12_min':0.0, 'phi_12_max':0.0,
        'phi_jl_min':0.0, 'phi_jl_max':0.0,
        'luminosity_distance_min':1000.0, 'luminosity_distance_max':3000.0}

# arbitrary value for normalization of timeseries (Don't change this)
y_normscale = 36.43879218007172

##########################
# Main tunable variables
##########################
ndata = 256                                                                     # sampling frequency
rand_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','phase',
                 'theta_jn','psi','ra','dec']                                   # parameters to randomize (those not listed here are fixed otherwise)
inf_pars=['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','ra','dec'] # psi phase before ra and dec                     # parameters to infer
batch_size = 512                                                                 # Number training samples shown to neural network per iteration
weight_init = 'xavier'                                                         #[xavier,VarianceScaling,Orthogonal] # Network model weight initialization    
n_modes=7                                                                      # number of modes in Gaussian mixture model (ideal 7, but may go higher/lower)
initial_training_rate=1e-4                                                     # initial training rate for ADAM optimiser inference model (inverse reconstruction)
batch_norm=True                                                                # if true, do batch normalization in all layers of neural network

# FYI, each item in lists below correspond to each layer in networks (i.e. first item first layer)
# pool size and pool stride should be same number in each layer
n_filters_r1 = [33, 33, 33, 33, 33]                                                        # number of convolutional filters to use in r1 network (must be divisible by 3)
n_filters_r2 = [33, 33, 33]                                                        # number of convolutional filters to use in r2 network (must be divisible by 3)
n_filters_q = [33, 33, 33]                                                         # number of convolutional filters to use in q network  (must be divisible by 3)
filter_size_r1 = [5,8,11,10,10]                                                         # size of convolutional fitlers in r1 network
filter_size_r2 = [5,8,11]                                                         # size of convolutional filters in r2 network
filter_size_q = [5,8,11]                                                          # size of convolutional filters in q network
drate = 0.5                                                                    # dropout rate to use in fully-connected layers
maxpool_r1 = [1,2,1,2,1]                                                             # size of maxpooling to use in r1 network
conv_strides_r1 = [1,1,1,1,1]                                                        # size of convolutional stride to use in r1 network
pool_strides_r1 = [1,2,1,2,1]                                                        # size of max pool stride to use in r1 network
maxpool_r2 = [1,2,1]                                                             # size of max pooling to use in r2 network
conv_strides_r2 = [1,1,1]                                                        # size of convolutional stride in r2 network
pool_strides_r2 = [1,2,1]                                                        # size of max pool stride in r2 network
maxpool_q = [1,2,1]                                                              # size of max pooling to use in q network
conv_strides_q = [1,1,1]                                                         # size of convolutional stride to use in q network
pool_strides_q = [1,2,1]                                                         # size of max pool stride to use in q network
n_fc = 2048                                                                      # Number of neurons in fully-connected layers
z_dimension=10                                                                 # number of latent space dimensions of model 
n_weights_r1 = [n_fc,n_fc,n_fc]                                                     # number fully-connected layers of encoders and decoders in the r1 model (inverse reconstruction)
n_weights_r2 = [n_fc,n_fc,n_fc]                                                     # number fully-connected layers of encoders and decoders in the r2 model (inverse reconstruction)
n_weights_q = [n_fc,n_fc,n_fc]                                                      # number fully-connected layers of encoders and decoders q model
##########################
# Main tunable variables
##########################

#############################
# optional tunable variables
#############################
run_label = 'demo_%ddet_%dpar_%dHz_testing' % (len(fixed_vals['det']),len(rand_pars),ndata) # label of run
bilby_results_label = 'all_4_samplers'                                             # label given to bilby results directory
r = 2                                                                           # number (to the power of 2) of test samples to use for testing. r = 2 means you want to use 2^2 (i.e 4) test samples
pe_test_num = 256                                                               # total number of test samples available to use in directory
tot_dataset_size = int(1e5)                                                     # total number of training samples available to use
tset_split = int(1e3)                                                           # number of training samples in each training data file
save_interval = int(5e4)                                                        # number of iterations to save model and plot validation results corner plots
num_iterations=int(1e6)+1                                                       # total number of iterations before ending training of model
ref_geocent_time=1126259642.5                                                   # reference gps time (not advised to change this)
load_chunk_size = 1e4                                                           # Number of training samples to load in at a time.
samplers=['vitamin','dynesty']                                                  # Bayesian samplers to use when comparing ML results (vitamin is ML approach) dynesty,ptemcee,cpnest,emcee

# Directory variables
plot_dir="./results/%s" % run_label  # output directory to save results plots
train_set_dir='./training_sets_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(fixed_vals['det']),len(rand_pars),ndata,tot_dataset_size,tset_split) # location of training set
test_set_dir='./test_sets/%s/test_waveforms' % bilby_results_label                                                            # location of test set directory waveforms
pe_dir='./test_sets/%s/test' % bilby_results_label                                                                            # location of test set directory Bayesian PE samples
#############################
# optional tunable variables

# Function for getting list of parameters that need to be fed into the models
def get_params():

    # Define dictionary to store values used in rest of code 
    params = dict(
        make_corner_plots = True,                                               # if True, make corner plots
        make_kl_plot = True,                                                    # If True, go through kl plotting function
        make_pp_plot = True,                                                    # If True, go through pp plotting function
        make_loss_plot = False,                                                 # If True, generate loss plot from previous plot data
        Make_sky_plot=False,                                                    # If True, generate sky plots on corner plots
        hyperparam_optim = False,                                               # optimize hyperparameters for model during training using gaussian process minimization
        resume_training=False,                                                  # if True, resume training of a model from saved checkpoint
        load_by_chunks = True,                                                  # if True, load training samples by a predefined chunk size rather than all at once
        ramp = True,                                                            # if true, apply linear ramp to KL loss
        print_values=True,                                                      # optionally print loss values every report interval
        by_channel = True,                                                      # if True, do convolutions as seperate 1-D channels, if False, stack training samples as 2-D images (n_detectors,(duration*sampling_frequency))
        load_plot_data=False,                                                   # Plotting data which has already been generated
        doPE = True,                                                            # if True then do bilby PE when generating new testing samples (not advised to change this)
        gpu_num=0,                                                              # gpu number run is running on
        ndata = ndata,                                                          
        run_label=run_label,                                                    
        bilby_results_label=bilby_results_label,                                
        tot_dataset_size = tot_dataset_size,                                    
        tset_split = tset_split,                                                
        plot_dir=plot_dir,

        # Gaussian Process automated hyperparameter tunning variables
        hyperparam_optim_stop = int(1.5e6),                                     # stopping iteration of hyperparameter optimizer per call (ideally 1.5 million) 
        hyperparam_n_call = 30,                                                 # number of hyperparameter optimization calls (ideally 30)
        load_chunk_size = load_chunk_size,                                      
        load_iteration = int((load_chunk_size * 25)/batch_size),                # How often to load another chunk of training samples
        weight_init = weight_init,                                           
        n_samples = 10000,                                                       # number of posterior samples to save per reconstruction upon inference (default 3000) 
        num_iterations=num_iterations,                                              # total number of iterations before ending training of model
        initial_training_rate=initial_training_rate,                         
        batch_size=batch_size,                                                  
        batch_norm=batch_norm,                                                  
        report_interval=500,                                                    # interval at which to save objective function values and optionally print info during training
        n_modes=n_modes,                                                     

        # FYI, each item in lists below correspond to each layer in networks (i.e. first item first layer)
        # pool size and pool stride should be same number in each layer
        n_filters_r1 = n_filters_r1,                                         
        n_filters_r2 = n_filters_r2,                                         
        n_filters_q = n_filters_q,                                           
        filter_size_r1 = filter_size_r1,                                     
        filter_size_r2 = filter_size_r2,                                     
        filter_size_q = filter_size_q,                                       
        drate = drate,                                                       
        maxpool_r1 = maxpool_r1,                                             
        conv_strides_r1 = conv_strides_r1,                                   
        pool_strides_r1 = pool_strides_r1,                                   
        maxpool_r2 = maxpool_r2,                                             
        conv_strides_r2 = conv_strides_r2,                                   
        pool_strides_r2 = pool_strides_r2,                                   
        maxpool_q = maxpool_q,                                               
        conv_strides_q = conv_strides_q,                                     
        pool_strides_q = pool_strides_q,                                     
        ramp_start = 1e4,                                                       # starting iteration of KL divergence ramp (if using)
        ramp_end = 1e5,                                                         # ending iteration of KL divergence ramp (if using)
        save_interval=save_interval,                                            
        plot_interval=save_interval,                                            
        z_dimension=z_dimension,                                              
        n_weights_r1 = n_weights_r1,                                         
        n_weights_r2 = n_weights_r2,                                         
        n_weights_q = n_weights_q,                                           
        duration = 1.0,                                                         # length of training/validation/test sample time series in seconds (haven't tried using at any other value than 1s)
        r = r,                                                                  
        rand_pars=rand_pars,                                                    
        cornercorner_parnames = ['$m_{1}\,(\mathrm{M}_{\odot})$','$m_{2}\,(\mathrm{M}_{\odot})$','$d_{\mathrm{L}}\,(\mathrm{Mpc})$','$t_{0}\,(\mathrm{seconds})$','${\phi}$','$\Theta_{jn}\,(\mathrm{rad})$','${\psi}$',r'${\alpha}\,(\mathrm{rad})$','${\delta}\,(\mathrm{rad})$'], # latex source parameter labels for plotting
        ref_geocent_time=ref_geocent_time,                                      
        training_data_seed=43,                                                  # tensorflow training random seed number
        testing_data_seed=44,                                                   # tensorflow testing random seed number
        inf_pars=inf_pars,                                                      
        train_set_dir=train_set_dir,
        test_set_dir=test_set_dir,
        pe_dir=pe_dir,
        KL_cycles = 1,                                                          # number of cycles to repeat for the KL approximation
        samplers=samplers,                                                      
        figure_sampler_names = ['VItamin','Dynesty'],#,'ptemcee','CPNest','emcee'],
        y_normscale = y_normscale,
        boost_pars=['ra','dec'],
        gauss_pars=['luminosity_distance','geocent_time','theta_jn'],        # parameters that require a truncated gaussian 
        vonmise_pars=['phase','psi'],                                        # parameters that get wrapped on the 1D parameter 
        sky_pars=['ra','dec'],                                               # sky parameters
        weighted_pars=None,#['ra','dec','geocent_time'],                     # set to None if not using, parameters to weight during training
        weighted_pars_factor=1,                       # Factor by which to weight parameters if `weighted_pars` is not None.
    )
    return params


# Save training/test parameters of run if files do not already exist
params=get_params()

# Make directory containing params files
try:
    os.mkdir('params_files')
except FileExistsError:
    print("directory 'params_files' already exists")

# Generate params files
if not os.path.isfile('./params_files/params.txt'):
    
    # save params json file
    with open('params_files/params.json', 'w') as fp:
        json.dump(params, fp, sort_keys=True, indent=4)

if not os.path.isfile('./params_files/bounds.txt'):

    # save bounds json file
    with open('params_files/bounds.json', 'w') as fp:
        json.dump(bounds, fp, sort_keys=True, indent=4)
if not os.path.isfile('./params_files/fixed_vals.txt'):

    # save fixed vals json file
    with open('params_files/fixed_vals.json', 'w') as fp:
        json.dump(fixed_vals, fp, sort_keys=True, indent=4)
