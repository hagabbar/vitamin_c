""" Script used to generate run parameters json files
"""

import numpy as np
import os
import json

# Source parameter values to use if chosen to be fixed
fixed_vals = {'mass_1':50.0,
        '__definition__mass_1':'m1 fixed value',
        'mass_2':50.0,
        '__definition__mass_2':'m2 fixed value',
        'mc':None,
        '__definition__mc': 'chirp mass fixed value',
        'geocent_time':0.0,
        '__definition__geocent_time': 'time of coalescence fixed value',
        'phase':0.0,
        '__definition__phase': 'phase fixed value',
        'ra':1.375,
        '__definition__ra': 'right ascension fixed value',
        'dec':-1.2108,
        '__definition__dec': 'declination fixed value',
        'psi':0.0,
        '__definition__psi': 'psi fixed value',
        'theta_jn':0.0,
        '__definition__theta_jn': 'inclination angle fixed value',
        'luminosity_distance':2000.0,
        '__definition__luminosity_distance': 'luminosity distance fixed value',
        'a_1':0.0,
        '__definition__a1': 'a1 fixed value',
        'a_2':0.0,
        '__definition__a2': 'a2 fixed value',
	    'tilt_1':0.0,
        '__definition__tilt1': 'tilt_1 fixed value',
	    'tilt_2':0.0,
        '__definition__tilt_2': 'tilt_2 fixed value',
        'phi_12':0.0,
        '__definition__phi_12': 'phi_12 fixed value',
        'phi_jl':0.0,
        '__definition__phi_jl': 'phi_jl fixed value'}                                                

# Prior bounds on source parameters
bounds = {'mass_1_min':35.0, 'mass_1_max':80.0,
        '__definition__mass_1': 'mass 1 range',
        'mass_2_min':35.0, 'mass_2_max':80.0,
        '__definition__mass_2': 'mass 2 range',
        'M_min':70.0, 'M_max':160.0,
        '__definition__M': 'total mass range',
        'geocent_time_min':0.15,'geocent_time_max':0.35,
        '__definition__geocent_time': 'time of coalescence range',
        'phase_min':0.0, 'phase_max':2.0*np.pi,
        '__definition__phase': 'phase range',
        'ra_min':0.0, 'ra_max':2.0*np.pi,
        '__definition__ra': 'right ascension range',
        'dec_min':-0.5*np.pi, 'dec_max':0.5*np.pi,
        '__definition__dec': 'declination range',
        'psi_min':0.0, 'psi_max':np.pi,
        '__definition__psi': 'psi range',
        'theta_jn_min':0.0, 'theta_jn_max':np.pi,
        '__definition__theta_jn': 'inclination angle range',
        'a_1_min':0.0, 'a_1_max':0.8,
        '__definition__a_1': 'a1 range',
        'a_2_min':0.0, 'a_2_max':0.8,
        '__definition__a_2': 'a2 range',
        'tilt_1_min':0.0, 'tilt_1_max':np.pi,
        '__definition__tilt1': 'tilt1 range',
        'tilt_2_min':0.0, 'tilt_2_max':np.pi,
        '__definition__tilt2': 'tilt2 range',
        'phi_12_min':0.0, 'phi_12_max':2.0*np.pi,
        '__definition__phi12': 'phi12 range',
        'phi_jl_min':0.0, 'phi_jl_max':2.0*np.pi,
        '__definition_phijl': 'phijl range',
        'luminosity_distance_min':1000.0, 'luminosity_distance_max':3000.0,
        '__definition__luminosity_distance': 'luminosity distance range'}

# arbitrary value for normalization of timeseries (Don't change this)
y_normscale = 36.0

##########################
# Main tunable variables
##########################
ndata = 1024                                                                     
det=['H1','L1','V1']                                                            
#psd_files=['cuda_11_env/lib/python3.6/site-packages/bilby/gw/detector/noise_curves/aLIGO_O4_high_asd.txt'] 
psd_files=[]
rand_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','phase',
                 'theta_jn','psi','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','ra','dec']                                   
bilby_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','psi','a_1','a_2',
          'tilt_1','tilt_2','phi_12','phi_jl','ra','dec']
inf_pars=['mass_1','mass_2','luminosity_distance','geocent_time','phase',
                 'theta_jn','psi','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl','ra','dec']
#rand_pars = ["mass_1","mass_2","luminosity_distance","geocent_time","phase","theta_jn","psi","ra","dec","a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
#bilby_pars = ["mass_1","mass_2","luminosity_distance","geocent_time","theta_jn","psi","ra","dec","a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
#inf_pars = ["mass_1","mass_2","luminosity_distance","geocent_time","phase","theta_jn","psi","ra","dec","a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
batch_size = int(512)                                                                 
weight_init = 'xavier'                                                            
n_modes=32; n_modes_q=1                                                                      
initial_training_rate=1e-4                                                     
batch_norm=False                                                                

# FYI, each item in lists below correspond to each layer in networks (i.e. first item first layer)
# pool size and pool stride should be same number in each layer
n_filters_r1 = [32,32,32,32]                                                        
n_filters_r2 = [32,32,32,32]                                                        
n_filters_q = [32,32,32,32]                                                         
filter_size_r1 = [5,5,5,5]                                                         
filter_size_r2 = [5,5,5,5]                                                        
filter_size_q = [5,5,5,5]                                                          
drate = 0.0                                                                    
maxpool_r1 = [1,2,1,2]                                                             
conv_strides_r1 = [1,1,1,1]                                                        
pool_strides_r1 = [1,2,1,2]
conv_dilations_r1=[1,1,1,1]                                                        
maxpool_r2 = [1,2,1,2]                                                             
conv_strides_r2 = [1,1,1,1]                                                        
pool_strides_r2 = [1,2,1,2]
conv_dilations_r2=[1,1,1,1]                                                        
maxpool_q = [1,2,1,2]                                                              
conv_strides_q = [1,1,1,1]                                                         
pool_strides_q = [1,2,1,2] 
conv_dilations_q=[1,1,1,1]                                                        
n_fc = 1024                                                                      
z_dimension=15                                                                  
n_weights_r1 = [n_fc,n_fc,n_fc]                                                     
n_weights_r2 = [n_fc,n_fc,n_fc]                                                     
n_weights_q = [n_fc,n_fc,n_fc]                                                      
##########################
# Main tunable variables
##########################

#############################
# optional tunable variables
#############################
run_label = 'vitamin_c_run53'#'vitamin_c_run31_fast_testing'#'demo_%ddet_%dpar_%dHz_hour_angle_with_late_kl_start' % (len(det),len(rand_pars),ndata) 
gpu_num = 2

# 1024 Hz label
#bilby_results_label = 'weichangfeng_theta_jn_issue'                                             
# 256 Hz label
bilby_results_label = '1024Hz_full_15par'

r = 2 # 251                                                         
pe_test_num = 256                                                               
tot_dataset_size = int(1e7)                                                     

tset_split = int(1e3)                                                           
save_interval = int(1000)
plot_interval = int(1000)                                                        
num_iterations=int(100000)+1 # 12500 for Green et al.                                                       
ref_geocent_time=1126259642.5                                                   
load_chunk_size = int(2e4)                                                           
samplers=['vitamin','dynesty','ptemcee']#,'cpnest','emcee']                                                  
val_dataset_size = int(1e3)

# Directory variables
plot_dir='/home/hunter.gabbard/public_html/CBC/chris_dec2020_vitamin/%s' % run_label  

# Training/testing for 1024 Hz full par case
#train_set_dir='/home/hunter.gabbard/CBC/public_VItamin/provided_models/vitamin_b/vitamin_b/training_sets_3det_15par_1024Hz/tset_tot-10000000_split-1000_O4PSDH1L1_AdvVirgoPSD'
#val_set_dir='./validation_sets_3det_15par_1024Hz/tset_tot-1000_split-1000'
#test_set_dir = '/home/hunter.gabbard/CBC/public_VItamin/provided_models/vitamin_b/vitamin_b/test_sets/1024_khz_spins_included_15par/test_waveforms'
#pe_dir='/home/hunter.gabbard/CBC/public_VItamin/provided_models/vitamin_b/vitamin_b/test_sets/1024_khz_spins_included_15par/test'

# default training/testing directories
train_set_dir='./training_sets_TiltSineDist_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(det),len(rand_pars),ndata,tot_dataset_size,tset_split)
val_set_dir='./validation_sets_TiltSineDist_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(det),len(rand_pars),ndata,val_dataset_size,tset_split) 
test_set_dir = './test_sets/%s/test_waveforms' % bilby_results_label
pe_dir='./test_sets/%s/test' % bilby_results_label

#############################
# optional tunable variables

# Function for getting list of parameters that need to be fed into the models
def get_params():

    # Define dictionary to store values used in rest of code 
    params = dict(
        val_dataset_size = val_dataset_size,
        __definition__val_dataset_size='validation dataset total size',
        bilby_pars=bilby_pars,
        __definition__bilby_pars='parameters for bilby to infer',
        vitamin_c = True,
        __definition__vitamin_c='If True, use new vitamin_c version of code with keras',
        parallel_conv=False,
        __definition__parallel_conv='if true analyse detectors separately in convolutions, otherwise as channels',
        extra_lr_decay_factor=False,
        __definition__extra_decay_factor='Use an extra decay factor of 0.96 after every load iteration',
        n_KLzsamp=1,
        __definition_n_KLzsamp='The number of samples to use to perfrom MonteCarlo integration over z to compute the KL',
        twod_conv=False,
        __definition__twod_conv='if true treat the detectors as an extra image dimension otherwise as channels',
        n_modes_q=n_modes_q,
        __definition__n_modes_q='number of modes in Gaussian mixture model for the q distribution',
        conv_dilations_r1=conv_dilations_r1,
        __definition__conv_dilations_r1='size of convolutional dilation to use in r1 network',
        conv_dilations_r2=conv_dilations_r2,
        __definition__conv_dilations_r2='size of convolutional dilation to use in r2 network',
        conv_dilations_q=conv_dilations_q,
        __definition__conv_dilations_q='size of convolutional dilation to use in q network', 
        hour_angle_range=[-3.813467684252483,2.469671758574231],
        __definition__hour_angle_range='min and max range of hour angle space',
        use_real_det_noise=False,
        __definition__use_real_det_noise='If True, use real detector noise around reference time',
        real_noise_time_range = [1126051217, 1137254417],
        __definition__real_noise_time_range = 'time range to produce real noise samples over',
        use_real_events=False, # e.g. 'GW150914'
        __definition__use_real_events='list containing real ligo events to be analyzed',
        convert_to_hour_angle = False,
        __definition__convert_to_hour_angle='If True, convert RA to hour angle during trianing and testing',
        make_corner_plots = True,                                               
        __definition__make_corner_plots='if True, make corner plots',
        make_kl_plot = False,                                                    
        __definition__make_kl_plot='If True, go through kl plotting function',
        make_pp_plot = False,                                                    
        __definition__make_pp_plot='If True, go through pp plotting function',
        make_loss_plot = True,                                                 
        __definition__make_loss_plot='If True, generate loss plot from previous plot data',
        Make_sky_plot=False,                                                    
        __definition__make_sky_plot='If True, generate sky plots on corner plots',
        hyperparam_optim = False,                                               
        __definition__hyperparam_optim='optimize hyperparameters for model during training using gaussian process minimization',
        resume_training=False,                                                  
        __definition___resume_training='if True, resume training of a model from saved checkpoint',
        load_by_chunks = True,                                                  
        __definition__load_by_chunks='if True, load training samples by a predefined chunk size rather than all at once',	
        ramp = True,                                                            
        __definition__ramp='# if true, apply linear ramp to KL loss',
        print_values=True,                                                      
        __definition__print_values='# optionally print loss values every report interval',
        by_channel = True,                                                      
        __definition__by_channel='if True, do convolutions as seperate 1-D channels, if False, stack training samples as 2-D images (n_detectors,(duration*sampling_frequency))',
        load_plot_data = False,
        __definition__load_plot_data='Use plotting data which has already been generated',
        doPE = True,                                                            
        __definition__doPE='if True then do bilby PE when generating new testing samples',
        gpu_num=gpu_num,                                                              
        __definition__gpu_num='gpu number run is running on',
        ndata = ndata,                                                          
        __definition__ndata='sampling frequency',
        run_label=run_label,                                                    
        __definition__run_label='label of run',
        bilby_results_label=bilby_results_label,                                
        __definition__bilby_result_label='label given to bilby results directory',
        tot_dataset_size = tot_dataset_size, 
        __definition__tot_dataset_size='total number of training samples available to use',                                   
        tset_split = tset_split,
        __definition__tset_split='number of training samples in each training data file',                                                
        plot_dir=plot_dir,
        __definition__plot_dir='output directory to save results plots',

        # Gaussian Process automated hyperparameter tunning variables
        hyperparam_optim_stop = int(5e5),                                     
        __definition__hyperparam_optim_stop='stopping iteration of hyperparameter optimizer per call', 
        hyperparam_n_call = 30,                                                 
        __definition__hyperparam_n_call='number of hyperparameter optimization calls',
        
        load_chunk_size = load_chunk_size,
        __definition__load_chunk_size='Number of training samples to load in at a time.',                                      
        load_iteration = int((load_chunk_size * 25)/batch_size),                
        __definition__load_iteration='How often to load another chunk of training samples',
        weight_init = weight_init,
        __definition__weight_init='[xavier,VarianceScaling,Orthogonal]. Network model weight initialization (default is xavier)',                                           
        n_samples = 8000,                                                        
        __definition__n_samples='number of posterior samples to save per reconstruction upon inference (default 3000)',
        num_iterations=num_iterations,                                              
        __definition__num_iterations='total number of iterations before ending training of model',
        initial_training_rate=initial_training_rate,
        __definition__initial_training_rate='initial training rate for ADAM optimiser inference model (inverse reconstruction)',                         
        batch_size=batch_size,
        __definition__batch_size=' Number training samples shown to neural network per iteration',                                                  
        batch_norm=batch_norm,
        __definition__batch_norm='if true, do batch normalization in all layers of neural network',                                                  
        report_interval=500,                                                    
        __definition__report_interval='interval at which to save objective function values and optionally print info during training',
        n_modes=n_modes,
        __definition__n_modes='number of modes in Gaussian mixture model (ideal 7, but may go higher/lower)',                                                     

        # FYI, each item in lists below correspond to each layer in networks (i.e. first item first layer)
        # pool size and pool stride should be same number in each layer
        n_filters_r1 = n_filters_r1,
        __definition__n_filters_r1='number of convolutional filters to use in r1 network (must be divisible by 3)',                                         
        n_filters_r2 = n_filters_r2,
        __definition__n_filters_r2='number of convolutional filters to use in r2 network (must be divisible by 3)',                                         
        n_filters_q = n_filters_q,
        __definition__n_filters_q='number of convolutional filters to use in q network  (must be divisible by 3)',                                           
        filter_size_r1 = filter_size_r1,  
        __definition__filter_size_r1='size of convolutional fitlers in r1 network',                                   
        filter_size_r2 = filter_size_r2,   
        __definition__filter_size_r2='size of convolutional filters in r2 network',                                  
        filter_size_q = filter_size_q,
        __definition__filter_size_q='size of convolutional filters in q network',                                       
        drate = drate,  
        __definition__drate='dropout rate to use in fully-connected layers',                                                     
        maxpool_r1 = maxpool_r1,                                            
        __definition__maxpool_r1='size of maxpooling to use in r1 network', 
        conv_strides_r1 = conv_strides_r1,
        __definition__conv_strides_r1='size of convolutional stride to use in r1 network',                                   
        pool_strides_r1 = pool_strides_r1,                                   
        __definition_pool_strides_r1='size of max pool stride to use in r1 network',
        maxpool_r2 = maxpool_r2,       
        __definition_maxpool_r2='size of max pooling to use in r2 network',                                      
        conv_strides_r2 = conv_strides_r2,
        __definition__conv_strides_r2='size of convolutional stride in r2 network',                                   
        pool_strides_r2 = pool_strides_r2,                                   
        __definition__pool_strides_r2='size of max pool stride in r2 network',
        maxpool_q = maxpool_q,  
        __definition__maxpool_q='size of max pooling to use in q network',                                             
        conv_strides_q = conv_strides_q,  
        __definition__conv_strides_q='size of convolutional stride to use in q network',                                   
        pool_strides_q = pool_strides_q,                                     
        __definition__pool_strides_q='size of max pool stride to use in q network',
        ramp_start = 100,                                                       
        __definition__ramp_start='starting iteration of KL divergence ramp',
        ramp_end = 200,                                                  
        __definition__ramp_end='ending iteration of KL divergence ramp',
        save_interval=save_interval, 
        __definition__save_interval='number of iterations to save model',                                           
        plot_interval=plot_interval,                                     
        __definition__plot_interval='number of iterations to plot validation results corner plots',       
        z_dimension=z_dimension,                                              
        __definition__z_dimension='number of latent space dimensions of model',
        n_weights_r1 = n_weights_r1,                                         
        __definition__n_weights_r1='number fully-connected neurons in layers of encoders and decoders in the r1 model',
        n_weights_r2 = n_weights_r2,
        __definition__n_weights_r2='number fully-connected neurons in layers of encoders and decoders in the r2 model',                                         
        n_weights_q = n_weights_q,                                           
        __definition__n_weights_q='number fully-connected neurons in layers of encoders and decoders in the q model',
        duration = 1.0,                                                         
        __definition__duration='length of training/validation/test sample time series in seconds',
        r = r,     
        __definition__r='number of GW timeseries to use for testing.',                                                             
        rand_pars=rand_pars,                                                    
        __definition__rand_pars='parameters to randomize (those not listed here are fixed otherwise)',
        corner_labels = {'mass_1': '$m_{1}\,(\mathrm{M}_{\odot})$',
                         'mass_2': '$m_{2}\,(\mathrm{M}_{\odot})$',
                         'luminosity_distance': '$d_{\mathrm{L}}\,(\mathrm{Mpc})$',
                         'geocent_time': '$t_{0}\,(\mathrm{seconds})$',
                         'phase': '${\phi}$',
                         'theta_jn': '$\Theta_{jn}\,(\mathrm{rad})$',
                         'psi': '${\psi}$',
                         'ra': r'${\alpha}\,(\mathrm{rad})$',
                         'dec': '${\delta}\,(\mathrm{rad})$',
                         'a_1': '${a_1}$',
                         'a_2': '${a_2}$',
                         'tilt_1': '${\Theta_{1}}$',
                         'tilt_2': '${\Theta_{2}}$',
                         'phi_12': '${\phi_{12}}$',
                         'phi_jl': '${\phi_{jl}}$'},

        __definition__corner_labels='latex source parameter labels for plotting',
        ref_geocent_time=ref_geocent_time,                                      
        __definition__ref_geocent_time='reference gps time (not advised to change this)',
        training_data_seed=43,                                                 
        __definition__training_data_seed='tensorflow training random seed number',
        testing_data_seed=44,                                                   
        __definition__testing_data_seed=' tensorflow testing random seed number',
        validation_data_seed=45,
        __definition__validation_data_seed='Seed for validation data',
        inf_pars=inf_pars,        
        __definition__inf_pars='parameters to infer',                                              
        train_set_dir=train_set_dir,
        __definition__train_set_dir='location of training set directory',
        val_set_dir=val_set_dir,
        __definition__val_set_dir='location of validation set directory',
        test_set_dir=test_set_dir,
        __definition__test_set_dir='location of test set directory waveforms',
        pe_dir=pe_dir,
        __definition__pe_dir='location of test set directory Bayesian PE samples. Set to None if you do not want to use posterior samples when training.',
        KL_cycles = 1,                                                          
        __definition__KL_cycles=' number of cycles to repeat for the KL approximation',
        samplers=samplers,       
        __definition__samplers='Samplers to use when comparing ML results (vitamin is ML approach) dynesty,ptemcee,cpnest,emcee',                                               
        figure_sampler_names = ['VItamin', 'Dynesty', 'Ptemcee', 'CPNest', 'Emcee'],
        __definition__figure_sampler_names='matplotlib figure sampler labels (e.g. [ptemcee,CPNest,emcee])',
        y_normscale = y_normscale,
        __definition__y_normscale='arbitrary normalization factor on all time series waveforms (helps convergence in training)',
        gauss_pars=['luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2','ra','dec'],         
        __definition__gauss_pars='parameters that require a truncated gaussian distribution',
        vonmise_pars=['phase','phi_12','phi_jl','psi'],                                        
        __definition__vonmises_pars='parameters that get wrapped on the 1D parameter', 
        sky_pars=['ra','dec'],                                              
        __definition__sky_pars='sky parameters',
        det=det,
        __definition__det='LIGO detectors to perform analysis on (default is 3detector H1,L1,V1)',
        psd_files=psd_files,
        __definition__psd_files='User may specficy their own psd files for each detector. Must be bilby compatible .txt files. If specifying your own files, do so for each detector. If list is empty, default Bilby PSD values will be used for each detector. ',
        weighted_pars=None,
        __definition__weighted_pars='set to None if not using, parameters to weight during training',
        weighted_pars_factor=1,                       
        __definition__weighted_pars_factor='factor by which to weight weighted parameters',
    )
    return params


# Save training/test parameters of run if files do not already exist
params=get_params()

# Make directory containing params files
try:
    os.mkdir('params_files')
except FileExistsError:
    print("directory 'params_files' already exists")
    print("Overwriting .json files in directory ...")

# Generate params files
#if not os.path.isfile('./params_files/params.json'):
    
# save params json file
with open('params_files/params.json', 'w') as fp:
    json.dump(params, fp, sort_keys=False, indent=4)

#if not os.path.isfile('./params_files/bounds.json'):

# save bounds json file
with open('params_files/bounds.json', 'w') as fp:
    json.dump(bounds, fp, sort_keys=False, indent=4)

#if not os.path.isfile('./params_files/fixed_vals.json'):

# save fixed vals json file
with open('params_files/fixed_vals.json', 'w') as fp:
    json.dump(fixed_vals, fp, sort_keys=False, indent=4)
