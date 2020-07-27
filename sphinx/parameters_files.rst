=================================
Customizing Your Parameters Files
=================================

VItamin allows the user to tailor their run to a reasonable degree. 
Some work still needs to be done to improve the customizability. 

First, import vitamin to produce the default parameters files

.. code-block:: console

   $ import vitamin

You will now see that there are three text files in your current directory 
(params.txt,bounds.txt,fixed_vals.txt). The params.txt file concerns 
the run parameters which make your run unique (batch size, parameters to infer 
etc.). The bounds.txt file concerns the allowed numerical bounds for the 
source parameters to be infered. The fixed_vals.txt file dictates the 
value at which source parameters are fixed if they are chosen to be 
fixed by the user.

To change the default values for each of these values, use your 
text editor of choice (e.g. vim).

Details about the customizable values for each file are listed below.

The bounds file:

.. code-block:: console

   # Prior bounds on source parameters
   'mass_1_min':35.0, 'mass_1_max':80.0,                               # m1 range
   'mass_2_min':35.0, 'mass_2_max':80.0,                               # m2 range
   'M_min':70.0, 'M_max':160.0,                                        # total mass range
   'geocent_time_min':0.15,'geocent_time_max':0.35,                    # time of coalescence range
   'phase_min':0.0, 'phase_max':2.0*np.pi,                             # phase range
   'ra_min':0.0, 'ra_max':2.0*np.pi,                                   # right ascension range
   'dec_min':-0.5*np.pi, 'dec_max':0.5*np.pi,                          # declintation range
   'psi_min':0.0, 'psi_max':np.pi,                                     # psi range
   'theta_jn_min':0.0, 'theta_jn_max':np.pi,                           # inclination angle range
   'a_1_min':0.0, 'a_1_max':0.0,                                       # a1 range
   'a_2_min':0.0, 'a_2_max':0.0,                                       # a2 range
   'tilt_1_min':0.0, 'tilt_1_max':0.0,                                 # tilt1 range
   'tilt_2_min':0.0, 'tilt_2_max':0.0,                                 # tilt2 range
   'phi_12_min':0.0, 'phi_12_max':0.0,                                 # phi_l2  range
   'phi_jl_min':0.0, 'phi_jl_max':0.0,                                 # phi_jl range
   'luminosity_distance_min':1000.0, 'luminosity_distance_max':3000.0  # luminosity distance range

The fixed values file:

.. code-block:: console

   'mass_1':50.0,                                                      # m1 default fixed value
   'mass_2':50.0,                                                      # m2 default fixed value
   'mc':None,                                                          # chirp mass default fixed value
   'geocent_time':0.0,                                                 # time of coalescence default fixed value
   'phase':0.0,                                                        # phase default fixed value
   'ra':1.375,                                                         # right ascension default fixed value
   'dec':-1.2108,                                                      # declination default fixed value
   'psi':0.0,                                                          # psi default fixed value
   'theta_jn':0.0,                                                     # inclination angle default fixed value
   'luminosity_distance':2000.0,                                       # luminosity distance default fixed value
   'a_1':0.0,                                                          # a1 default fixed value
   'a_2':0.0,                                                          # a2 default fixed value
   'tilt_1':0.0,                                                       # tilt1 default fixed value
   'tilt_2':0.0,                                                       # tilt2 default fixed value
   'phi_12':0.0,                                                       # phi_l2 default fixed value
   'phi_jl':0.0,                                                       # phi_jl default fixed value
   'det':['H1','L1','V1']                                              # Edit this if more or less detectors wanted

The run parameters file:

.. code-block:: console

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
   ndata = 256,                                                            # sampling frequency
   run_label = 'demo_%ddet_%dpar_%dHz_run1' % (len(fixed_vals['det']),len(rand_pars),ndata) # label of run
   bilby_results_label = 'all_4_samplers'                                  # label given to bilby results directory
   tot_dataset_size = int(1e3)                                             # total number of training samples available to use
   tset_split = int(1e3)                                                   # number of training samples in each training data file
   plot_dir="./results/%s" % run_label  # output directory to save results plots

   # Gaussian Process automated hyperparameter tunning variables
   hyperparam_optim_stop = int(1.5e6),                                     # stopping iteration of hyperparameter optimizer per call (ideally 1.5 million) 
   hyperparam_n_call = 30,                                                 # number of hyperparameter optimization calls (ideally 30)

   load_chunk_size = 1e3                                                   # Number of training samples to load in at a time.
   load_iteration = int((load_chunk_size * 25)/batch_size),                # How often to load another chunk of training samples
   weight_init = 'xavier'                                                  #[xavier,VarianceScaling,Orthogonal] # Network model weight initialization
   n_samples = 100,                                                        # number of posterior samples to save per reconstruction upon inference (default 3000) 
   num_iterations=int(5e3)+1,                                              # total number of iterations before ending training of model
   initial_training_rate=1e-4                                              # initial training rate for ADAM optimiser inference model (inverse reconstruction)
   batch_size = 64                                                         # Number training samples shown to neural network per iteration
   batch_norm=True                                                         # if true, do batch normalization in all layers of neural network
   report_interval=500,                                                    # interval at which to save objective function values and optionally print info during training
   n_modes=7                                                               # number of modes in Gaussian mixture model (ideal 7, but may go higher/lower)

   # Each item in lists below correspond to each layer in networks (i.e. first item first layer)
   # pool size and pool stride should be same number in each layer
   n_filters_r1 = [33, 33]                                                 # number of convolutional filters to use in r1 network
   n_filters_r2 = [33, 33]                                                 # number of convolutional filters to use in r2 network
   n_filters_q = [33, 33]                                                  # number of convolutional filters to use in q network
   filter_size_r1 = [7,7]                                                  # size of convolutional fitlers in r1 network
   filter_size_r2 = [7,7]                                                  # size of convolutional filters in r2 network
   filter_size_q = [7,7]                                                   # size of convolutional filters in q network
   drate = 0.5                                                             # dropout rate to use in fully-connected layers
   maxpool_r1 = [1,2]                                                      # size of maxpooling to use in r1 network
   conv_strides_r1 = [1,1]                                                 # size of convolutional stride to use in r1 network
   pool_strides_r1 = [1,2]                                                 # size of max pool stride to use in r1 network
   maxpool_r2 = [1,2]                                                      # size of max pooling to use in r2 network
   conv_strides_r2 = [1,1]                                                 # size of convolutional stride in r2 network
   pool_strides_r2 = [1,2]                                                 # size of max pool stride in r2 network
   maxpool_q = [1,2]                                                       # size of max pooling to use in q network
   conv_strides_q = [1,1]                                                  # size of convolutional stride to use in q network
   pool_strides_q = [1,2]                                                  # size of max pool stride to use in q network
   ramp_end = 1e5,                                                         # ending iteration of KL divergence ramp (if using)
   save_interval = int(1e3)                                                # number of iterations to save model and plot validation results corner plots
   plot_interval=int(1e3),                                                 # interval over which to generate plots during training
   z_dimension=10                                                          # number of latent space dimensions of model
   n_weights_r1 = [2048,2048,2048]                                         # number fully-connected layers of encoders and decoders in the r1 model (inverse reconstruction)
   n_weights_r2 = [2048,2048,2048]                                         # number fully-connected layers of encoders and decoders in the r2 model (inverse reconstruction)
   n_weights_q = [2048,2048,2048]                                          # number fully-connected layers of encoders and decoders q model
   duration = 1.0,                                                         # length of training/validation/test sample time series in seconds (haven't tried using at any other value than 1s)
   r = 2                                                                   # number (to the power of 2) of test samples to use for testing. r = 2 means you want to use 2^2 (i.e 4) test samples
   rand_pars = ['mass_1','mass_2','luminosity_distance','geocent_time','phase',
                 'theta_jn','psi','ra','dec']                              # parameters to randomize (those not listed here  are fixed otherwise)
   corner_parnames = ['m_{1}\,(\mathrm{M}_{\odot})','m_{2}\,(\mathrm{M}_{\odot})','d_{\mathrm{L}}\,(\mathrm{Mpc})','t_{0}\,(\mathrm{seconds})','{\phi}','\Theta_{jn}\,(\mathrm{rad})','{\psi}',r'{\alpha}\,(\mathrm{rad})','{\delta}\,(\mathrm{rad})'], # latex source parameter labels for plotting
   cornercorner_parnames = ['$m_{1}\,(\mathrm{M}_{\odot})$','$m_{2}\,(\mathrm{M}_{\odot})$','$d_{\mathrm{L}}\,(\mathrm{Mpc})$','$t_{0}\,(\mathrm{seconds})$','${\phi}$','$\Theta_{jn}\,(\mathrm{rad})$','${\psi}$',r'${\alpha}\,(\mathrm{rad})$','${\delta}\,(\mathrm{rad})$'], # latex source parameter labels for plotting
   ref_geocent_time=1126259642.5                                           # reference gps time
   training_data_seed=43,                                                  # tensorflow training random seed number
   testing_data_seed=44,                                                   # tensorflow testing random seed number
   inf_pars=['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','ra','dec'] # psi phase before ra and dec
                                                                           # parameters to infer
   train_set_dir='./training_sets_%ddet_%dpar_%dHz/tset_tot-%d_split-%d' % (len(fixed_vals['det']),len(rand_pars),ndata,tot_dataset_size,tset_split)                                                        # location of training set
   test_set_dir='./test_sets/%s/test_waveforms' % bilby_results_label      # location of test set directory waveforms
   pe_dir='./test_sets/%s/test' % bilby_results_label                      # location of test set directory Bayesian PE samples
   KL_cycles = 1,

   samplers=['vitamin','dynesty']                                          # Bayesian samplers to use when comparing ML results (vitamin is ML approach) dynesty,ptemcee,cpnest,emcee
   figure_sampler_names = ['VItamin','Dynesty'],                           # Labels for samplers in figures 
   y_normscale = 36.43879218007172,                                        # arbitrary scaling value for waveforms (make sure it is safe b/w testing and training of model)
   boost_pars=['ra','dec'],                                                # Parameters to provide extra boosting scaler factor to loss during training.
   gauss_pars=['luminosity_distance','geocent_time','theta_jn'],           # parameters that require a truncated gaussian      
   vonmise_pars=['phase','psi'],                                           # parameters that get wrapped on the 1D parameter   
   sky_pars=['ra','dec'],                                                  # sky parameters
   weighted_pars=None,#['ra','dec','geocent_time'],                        # set to None if not using, parameters to weight during training
   weighted_pars_factor=1,                                                 # Factor by which to weight parameters if `weighted_pars` is not None.
