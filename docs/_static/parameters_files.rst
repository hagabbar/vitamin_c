=================================
Customizing Your Parameters Files
=================================

VItamin allows the user to tailor their run to a reasonable degree. 
Some work still needs to be done to improve the customizability. 

First, import vitamin to produce the default parameters files

.. code-block:: console

   $ import vitamin

You will now see that there are three json files in your current directory 
(params.json,bounds.json,fixed_vals.json). The params.json file concerns 
the run parameters which make your run unique (batch size, parameters to infer 
etc.). The bounds.json file concerns the allowed numerical bounds for the 
source parameters to be infered. The fixed_vals.json file dictates the 
value at which source parameters are fixed if they are chosen to be 
fixed by the user.

To change the default values for each of these values, use your 
text editor of choice (e.g. vim).

Example json parameters files are provided below..

Bounds json file example:

.. code-block:: json

    {
    "mass_1_min": 35.0,
    "mass_1_max": 80.0,
    "__definition__mass_1": "mass 1 range",
    "mass_2_min": 35.0,
    "mass_2_max": 80.0,
    "__definition__mass_2": "mass 2 range",
    "M_min": 70.0,
    "M_max": 160.0,
    "__definition__M": "total mass range",
    "geocent_time_min": 0.15,
    "geocent_time_max": 0.35,
    "__definition__geocent_time": "time of coalescence range",
    "phase_min": 0.0,
    "phase_max": 6.283185307179586,
    "__definition__phase": "phase range",
    "ra_min": 0.0,
    "ra_max": 6.283185307179586,
    "__definition__ra": "right ascension range",
    "dec_min": -1.5707963267948966,
    "dec_max": 1.5707963267948966,
    "__definition__dec": "declination range",
    "psi_min": 0.0,
    "psi_max": 3.141592653589793,
    "__definition__psi": "psi range",
    "theta_jn_min": 0.0,
    "theta_jn_max": 3.141592653589793,
    "__definition__theta_jn": "inclination angle range",
    "a_1_min": 0.0,
    "a_1_max": 0.8,
    "__definition__a_1": "a1 range",
    "a_2_min": 0.0,
    "a_2_max": 0.8,
    "__definition__a_2": "a2 range",
    "tilt_1_min": 0.0,
    "tilt_1_max": 3.141592653589793,
    "__definition__tilt1": "tilt1 range",
    "tilt_2_min": 0.0,
    "tilt_2_max": 3.141592653589793,
    "__definition__tilt2": "tilt2 range",
    "phi_12_min": 0.0,
    "phi_12_max": 6.283185307179586,
    "__definition__phi12": "phi12 range",
    "phi_jl_min": 0.0,
    "phi_jl_max": 6.283185307179586,
    "__definition_phijl": "phijl range",
    "luminosity_distance_min": 1000.0,
    "luminosity_distance_max": 3000.0,
    "__definition__luminosity_distance": "luminosity distance range"
    }

Fixed values json file:

.. code-block:: json

    {
    "mass_1": 50.0,
    "__definition__mass_1": "m1 fixed value",
    "mass_2": 50.0,
    "__definition__mass_2": "m2 fixed value",
    "mc": null,
    "__definition__mc": "chirp mass fixed value",
    "geocent_time": 0.0,
    "__definition__geocent_time": "time of coalescence fixed value",
    "phase": 0.0,
    "__definition__phase": "phase fixed value",
    "ra": 1.375,
    "__definition__ra": "right ascension fixed value",
    "dec": -1.2108,
    "__definition__dec": "declination fixed value",
    "psi": 0.0,
    "__definition__psi": "psi fixed value",
    "theta_jn": 0.0,
    "__definition__theta_jn": "inclination angle fixed value",
    "luminosity_distance": 2000.0,
    "__definition__luminosity_distance": "luminosity distance fixed value",
    "a_1": 0.0,
    "__definition__a1": "a1 fixed value",
    "a_2": 0.0,
    "__definition__a2": "a2 fixed value",
    "tilt_1": 0.0,
    "__definition__tilt1": "tilt_1 fixed value",
    "tilt_2": 0.0,
    "__definition__tilt_2": "tilt_2 fixed value",
    "phi_12": 0.0,
    "__definition__phi_12": "phi_12 fixed value",
    "phi_jl": 0.0,
    "__definition__phi_jl": "phi_jl fixed value"
    }

General run parameters json file:

.. code-block:: json

    {
    "make_corner_plots": true,
    "__definition__make_corner_plots": "if True, make corner plots",
    "make_kl_plot": true,
    "__definition__make_kl_plot": "If True, go through kl plotting function",
    "make_pp_plot": true,
    "__definition__make_pp_plot": "If True, go through pp plotting function",
    "make_loss_plot": false,
    "__definition__make_loss_plot": "If True, generate loss plot from previous plot data",
    "Make_sky_plot": false,
    "__definition__make_sky_plot": "If True, generate sky plots on corner plots",
    "hyperparam_optim": false,
    "__definition__hyperparam_optim": "optimize hyperparameters for model during training using gaussian process minimization",  
    "resume_training": false,
    "__definition___resume_training": "if True, resume training of a model from saved checkpoint",
    "load_by_chunks": true,
    "__definition__load_by_chunks": "if True, load training samples by a predefined chunk size rather than all at once",
    "ramp": true,
    "__definition__ramp": "# if true, apply linear ramp to KL loss",
    "print_values": true,
    "__definition__print_values": "# optionally print loss values every report interval",
    "by_channel": true,
    "__definition__by_channel": "if True, do convolutions as seperate 1-D channels, if False, stack training samples as 2-D images (n_detectors,(duration*sampling_frequency))",
    "load_plot_data": false,
    "__definition__load_plot_data": "Use plotting data which has already been generated",
    "doPE": false,
    "__definition__doPE": "if True then do bilby PE when generating new testing samples",
    "gpu_num": 0,
    "__definition__gpu_num": "gpu number run is running on",
    "ndata": 256,
    "__definition__ndata": "sampling frequency",
    "run_label": "demo_3det_9par_256Hz",
    "__definition__run_label": "label of run",
    "bilby_results_label": "bilby_results",
    "__definition__bilby_result_label": "label given to bilby results directory",
    "tot_dataset_size": 10000000,
    "__definition__tot_dataset_size": "total number of training samples available to use",
    "tset_split": 1000,
    "__definition__tset_split": "number of training samples in each training data file",
    "plot_dir": "./results/demo_3det_9par_256Hz",
    "__definition__plot_dir": "output directory to save results plots",
    "hyperparam_optim_stop": 1500000,
    "__definition__hyperparam_optim_stop": "stopping iteration of hyperparameter optimizer per call (ideally 1.5 million)",      
    "hyperparam_n_call": 30,
    "__definition__hyperparam_n_call": "number of hyperparameter optimization calls (ideally 30)",
    "load_chunk_size": 1000.0,
    "__definition__load_chunk_size": "Number of training samples to load in at a time.",
    "load_iteration": 390,
    "__definition__load_iteration": "How often to load another chunk of training samples",
    "weight_init": "xavier",
    "__definition__weight_init": "[xavier,VarianceScaling,Orthogonal]. Network model weight initialization (default is xavier)", 
    "n_samples": 10000,
    "__definition__n_samples": "number of posterior samples to save per reconstruction upon inference (default 3000)",
    "num_iterations": 1000001,
    "__definition__num_iterations": "total number of iterations before ending training of model",
    "initial_training_rate": 0.0001,
    "__definition__initial_training_rate": "initial training rate for ADAM optimiser inference model (inverse reconstruction)",  
    "batch_size": 64,
    "__definition__batch_size": " Number training samples shown to neural network per iteration",
    "batch_norm": true,
    "__definition__batch_norm": "if true, do batch normalization in all layers of neural network",
    "report_interval": 500,
    "__definition__report_interval": "interval at which to save objective function values and optionally print info during training",
    "n_modes": 7,
    "__definition__n_modes": "number of modes in Gaussian mixture model (ideal 7, but may go higher/lower)",
    "n_filters_r1": [
        33,
        33,
        33,
        33,
        33
    ],
    "__definition__n_filters_r1": "number of convolutional filters to use in r1 network (must be divisible by 3)",
    "n_filters_r2": [
        33,
        33,
        33
    ],
    "__definition__n_filters_r2": "number of convolutional filters to use in r2 network (must be divisible by 3)",
    "n_filters_q": [
        33,
        33,
        33
    ],
    "__definition__n_filters_q": "number of convolutional filters to use in q network  (must be divisible by 3)",
    "filter_size_r1": [
        5,
        8,
        11,
        10,
        10
    ],
    "__definition__filter_size_r1": "size of convolutional fitlers in r1 network",
    "filter_size_r2": [
        5,
        8,
        11
    ],
    "__definition__filter_size_r2": "size of convolutional filters in r2 network",
    "filter_size_q": [
        5,
        8,
        11
    ],
    "__definition__filter_size_q": "size of convolutional filters in q network",
    "drate": 0.5,
    "__definition__drate": "dropout rate to use in fully-connected layers",
    "maxpool_r1": [
        1,
        2,
        1,
        2,
        1
    ],
    "__definition__maxpool_r1": "size of maxpooling to use in r1 network",
    "conv_strides_r1": [
        1,
        1,
        1,
        1,
        1
    ],
    "__definition__conv_strides_r1": "size of convolutional stride to use in r1 network",
    "pool_strides_r1": [
        1,
        2,
        1,
        2,
        1
    ],
    "__definition_pool_strides_r1": "size of max pool stride to use in r1 network",
    "maxpool_r2": [
        1,
        2,
        1
    ],
    "__definition_maxpool_r2": "size of max pooling to use in r2 network",
    "conv_strides_r2": [
        1,
        1,
        1
    ],
    "__definition__conv_strides_r2": "size of convolutional stride in r2 network",
    "pool_strides_r2": [
        1,
        2,
        1
    ],
    "__definition__pool_strides_r2": "size of max pool stride in r2 network",
    "maxpool_q": [
        1,
        2,
        1
    ],
    "__definition__maxpool_q": "size of max pooling to use in q network",
    "conv_strides_q": [
        1,
        1,
        1
    ],
    "__definition__conv_strides_q": "size of convolutional stride to use in q network",
    "pool_strides_q": [
        1,
        2,
        1
    ],
    "__definition__pool_strides_q": "size of max pool stride to use in q network",
    "ramp_start": 10000.0,
    "__definition__ramp_start": "starting iteration of KL divergence ramp",
    "ramp_end": 100000.0,
    "__definition__ramp_end": "ending iteration of KL divergence ramp",
    "save_interval": 1000,
    "__definition__save_interval": "number of iterations to save model",
    "plot_interval": 1000,
    "__definition__plot_interval": "number of iterations to plot validation results corner plots",
    "z_dimension": 10,
    "__definition__z_dimension": "number of latent space dimensions of model",
    "n_weights_r1": [
        2048,
        2048,
        2048
    ],
    "__definition__n_weights_r1": "number fully-connected neurons in layers of encoders and decoders in the r1 model",
    "n_weights_r2": [
        2048,
        2048,
        2048
    ],
    "__definition__n_weights_r2": "number fully-connected neurons in layers of encoders and decoders in the r2 model",
    "n_weights_q": [
        2048,
        2048,
        2048
    ],
    "__definition__n_weights_q": "number fully-connected neurons in layers of encoders and decoders in the q model",
    "duration": 1.0,
    "__definition__duration": "length of training/validation/test sample time series in seconds",
    "r": 1,
    "__definition__r": "number of GW timeseries to use for testing.",
    "rand_pars": [
        "mass_1",
        "mass_2",
        "luminosity_distance",
        "geocent_time",
        "phase",
        "theta_jn",
        "psi",
        "ra",
        "dec"
    ],
    "__definition__rand_pars": "parameters to randomize (those not listed here are fixed otherwise)",
    "corner_labels": {
        "mass_1": "$m_{1}\\,(\\mathrm{M}_{\\odot})$",
        "mass_2": "$m_{2}\\,(\\mathrm{M}_{\\odot})$",
        "luminosity_distance": "$d_{\\mathrm{L}}\\,(\\mathrm{Mpc})$",
        "geocent_time": "$t_{0}\\,(\\mathrm{seconds})$",
        "phase": "${\\phi}$",
        "theta_jn": "$\\Theta_{jn}\\,(\\mathrm{rad})$",
        "psi": "${\\psi}$",
        "ra": "${\\alpha}\\,(\\mathrm{rad})$",
        "dec": "${\\delta}\\,(\\mathrm{rad})$",
        "a_1": "${a_1}$",
        "a_2": "${a_2}$",
        "tilt_1": "${\\Theta_{1}}$",
        "tilt_2": "${\\Theta_{2}}$",
        "phi_12": "${\\phi_{12}}$",
        "phi_jl": "${\\phi_{jl}}$"
    },
    "__definition__corner_labels": "latex source parameter labels for plotting",
    "ref_geocent_time": 1126259642.5,
    "__definition__ref_geocent_time": "reference gps time (not advised to change this)",
    "training_data_seed": 43,
    "__definition__training_data_seed": "tensorflow training random seed number",
    "testing_data_seed": 44,
    "__definition__testing_data_seed": " tensorflow testing random seed number",
    "inf_pars": [
        "mass_1",
        "mass_2",
        "luminosity_distance",
        "geocent_time",
        "theta_jn",
        "ra",
        "dec"
    ],
    "__definition__inf_pars": "parameters to infer",
    "train_set_dir": "./training_sets_3det_9par_256Hz/tset_tot-10000000_split-1000",
    "__definition__train_set_dir": "location of training set directory",
    "test_set_dir": "./test_sets/bilby_results/test_waveforms",
    "__definition__test_set_dir": "location of test set directory waveforms",
    "pe_dir": "./test_sets/bilby_results/test",
    "__definition__pe_dir": "location of test set directory Bayesian PE samples. Set to None if you do not want to use posterior samples when training.",
    "KL_cycles": 1,
    "__definition__KL_cycles": " number of cycles to repeat for the KL approximation",
    "samplers": [
        "vitamin",
        "dynesty"
    ],
    "__definition__samplers": "Samplers to use when comparing ML results (vitamin is ML approach) dynesty,ptemcee,cpnest,emcee", 
    "figure_sampler_names": [
        "VItamin",
        "Dynesty"
    ],
    "__definition__figure_sampler_names": "matplotlib figure sampler labels (e.g. [ptemcee,CPNest,emcee])",
    "y_normscale": 36.0,
    "__definition__y_normscale": "arbitrary normalization factor on all time series waveforms (helps convergence in training)",  
    "boost_pars": [
        "ra",
        "dec"
    ],
    "__definition__boost_pars": "parameters to boost during training (by default this is off)",
    "gauss_pars": [
        "luminosity_distance",
        "geocent_time",
        "theta_jn",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl"
    ],
    "__definition__gauss_pars": "parameters that require a truncated gaussian distribution",
    "vonmise_pars": [
        "phase",
        "psi"
    ],
    "__definition__vonmises_pars": "parameters that get wrapped on the 1D parameter",
    "sky_pars": [
        "ra",
        "dec"
    ],
    "__definition__sky_pars": "sky parameters",
    "det": [
        "H1",
        "L1",
        "V1"
    ],
    "__definition__det": "LIGO detectors to perform analysis on (default is 3detector H1,L1,V1)",
    "psd_files": [],
    "__definition__psd_files": "User may specficy their own psd files for each detector. Must be bilby compatible .txt files. If specifying your own files, do so for each detector. If list is empty, default Bilby PSD values will be used for each detector. ",    "weighted_pars": null,
    "__definition__weighted_pars": "set to None if not using, parameters to weight during training",
    "weighted_pars_factor": 1,
    "__definition__weighted_pars_factor": "factor by which to weight weighted parameters"
    }
