================================
Customizing Your Parameters File
================================

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
