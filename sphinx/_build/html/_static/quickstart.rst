==========
Quickstart
==========

.. note:: This page is currently being worked on and is NOT an operational part of the 
   code just yet.

This section will detail how you can produce samples from a gravitational wave (GW) postrior within 
minutes of downloading VItamin given your own GW time series binary black hole data. You may also 
optionally use the pregenerated sets of waveforms provided here

Some things to note about the provided black box:
* We currently set all priors to be uniformly distributed and produce posteriors 
on 7 parameters (m1,m2,luminosity distance,time of coalescence,inclination angle,
right ascension and declination). Both phase and psi are internally marginalized out.

* Sampling rate is locked at 256 Hz and duration is locked at 1s (more boxes with other values will become available in future).

* Only works on binary black hole signals (more boxes trained on other signals to be released in the future).

* Does not require a GPU to generate samples, but will be ~1s slower to generate all samples per time series.  

* Start an ipython notebook (or Google Colab)

.. code-block:: console

   $ ipython3

* import vitamin_b and run_vitamin module

.. code-block:: console

   $ import vitamin_b
   $ from vitamin_b import run_vitamin

.. note:: Test samples should be of the format 'data_<test sample number>.h5py'. Where the h5py file 
   should have a directory containing the source parameter scalar values to be infered of the timeseries labeled 'x_data', 
   and a directory containing the noisy time series labeled 'y_data_noisy'. 'x_data' should be a numpy array of shape (1,<
   number of source parameters to be infered>). 'y_data' should be a numpy array of shape (<number of detectors>,
   <sample rate X duration>) 

* To produce test samples, simply point vitamin to the directory containing your test waveforms, the pre-trained model (provided here) and specify the number of samples per posterior requested.

.. code-block:: console

   $ samples,_,x_data = run_vitamin.gen_samples(model_loc='inverse_model_dir_demo_3det_9par_256Hz_run1/inverse_model.ckpt',
                                                test_set='test_sets/all_4_samplers/test_waveforms/',num_samples=10000,
                                                plot_corner=True)


* The function will now return a set of samples (by default it will return 10000 per timeseries) and the true source parameter values for each timeseries. 

* Since we set the option plot_corner=True, you will also find a corner plot in the same directory as we ran the code under the title 'vitamin_example_corner.png'.

