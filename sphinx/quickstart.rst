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
* Sampling rate is locked at 256 Hz (to be increased in the future).
* Only works on binary black hole signals (more boxes trained on other signals to be released in the future).
  

* Start an ipython notebook (or Google Colab)

.. code-block:: console

   $ ipython3

* import vitamin_b and run_vitamin module

.. code-block:: console

   $ import vitamin_b
   $ from vitamin_b import run_vitamin

.. note:: Need to add a section detailing how to format the test set waveforms

* To produce test samples, simply point vitamin to the directory containing your test waveforms, 
  the pre-trained model (provided here) and the number of samples requested.

.. code-block:: console

   $ samples = run_vitamin.gen_samples(model_loc='model-ex/model.ckpt',test_set='test_set-ex/',
                                       num_samples=10000)


* The function will now return a set of samples (by default it will return 10000 per waveform).

* We can plot these samples in a corner plot by using the following command

.. code-block:: console

   $ import corner
   $ figure = corner.corner(samples)
   $ plt.savefig('./vitamin_example_corner.png')


