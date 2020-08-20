==========================================
Run Your Own Analysis from Start to Finish
==========================================

Start an ipython notebook (or Google Colab)

.. code-block:: console

   $ ipython3

Import vitamin_b and run_vitamin module

.. code-block:: console

   $ import vitamin_b
   $ from vitamin_b import run_vitamin

You will notice that three json files have now appeared in your repository (under dictionary params_files).
These are the default parameter files which vitamin will use to run. Feel 
free to edit these files as you wish. More documentation on customizing your 
run can be found here. (need to provide link to parameters documentation).

We will now generate some training data using the default hyperparameters. 
The code will generate 1000 timeseries by default, but it is recommended to use 
at least 1 million for ideal convergence.

.. code-block:: console

   $ run_vitamin.gen_train()

\
You may use your own customized parameter files by inserting full paths 
to the text files in any of the run_vitamin function like so:

.. note:: Time to run gen_train(): ~(30 mins - 1 hr)

.. code-block:: console

   $ run_vitamin.gen_train(params='params.txt',bounds='bounds.txt',fixed_vals='fixed_vals.txt')

Now generate test timeseries. By default vitamin does not produce Bayesian posteriors for each of these 
timeseries, but you may turn this functionality on by setting `do_pe=True` in the params.json file will. 
Vitamin uses the dynesty sampler by default with 1000 live points 
and a dlogz of 0.1.

.. code-block:: console

   $ run_vitamin.gen_test()

.. note:: Time to run gen_test(): A few hours depending on hardware.

We are ready to start training our network. To do so, simply run the command 
below.

.. code-block:: console

   $ run_vitamin.train()

\   
By default, the code will run for 1,000,000 iterations (saving the model and making plots 
every 50,000 iterations). It is recommended to have the code run for at least a few 100 thousand 
iterations (depending on you batch size).

.. note:: Time to run train(): Will take several hours to run all the way through depending on hardware. Estimate is made using an Nvidia V100 GPU.

To resume training from the last saved checkpoint, simply set the resume_training 
option in train() to True.

.. code-block:: console

   $ run_vitamin.train(resume_training=True)
   
Once your model has been fully trained, we can now produce final results plots. In order 
to get the most use out of thse plots, it is recommended to produce test samples using more 
than one sampler. However, by default vitamin will only use the Dynesty sampler. 

The VItamin test() function will by default make corner plots (without sky map), 
KL divergence plots between posteriors produced by other samplers and VItamin and PP plots.

.. code-block:: console

   $ run_vitamin.test()

.. note:: Time to run test(): ~1s per timeseries to make corner plot results. KL and PP plot will take a few minutes to generate if only using 4 test timeseries and 1 sampler. Time to generate KL and PP plots will increase as more test timeseries and samplers are added.

