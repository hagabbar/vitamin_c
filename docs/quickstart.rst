=======
Example
=======

1. Start an ipython notebook (or Google Colab)

.. code-block:: console

   $ ipython3

2. import vitamin_b and run_vitamin module

.. code-block:: console

   $ import vitamin_b
   $ from vitamin_b import run_vitamin

3. You will notice that three text files have now appeared in your repository.
   These are the default parameter files which vitamin will use to run. Feel 
   free to edit these files as you wish. More documentation on customizing your 
   run can be found here. (need to provide link to parameters documentation).

   We will now generate some training data using the default hyperparameters. 
   The code will generate 1000 samples by default, but it is recommended to use 
   at least 1 million for ideal convergence.

.. code-block:: console

   $ run_vitamin.gen_train()

You may use your own customized parameter files by inserting full paths 
to the text files in any of the run_vitamin function like so:

.. code-block:: console

   $ run_vitamin.gen_train('params.txt','bounds.txt','fixed_vals.txt')

4. Now generate test samples with corresponding posteriors. By default vitamin 
   will generate 4 test samples using the dynesty sampler.

.. code-block:: console

   $ run_vitamin.gen_test()

5. We are ready to start training our network. To do so, simply run the command 
   below.

.. code-block:: console

   $ run_vitamin.train()
   
By default, the code will only run for 5000 iterations (saving the model and making plots 
every 1000 iterations). It is recommended to have the code run for at least a few 100 thousand 
iterations (depending on you batch size).

6. To resume training from the last saved checkpoint, simply set the resume_training 
option in train() to True.

.. code-block:: console

   $ run_vitamin.train(resume_training=True)
   
7. Once your model has been fully trained, we can now produce final results plots. In order 
to get the most use out of thse plots, it is recommended to produce test samples using more 
than one sampler. However, by default vitamin will only use the Dynesty sampler. 

The VItamin test() function will by default make corner plots (without sky map), 
KL divergence plots between posteriors produced by other samplers and VItamin, PP plots 
and loss plots.

.. code-block:: console

   $ run_vitamin.test()

