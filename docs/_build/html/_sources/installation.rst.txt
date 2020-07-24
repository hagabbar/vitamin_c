============
Installation
============

Clone vitamin repository

.. code-block:: console

   $ git clone https://github.com/hagabbar/vitamin_b

Make a virtual environment

.. code-block:: console

   $ virtualenv -p python3.6 myenv
   $ source myenv/bin/activate
   $ pip install --upgrade pip

cd into your environment and download geos library

.. code-block:: console

   $ cd myenv
   $ git clone https://github.com/matplotlib/basemap.git
   $ cd basemap/geos-3.3.3/

Install geos-3.3.3

.. code-block:: console

   $ mkdir opt
   $ export GEOS_DIR=<full path to opt direcotyr>/opt:$GEOS_DIR
   $ ./configure --prefix=$GEOS_DIR
   $ make; make install

Install basemap, vitamin_b and other required pacakges

.. code-block:: console

   $ cd ../../..
   $ pip install git+https://github.com/matplotlib/basemap.git
   $ pip install .


.. tabs::

   .. tab:: Conda

      .. code-block:: console

          $ conda install -c conda-forge bilby

      Supported python versions: 3.5+.

   .. tab:: Pip

      .. code-block:: console

          $ pip install bilby

      Supported python versions: 3.5+.


This will install all requirements for running :code:`bilby` for general
inference problems, including our default sampler `dynesty
<https://dynesty.readthedocs.io/en/latest/>`_. Other samplers will need to be
installed via pip or the appropriate means.

To install the requirements for running :code:`bilby.gw` for gravitational
wave inference, please additionally run the following commands.

.. tabs::

   .. tab:: Conda

      .. code-block:: console

          $ conda install -c conda-forge gwpy python-lalsimulation

   .. tab:: Pip

      .. code-block:: console

          $ pip install gwpy lalsuite


Install bilby from source
-------------------------

:code:`bilby` is developed and tested with Python 3.6+. In the
following, we assume you have a working python installation, `python pip
<https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing)>`_,
and `git <https://git-scm.com/>`_. See :ref:`installing-python` for our
advise on installing anaconda python.

