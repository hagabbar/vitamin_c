============
Installation
============

This page will tell you how to install VItamin both from source and through PyPI. 
If you would like to also produce sky plots, you will need to install from source 
and install the optional basemap/geos packages.

----------------------
Notes on Compatibility
----------------------
VItamin requires Python3.6 (e.g. python3.7 will not work) More versions of python will be compatible in the future.

A GPU is required in order to train a model from scratch, however it is not necessary 
to have a GPU in order to test a pre-trained model (VItamin will default to the CPU 
in this case, unlesss otherwise specified by the user).

-----------
From Source
-----------

Clone vitamin repository

.. code-block:: console

   $ git clone https://github.com/hagabbar/vitamin_b
   $ cd vitamin_b

Make a virtual environment

.. code-block:: console

   $ virtualenv -p python3.6 myenv
   $ source myenv/bin/activate
   $ pip install --upgrade pip

(optional skyplotting install) cd into your environment and download geos library

.. code-block:: console

   $ cd myenv
   $ git clone https://github.com/matplotlib/basemap.git
   $ cd basemap/geos-3.3.3/

(optional skyplotting install) Install geos-3.3.3

.. code-block:: console

   $ mkdir opt
   $ export GEOS_DIR=<full path to opt direcotyr>/opt:$GEOS_DIR
   $ ./configure --prefix=$GEOS_DIR
   $ make; make install
   $ cd ../../..
   $ pip install git+https://github.com/matplotlib/basemap.git

Install vitamin_b and other required pacakges

.. code-block:: console

   $ pip install -r requirements.txt
   $ pip install .

---------
From PyPi
---------

Make a virtual environment

.. code-block:: console

   $ virtualenv -p python3.6 myenv
   $ source myenv/bin/activate
   $ pip install --upgrade pip

Install vitamin_b and other required pacakges

.. code-block:: console

   $ pip install vitamin_b
