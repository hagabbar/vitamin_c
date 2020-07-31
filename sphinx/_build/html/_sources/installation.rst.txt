============
Installation
============

This page will tell you how to install VItamin from scratch. There is also
the option to produce sky plots of results, however this requires the 
installation of basemap. VItamin also requires that you use Python3.6 (
more versions to be available in the future). 

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
