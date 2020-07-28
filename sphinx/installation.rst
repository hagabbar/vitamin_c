============
Installation
============

Clone vitamin repository

.. code-block:: console

   $ git clone https://github.com/hagabbar/vitamin_b
   $ cd vitamin_b

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
   $ pip install -r requirements.txt
   $ pip install vitamin-b


