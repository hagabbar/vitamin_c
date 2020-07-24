============
Installation
============

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

