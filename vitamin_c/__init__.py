"""
VItamin
=====

VItamin: a user-friendly machine learning posterior generation library.

The aim of VItamin is to provide a user-friendly interface to perform fast parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code is hosted at https://github.com/hagabbar/vitamin_b.
For installation instructions see
"need to add link here".

"""
from __future__ import absolute_import
import sys

from . import gen_benchmark_pe, plotting
from . import params_files
from . import make_params_files

# Check for optional basemap installation
try:
    from mpl_toolkits.basemap import Basemap
    print("module 'basemap' is installed")
except (ModuleNotFoundError, ImportError):
    print("module 'basemap' is not installed")
    print("Skyplotting functionality is automatically disabled.")
else:
    from .skyplotting import plot_sky

from . import run_vitamin

__version__ = "0.2.10"
__author__ = 'Hunter Gabbard'
__credits__ = 'University of Glasgow'

