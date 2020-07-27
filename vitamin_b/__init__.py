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

from . import models
from . import gen_benchmark_pe, plotting, skyplotting
from . import params_files

from .models import CVAE_model
from .models.neural_networks import batch_manager, vae_utils, VI_decoder_r2, VI_encoder_q, VI_encoder_r1 

from . import run_vitamin

__version__ = "1.0.4"
__author__ = 'Hunter Gabbard'
__credits__ = 'University of Glasgow'

