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

import sys

from . import models, neural_networks

from .models import CVAE_model
from .neural_networks import batch_manager, vae_utils, VI_decoder_r2, VI_encoder_q, VI_encoder_r1 


__version__ = "1.0.0"
__author__ = 'Hunter Gabbard'
__credits__ = 'University of Glasgow'

