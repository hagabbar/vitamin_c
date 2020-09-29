#!/bin/bash

# Shell script to run bilby parameter estimation codes via condor
./gen_benchmark_pe.py -training=${1} -randpars=${2} -samplingfrequency=${3} -samplers=${4} -Ngen=${5} -bounds=${6} -label=${7} -infpars=${8} -fixedvals=${9} -refgeocenttime=${10} -duration=${11} -seed=${12} -dope=${13} -outdir=${14}
