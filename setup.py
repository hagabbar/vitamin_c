#!/usr/bin/env python

from setuptools import setup
import subprocess
import sys
import os

# check that python version is 3.5 or above
python_version = sys.version_info
if python_version < (3, 0):
    sys.exit("Python < 3.0 is not supported, aborting setup")
print("Confirmed Python version {}.{}.{} >= 3.0.0".format(*python_version[:3]))


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = 'release'

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('vitamin_b/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md')) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.2.10'
version_file = write_version_file(VERSION)
long_description = get_long_description()


setup(
    name='vitamin_b',
    version='0.2.10',    
    description='A user-friendly machine learning Bayesian inference library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hagabbar/vitamin_b',
    author='Hunter Gabbard, Chris Messenger, Ik Siong Heng, Francesco Tonolini, Roderick Murray-Smith',
    author_email='h.gabbard.1@research.gla.ac.uk',
    license='GNU General Public License v3 (GPLv3)',
    packages=['vitamin_b','vitamin_b.models','vitamin_b.models.neural_networks',
              'vitamin_b.params_files'],
    package_dir={'vitamin_b': 'vitamin_b'},
    package_data={'vitamin_b': ['params_files/*.json'],
                  'vitamin_b': [version_file]},

    python_requires='>=3.5', 
    install_requires=['universal-divergence==0.2.0',
                      'absl-py==0.9.0',
                      'asn1crypto==0.24.0',
                      'astor==0.8.1',
                      'astroplan==0.5',
                      'astropy==3.2.1',
                      'astropy-healpix==0.4',
                      'astroquery==0.3.10',
                      'beautifulsoup4==4.8.0',
                      'bilby==0.5.5',
                      'cachetools==4.0.0',
                      'certifi==2019.9.11',
                      'cffi==1.12.3',
                      'chardet==3.0.4',
                      'cloudpickle==1.2.2',
                      'corner==2.0.1',
                      'cpnest==0.9.9',
                      'cryptography==2.7',
                      'cycler==0.10.0',
                      'Cython==0.29.15',
                      'decorator==4.4.0',
                      'deepdish==0.3.6',
                      'dill==0.3.1.1',
                      'dqsegdb2==1.0.1',
                      'dynesty==0.9.7',
                      'emcee==2.2.1',
                      'entrypoints==0.3',
                      'future==0.18.2',
                      'gast==0.2.2',
                      'google-auth==1.11.3',
                      'google-auth-oauthlib==0.4.1',
                      'google-pasta==0.2.0',
                      'grpcio==1.27.2',
                      'gwdatafind==1.0.4',
                      'gwosc==0.4.3',
                      'gwpy==0.15.0',
                      'h5py==2.9.0',
                      'healpy==1.12.10',
                      'html5lib==1.0.1',
                      'idna==2.8',
                      'jeepney==0.4.1',
                      'joblib==0.14.1',
                      'Keras==2.3.1',
                      'Keras-Applications==1.0.8',
                      'Keras-Preprocessing==1.1.0',
                      'keyring==19.2.0',
                      'kiwisolver==1.1.0',
                      'lalsuite==6.62',
                      'ligo-gracedb==2.4.0',
                      'ligo-segments==1.2.0',
                      'ligo.skymap==0.1.12',
                      'ligotimegps==2.0.1',
                      'lscsoft-glue==2.0.0',
                      'Markdown==3.2.1',
                      'matplotlib==3.1.3',
                      'mock==3.0.5',
                      'networkx==2.3',
                      'numexpr==2.7.0',
                      'numpy==1.18.4',
                      'oauthlib==3.1.0',
                      'opt-einsum==3.2.0',
                      'pandas==1.0.1',
                      'patsy==0.5.1',
                      'Pillow==6.1.0',
                      'protobuf==3.11.3',
                      'ptemcee==1.0.0',
                      'pyaml==20.4.0',
                      'pyasn1==0.4.8',
                      'pyasn1-modules==0.2.8',
                      'pycparser==2.19',
                      'pymc3==3.7',
                      'pyOpenSSL==19.0.0',
                      'pyparsing==2.4.6',
                      'python-dateutil==2.8.1',
                      'python-ligo-lw==1.5.3',
                      'pytz==2019.3',
                      'PyYAML==5.1.2',
                      'reproject==0.5.1',
                      'requests==2.22.0',
                      'requests-oauthlib==1.3.0',
                      'rsa==4.0',
                      'scikit-learn==0.22.2.post1',
                      'scikit-optimize==0.7.4',
                      'scipy==1.4.1',
                      'seaborn==0.9.0',
                      'SecretStorage==3.1.1',
                      'six==1.14.0',
                      'soupsieve==1.9.3',
                      'tables==3.5.2',
                      'tensorboard==2.1.1',
                      'tensorflow==2.1.0',
                      'tensorflow-estimator==2.1.0',
                      'tensorflow-probability==0.9.0',
                      'termcolor==1.1.0',
                      'Theano==1.0.4',
                      'tqdm==4.42.1',
                      'urllib3==1.25.6',
                      'webencodings==0.5.1',
                      'Werkzeug==1.0.0',
                      'wrapt==1.12.1',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)

