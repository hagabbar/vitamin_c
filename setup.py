#!/usr/bin/env python

from setuptools import setup
import subprocess
import sys
import os

# check that python version is 3.5 or above
python_version = sys.version_info
if python_version < (3, 6):
    sys.exit("Python < 3.6 is not supported, aborting setup")
print("Confirmed Python version {}.{}.{} >= 3.6.0".format(*python_version[:3]))


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

    python_requires='>=3.6', 
    install_requires=['numpy',
                      'universal-divergence',
                      'absl-py',
                      'asn1crypto',
                      'astor',
                      'astroplan',
                      'astropy',
                      'astropy-healpix',
                      'astroquery',
                      'beautifulsoup4',
                      'bilby>=1.0.0',
                      'cachetools',
                      'certifi',
                      'cffi',
                      'chardet',
                      'cloudpickle',
                      'corner',
                      'cpnest',
                      'cryptography',
                      'cycler',
                      'Cython',
                      'decorator',
                      'deepdish',
                      'dill',
                      'dqsegdb2',
                      'dynesty',
                      'emcee',
                      'entrypoints',
                      'future',
                      'gast',
                      'google-auth',
                      'google-auth-oauthlib',
                      'google-pasta',
                      'grpcio',
                      'gwdatafind',
                      'gwosc',
                      'gwpy',
                      'h5py',
                      'healpy',
                      'html5lib',
                      'idna',
                      'jeepney',
                      'joblib',
                      'keyring',
                      'kiwisolver',
                      'lalsuite',
                      'ligo-gracedb',
                      'ligo-segments',
                      'ligo.skymap',
                      'ligotimegps',
                      'lscsoft-glue',
                      'Markdown',
                      'matplotlib',
                      'mock',
                      'networkx',
                      'numexpr',
                      'oauthlib',
                      'opt-einsum',
                      'pandas',
                      'patsy',
                      'Pillow',
                      'protobuf',
                      'ptemcee',
                      'pyaml',
                      'pyasn1',
                      'pyasn1-modules',
                      'pycparser',
                      'pymc3',
                      'pyOpenSSL',
                      'pyparsing',
                      'python-dateutil',
                      'python-ligo-lw',
                      'pytz',
                      'PyYAML',
                      'reproject',
                      'requests',
                      'requests-oauthlib',
                      'rsa',
                      'scikit-learn',
                      'scikit-optimize',
                      'scipy',
                      'seaborn',
                      'SecretStorage',
                      'six',
                      'soupsieve',
                      'tables',
                      'tensorboard>=2.1.1',
                      'tensorflow>=2.1.0',
                      'tensorflow-estimator>=2.1.0',
                      'tensorflow-probability>=0.9.0',
                      'termcolor',
                      'tqdm',
                      'urllib3',
                      'webencodings',
                      'Werkzeug',
                      'wrapt',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: >=3.6',
    ],
)

