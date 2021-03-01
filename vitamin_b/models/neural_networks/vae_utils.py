import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lal import GreenwichMeanSiderealTime
import bilby
from astropy.time import Time
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :

    with tf.variable_scope(scope) :

        x = tf.nn.relu(x_init)

        if downsample:
            x = tf.add(tf.nn.conv2d(x, get_res_weights(),strides=[2,1],dilations=[1,1],padding='SAME'),get_res_bias())
            x_init = tf.add(tf.nn.conv2d(x_init, tf.Variable(tf.reshape(xavier_init(1, 16*16),[1, 1, 16, 16]), dtype=tf.float32),strides=[2,1],dilations=[1,1],padding='SAME'),get_res_bias())
        else:
            x = tf.add(tf.nn.conv2d(x, get_res_weights(),strides=[1,1],dilations=[1,1],padding='SAME'),get_res_bias())

        x = tf.nn.relu(x)

        x = tf.add(tf.nn.conv2d(x, get_res_weights(),strides=[1,1],dilations=[1,1],padding='SAME'),get_res_bias())
        return x + x_init

def get_res_weights():
    dummy_t = 16
    with tf.variable_scope("weights"):
        res_weights = tf.Variable(tf.reshape(xavier_init(3, dummy_t*16),[3, 1, dummy_t, 16]), dtype=tf.float32)
#    tf.summary.histogram(weight_name+'t', all_weights['VI_decoder_r2'][weight_name+'t'])

        return res_weights

def get_res_bias():
    with tf.variable_scope("biases"):
        res_bias = tf.Variable(tf.zeros([16], dtype=tf.float32))
#    tf.summary.histogram(''+'t', all_weights['VI_decoder_r2'][bias_name+'t'])

        return res_bias

def convert_ra_to_hour_angle(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again

    Parameters
    ----------
    data: array-like
        array containing training/testing data source parameter values
    params: dict
        general parameters of run
    rand_pars: bool
        if True, base ra idx on randomized paramters list
    ra: float
        if not None, convert single ra value to hour angle

    Returns
    -------
    data: array-like
        converted array of source parameter values
    """
    return data

    print('...... Using hour angle conversion')
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian
   
    # compute single instance
    if single:
        return t - data

    # get ra index
    #if rand_pars == True:
    #    enume_pars = params['rand_pars']
    #else:
    #    enume_pars = params['inf_pars']

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i 

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('...... RA is fixed. Not converting RA to hour angle.')
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = t - data[i,ra_idx]

    return data

def convert_hour_angle_to_ra(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again

    Parameters
    ----------
    data: array-like
        array containing training/testing data source parameter values
    params: dict
        general parameters of run
    rand_pars: bool
        if True, base ra idx on randomized paramters list
    ra: float
        if not None, convert single ra value to hour angle

    Returns
    -------
    data: array-like
        converted array of source parameter values
    """
    return data
    print('...... Using hour angle conversion')
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return np.remainder(t - data,2.0*np.pi)

    # get ra index
    #if rand_pars == True:
    #    enume_pars = params['rand_pars']
    #else:
    #    enume_pars = params['inf_pars']

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('...... RA is fixed. Not converting RA to hour angle.')
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = np.remainder(t - data[i,ra_idx],2.0*np.pi)

    return data

def xavier_init(fan_in, fan_out, constant = 1):
    """ xavier weight initialization
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.
    FROM http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363
    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def batch_norm_wrapper(inputs, pop_mean, pop_var, is_training, epsilon, decay = 0.999):

    #scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    #beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    #pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    #pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        pop_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay),validate_shape=True, use_locking=True)
        pop_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay),validate_shape=True, use_locking=True)
        with tf.control_dependencies([pop_mean, pop_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, None, None, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, None, None, epsilon)

