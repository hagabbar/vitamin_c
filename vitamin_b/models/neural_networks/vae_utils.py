import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from lal import GreenwichMeanSiderealTime
import bilby

def convert_ra_to_hour_angle(data, params, rand_pars=False, to_ra=False):
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
    to_ra: bool
        if True, convert from hour angle to RA

    Returns
    -------
    data: array-like
        converted array of source parameter values
    """

    # ignore hour angle conversion if requested by user
    if not params['convert_to_hour_angle']:
        print()
        print('... NOT using hour angle conversion')
        print()
        return data

    print()
    print('... Using hour angle conversion')
    print()
    from astropy.time import Time
    from astropy import coordinates as coord
    from astropy.coordinates import SkyCoord, Angle
    from astropy import units as u
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian
   
    # get ra index
    if rand_pars == True:
        enume_pars = params['rand_pars']
    else:
        enume_pars = params['inf_pars']

    for i,k in enumerate(enume_pars):
        if k == 'ra':
            ra_idx = i 

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('Either time or RA is fixed. Not converting RA to hour angle.')
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
#            if to_ra:
#                conver_result = (t - data[i,ra_idx])
#                data[i,ra_idx] = conver_result / 2*np.pi # why is this factor needed??
#            else:
            conver_result = (t - data[i,ra_idx])
            data[i,ra_idx] = conver_result

#            data[i,ra_idx]=np.mod(GreenwichMeanSiderealTime(params['ref_geocent_time']), 2*np.pi) - data[i,ra_idx]
    return data

def xavier_init(fan_in, fan_out, constant = 1):
    """ xavier weight initialization
    """
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def chris_init(fan_in, fan_out, constant = 1):
    low = -constant # * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant # * np.sqrt(6.0 / (fan_in + fan_out))
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
