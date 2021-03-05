import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# enable eager execution
tf.compat.v1.enable_eager_execution()
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import corner
import os
import h5py
import json
from universal_divergence import estimate

def get_param_index(all_pars,pars,sky_extra=None):
    """ 
    Get the list index of requested source parameter types
    """
    # identify the indices of wrapped and non-wrapped parameters - clunky code
    mask = []
    idx = []

    # loop over inference params
    for i,p in enumerate(all_pars):

        # loop over wrapped params
        flag = False
        for q in pars:
            if p==q:
                flag = True    # if inf params is a wrapped param set flag

        # record the true/false value for this inference param
        if flag==True:
            mask.append(True)
            idx.append(i)
        elif flag==False:
            mask.append(False)

    if sky_extra is not None:
        if sky_extra:
            mask.append(True)
            idx.append(len(all_pars))
        else:
            mask.append(False)

    return mask, idx, np.sum(mask)

def convert_ra_to_hour_angle(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """

    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return t - data

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        #print('...... RA is fixed. Not converting RA to hour angle.')
        pass
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = t - data[i,ra_idx]

    return data

def convert_hour_angle_to_ra(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return np.remainder(t - data,2.0*np.pi)

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        #print('...... RA is fixed. Not converting RA to hour angle.')
        pass
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = np.remainder(t - data[i,ra_idx],2.0*np.pi)

    return data

def load_data(params,bounds,fixed_vals,input_dir,inf_pars,test_data=False,silent=False):
    """ Function to load either training or testing data.
    """

    # Get list of all training/testing files and define dictionary to store values in files
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
    else:
        print('ERROR: input directory not a string')
        exit(0)

    # Sort files from first generated to last generated
    filenames = sorted(os.listdir(dataLocations[0]))

    # If loading by chunks, randomly shuffle list of training/testing filenames
    if params['load_by_chunks'] == True and not test_data:
        nfiles = np.min([int(params['load_chunk_size']/float(params['tset_split'])),len(filenames)])
        files_idx = np.random.randint(0,len(filenames),nfiles) 
        filenames= np.array(filenames)[files_idx]
        if not silent:
            print('...... shuffled filenames since we are loading in by chunks')

    # Iterate over all training/testing files and store source parameters, time series and SNR info in dictionary
    data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': [], 'snrs': []}
    for filename in filenames:
        try:
            data['x_data'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:])
            data['y_data_noisefree'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:])
            if test_data:
                data['y_data_noisy'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:])
            data['rand_pars'] = h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]
            data['snrs'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['snrs'][:])
            if not silent:
                print('...... Loaded file ' + dataLocations[0] + '/' + filename)
        except OSError:
            print('Could not load requested file')
            continue
    if np.array(data['y_data_noisefree']).ndim == 3:
        data['y_data_noisefree'] = np.expand_dims(np.array(data['y_data_noisefree']),axis=0)
        data['y_data_noisy'] = np.expand_dims(np.array(data['y_data_noisy']),axis=0)
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.transpose(np.concatenate(np.array(data['y_data_noisefree']), axis=0),[0,2,1])
    if test_data:
        data['y_data_noisy'] = np.transpose(np.concatenate(np.array(data['y_data_noisy']), axis=0),[0,2,1])
    data['snrs'] = np.concatenate(np.array(data['snrs']), axis=0)

    # convert ra to hour angle
    data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, params['rand_pars'])

    # Normalise the source parameters
    for i,k in enumerate(data['rand_pars']):
        par_min = k.decode('utf-8') + '_min'
        par_max = k.decode('utf-8') + '_max'
        # normalize by bounds
        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])

    # extract inference parameters from all source parameters loaded earlier
    idx = []
    infparlist = ''
    for k in inf_pars:
        infparlist = infparlist + k + ', '
        for i,q in enumerate(data['rand_pars']):
            m = q.decode('utf-8')
            if k==m:
                idx.append(i)
    data['x_data'] = tf.cast(data['x_data'][:,idx],dtype=tf.float32)
    if not silent:
        print('...... {} will be inferred'.format(infparlist))

    return data['x_data'], tf.cast(data['y_data_noisefree'],dtype=tf.float32), tf.cast(data['y_data_noisy'],dtype=tf.float32), data['snrs']

def load_samples(params):
    """
    read in pre-computed posterior samples
    """
    if type("%s" % params['pe_dir']) is str:
        # load generated samples back in
        dataLocations = '%s_%s' % (params['pe_dir'],params['samplers'][1])
        print('... looking in {} for posterior samples'.format(dataLocations))
    else:
        print('ERROR: input samples directory not a string')
        exit(0)

    # Iterate over requested number of testing samples to use
    for i in range(params['r']):

        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)
        if not os.path.isfile(filename):
            print('... unable to find file {}. Exiting.'.format(filename))
            exit(0)

        print('... Loading test sample -> ' + filename)
        data_temp = {}
        n = 0

        # Retrieve all source parameters to do inference on
        for q in params['bilby_pars']:
            p = q + '_post'
            par_min = q + '_min'
            par_max = q + '_max'
            data_temp[p] = h5py.File(filename, 'r')[p][:]
            if p == 'geocent_time_post':
                data_temp[p] = data_temp[p] - params['ref_geocent_time']
            data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
            Nsamp = data_temp[p].shape[0]
            n = n + 1
        print('... read in {} samples from {}'.format(Nsamp,filename))

        # place retrieved source parameters in numpy array rather than dictionary
        j = 0
        XS = np.zeros((Nsamp,n))
        for p,d in data_temp.items():
            XS[:,j] = d
            j += 1
        print('... put the samples in an array')

        # Append test sample posteriors to existing array of other test sample posteriors
        rand_idx_posterior = np.random.randint(0,Nsamp,params['n_samples'])
        if i == 0:
            XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
        else:
            XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))
        print('... appended {} samples to the total'.format(params['n_samples']))
    return XS_all

def plot_losses(train_loss, val_loss, epoch, run='testing'):
    """
    plots the losses
    """
    plt.figure()
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,0],'b',label='RECON')       
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,1],'r',label='KL')
    plt.semilogx(np.arange(1,epoch+1),train_loss[:epoch,2],'g',label='TOTAL')       
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,0],'--b',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,1],'--r',alpha=0.5)
    plt.semilogx(np.arange(1,epoch+1),val_loss[:epoch,2],'--g',alpha=0.5)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.ylim([np.min(1.1*train_loss[int(0.1*epoch):epoch,:]),np.max(1.1*train_loss[int(0.1*epoch):epoch,:])])
    plt.savefig('%s/loss.png' % (run))
    plt.close()
 
def plot_KL(KL_samples, step, run='testing'):
    """
    plots the KL evolution
    """
    N = KL_samples.shape[0]
    plt.figure()
    for kl in np.transpose(KL_samples):
        plt.semilogx(np.arange(1,N+1)*step,kl)
    plt.xlabel('epoch')
    plt.ylabel('KL')
    plt.grid()
    plt.ylim([-0.2,1.0])
    plt.savefig('%s/kl.png' % (run))
    plt.close()


def plot_posterior(samples,x_truth,epoch,idx,run='testing',other_samples=None):
    """
    plots the posteriors
    """

    if other_samples is not None:
        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])
        true_x = np.zeros(inf_ol_len)
        true_XS = np.zeros([samples.shape[0],inf_ol_len])
        ol_pars = []
        cnt = 0
        for inf_idx,bilby_idx in zip(inf_ol_idx,bilby_ol_idx):
            inf_par = params['inf_pars'][inf_idx]
            bilby_par = params['bilby_pars'][bilby_idx]
            true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
            true_post[:,cnt] = (other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
            true_x[cnt] = (x_truth[inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
            ol_pars.append(inf_par)
            cnt += 1
        parnames = []
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, ol_pars):
                parnames.append(params['corner_labels'][k])

        # convert to RA
        true_XS = convert_hour_angle_to_ra(true_XS,params,ol_pars)
        true_x = convert_hour_angle_to_ra(np.reshape(true_x,[1,true_XS.shape[1]]),params,ol_pars).flatten()

        # compute KL estimate
        idx1 = np.random.randint(0,true_XS.shape[0],1000)
        idx2 = np.random.randint(0,true_post.shape[0],1000)
        try:
            KL_est = estimate(true_XS[idx1,:],true_post[idx2,:])
        except:
            KL_est = -1.0
            pass

    else:
        # Get corner parnames to use in plotting labels
        parnames = []
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, params['inf_pars']):
                parnames.append(params['corner_labels'][k])
        # un-normalise full inference parameters
        full_true_x = np.zeros(len(params['inf_pars']))
        new_samples = np.zeros([samples.shape[0],len(params['inf_pars'])])
        for inf_par_idx,inf_par in enumerate(params['inf_pars']):
            new_samples[:,inf_par_idx] = (samples[:,inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
            full_true_x[inf_par_idx] = (x_truth[inf_par_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
        new_samples = convert_hour_angle_to_ra(new_samples,params,params['inf_pars'])
        full_true_x = convert_hour_angle_to_ra(np.reshape(full_true_x,[1,samples.shape[1]]),params,params['inf_pars']).flatten()       
        KL_est = -1.0

    # define general plotting arguments
    defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

    # 1-d hist kwargs for normalisation
    hist_kwargs = dict(density=True,color='tab:red')
    hist_kwargs_other = dict(density=True,color='tab:blue')

    if other_samples is None:
        figure = corner.corner(new_samples,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=True, truths=x_truth,
                           show_titles=True, hist_kwargs=hist_kwargs)
        plt.savefig('%s/full_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
    else:
        figure = corner.corner(true_post, **defaults_kwargs,labels=parnames,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
        corner.corner(true_XS,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True, truths=true_x,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
        plt.annotate('KL = {:.3f}'.format(KL_est),(0.2,0.95),xycoords='figure fraction',fontsize=18)
        plt.savefig('%s/comp_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
    return KL_est

def plot_latent(mu_r1, z_r1, mu_q, z_q, epoch, idx, run='testing'):

    # define general plotting arguments
    defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

    # 1-d hist kwargs for normalisation
    hist_kwargs = dict(density=True,color='tab:red')
    hist_kwargs_other = dict(density=True,color='tab:blue')

    figure = corner.corner(z_q, **defaults_kwargs,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
    corner.corner(z_r1,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
    # Extract the axes
    z_dim = z_r1.shape[1]
    axes = np.array(figure.axes).reshape((z_dim, z_dim))

    # Loop over the histograms
    for yi in range(z_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(mu_r1[0,:,xi], mu_r1[0,:,yi], "sr")
            ax.plot(mu_q[0,xi], mu_q[0,yi], "sb")
    plt.savefig('%s/latent_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()

params = './params_files/params.json'
bounds = './params_files/bounds.json'
fixed_vals = './params_files/fixed_vals.json'
run = time.ctime().replace(' ', '-')
EPS = 1e-6

# Load parameters files
with open(params, 'r') as fp:
    params = json.load(fp)
with open(bounds, 'r') as fp:
    bounds = json.load(fp)
with open(fixed_vals, 'r') as fp:
    fixed_vals = json.load(fp)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_visible_devices(gpus[int(params['gpu_num'])], 'GPU')
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# if doing hour angle, use hour angle bounds on RA
bounds['ra_min'] = convert_ra_to_hour_angle(bounds['ra_min'],params,None,single=True)
bounds['ra_max'] = convert_ra_to_hour_angle(bounds['ra_max'],params,None,single=True)
print('... converted RA bounds to hour angle')
inf_ol_mask, inf_ol_idx, inf_ol_len = get_param_index(params['inf_pars'],params['bilby_pars'])
bilby_ol_mask, bilby_ol_idx, bilby_ol_len = get_param_index(params['bilby_pars'],params['inf_pars'])

# identify the indices of different sets of physical parameters
vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])       
m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])       
m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])       
#idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)
idx_mask = np.argsort(m1_idx_mask + m2_idx_mask + gauss_idx_mask + vonmise_idx_mask) # + sky_idx_mask)
dist_mask, dist_idx_mask, dist_len = get_param_index(params['inf_pars'],['luminosity_distance'])
xyz_mask, xyz_idx_mask, xyz_len = get_param_index(params['inf_pars'],['luminosity_distance','ra','dec'])
not_xyz_mask, not_xyz_idx_mask, not_xyz_len = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])
idx_xyz_mask = np.argsort(xyz_idx_mask + not_xyz_idx_mask)
print(xyz_mask)
print(not_xyz_mask)
print(idx_xyz_mask)
#masses_len = m1_len + m2_len
print(params['inf_pars'])
print(vonmise_mask,vonmise_idx_mask)
print(gauss_mask,gauss_idx_mask)
print(m1_mask,m1_idx_mask)
print(m2_mask,m2_idx_mask)
print(sky_mask,sky_idx_mask)
print(idx_mask)

train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_log_dir = params['plot_dir'] + '/logs'

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, x_dim, y_dim, n_channels, z_dim, n_modes, params):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.params = params

#        n_filters_r1 = self.params['n_filters_r1']
#        filter_size_r1 = self.params['filter_size_r1']
#        conv_strides_r1 = self.params['conv_strides_r1']
#        pool_strides_r1 = self.params['pool_strides_r1']
        n_weights_r1 = self.params['n_weights_r1']

        # the r1 encoder network
        r1_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))

        ### 1st layer
        layer_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=1, padding='same', activation='relu')(r1_input_y)
        layer_1 = tf.keras.layers.Conv1D(filters=10, kernel_size=3, padding='same', activation='relu')(layer_1)

        layer_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=1, padding='same', activation='relu')(r1_input_y)
        layer_2 = tf.keras.layers.Conv1D(filters=10, kernel_size=5, padding='same', activation='relu')(layer_2)

        layer_3 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(r1_input_y)
        layer_3 = tf.keras.layers.Conv1D(filters=10, kernel_size=1, padding='same', activation='relu')(layer_3)

        mid_1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis=2)    
        a = tf.keras.layers.Flatten()(mid_1)

        """
        for idx,i in enumerate(n_filters_r1):
            if idx == 0:
                a = tf.keras.layers.Conv1D(filters=i, kernel_size=filter_size_r1[idx]
                                           , strides=conv_strides_r1[idx], activation='relu')(r1_input_y)
            else:
                a = tf.keras.layers.Conv1D(filters=i, kernel_size=filter_size_r1[idx]
                                           , strides=conv_strides_r1[idx], activation='relu')(a)
            a = tf.keras.layers.MaxPooling1D(pool_size=pool_strides_r1[idx],
                                             strides=pool_strides_r1[idx])(a)
        a = tf.keras.layers.Flatten()(a)

        for idx,i in enumerate(n_weights_r1):
            if idx==0:
                a2 = tf.keras.layers.Dense(n_weights_r1[idx],activation='relu')(a)
            else:
                a2 = tf.keras.layers.Dense(n_weights_r1[idx],activation='relu')(a2)
        a2 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(a2)
        self.encoder_r1 = tf.keras.Model(inputs=r1_input_y, outputs=a2)
        print(self.encoder_r1.summary())
        """
        a2 = tf.keras.layers.Dense(64,activation='relu')(a)
        a2 = tf.keras.layers.Dense(64,activation='relu')(a2)
        a2 = tf.keras.layers.Dense(64,activation='relu')(a2)
        a2 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(a2)       
        self.encoder_r1 = tf.keras.Model(inputs=r1_input_y, outputs=a2)
        print(self.encoder_r1.summary())

        """
        # the q encoder network
#        q_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        for idx,i in enumerate(self.params['n_filters_q']):
            if idx == 0:
                b = tf.keras.layers.Conv1D(filters=i, kernel_size=self.params['filter_size_q'][idx], 
                                           strides=self.params['conv_strides_q'][idx], activation='relu')(q_input_y)
            else:
                b = tf.keras.layers.Conv1D(filters=i, kernel_size=self.params['filter_size_q'][idx], 
                                           strides=self.params['conv_strides_q'][idx], activation='relu')(b)
            b = tf.keras.layers.MaxPooling1D(pool_size=self.params['pool_strides_q'][idx],
                                             strides=self.params['pool_strides_q'][idx])(b)  
        b = tf.keras.layers.Flatten()(b)              

        c = tf.keras.layers.Flatten()(q_input_x)
#        d = tf.keras.layers.concatenate([b,c])
        d = tf.keras.layers.concatenate([a,c])
        for idx,i in enumerate(self.params['n_weights_q']):
            if idx == 0:
                e = tf.keras.layers.Dense(self.params['n_weights_q'][idx],activation='relu')(d)
            else:
                e = tf.keras.layers.Dense(self.params['n_weights_q'][idx],activation='relu')(e)
        e = tf.keras.layers.Dense(2*self.z_dim)(e)
#        self.encoder_q = tf.keras.Model(inputs=[q_input_y, q_input_x], outputs=e)
        self.encoder_q = tf.keras.Model(inputs=[r1_input_y, q_input_x], outputs=e)
        print(self.encoder_q.summary())
        """
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        c = tf.keras.layers.Flatten()(q_input_x)
        d = tf.keras.layers.concatenate([a,c])
        e = tf.keras.layers.Dense(64,activation='relu')(d)
        e = tf.keras.layers.Dense(64,activation='relu')(e)
        e = tf.keras.layers.Dense(64,activation='relu')(e)
        e = tf.keras.layers.Dense(2*self.z_dim)(e)
        self.encoder_q = tf.keras.Model(inputs=[r1_input_y, q_input_x], outputs=e)     
        print(self.encoder_q.summary())

        """
        # the r2 decoder network
#        r2_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        for idx,i in enumerate(self.params['n_filters_r2']):
            if idx==0:
                f = tf.keras.layers.Conv1D(filters=i, kernel_size=self.params['filter_size_r2'][idx], 
                                           strides=self.params['conv_strides_r2'][idx], activation='relu')(r2_input_y)
            else:
                f = tf.keras.layers.Conv1D(filters=i, kernel_size=self.params['filter_size_r2'][idx], 
                                           strides=self.params['conv_strides_r2'][idx], activation='relu')(f)
            f = tf.keras.layers.MaxPooling1D(pool_size=self.params['pool_strides_r2'][idx],
                                             strides=self.params['pool_strides_r2'][idx])(f)
        f = tf.keras.layers.Flatten()(f)
        g = tf.keras.layers.Flatten()(r2_input_z)
#        h = tf.keras.layers.concatenate([f,g])
        h = tf.keras.layers.concatenate([a,g])
        for idx,_ in enumerate(self.params['n_weights_r2']):
            if idx == 0:
                i = tf.keras.layers.Dense(self.params['n_weights_r2'][idx],activation='relu')(h)
            else:
                i = tf.keras.layers.Dense(self.params['n_weights_r2'][idx],activation='relu')(i)
        j = tf.keras.layers.Dense(2*self.x_dim)(i)
#        k = tf.keras.layers.Dense(self.x_dim,activation='relu')(i) 
#        l = tf.keras.layers.concatenate([j,k])
#        self.decoder_r2 = tf.keras.Model(inputs=[r2_input_y, r2_input_z], outputs=l)
        self.decoder_r2 = tf.keras.Model(inputs=[r1_input_y, r2_input_z], outputs=j)
        print(self.decoder_r2.summary())
        """
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        g = tf.keras.layers.Flatten()(r2_input_z)
        h = tf.keras.layers.concatenate([a,g])
        i = tf.keras.layers.Dense(64,activation='relu')(h)
        i = tf.keras.layers.Dense(64,activation='relu')(i)
        i = tf.keras.layers.Dense(64,activation='relu')(i)
        j = tf.keras.layers.Dense(2*self.x_dim)(i)
        self.decoder_r2 = tf.keras.Model(inputs=[r1_input_y, r2_input_z], outputs=j)   
        print(self.decoder_r2.summary()) 

    def encode_r1(self, y=None):
        mean, logvar, weight = tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        return tf.reshape(mean,[-1,self.n_modes,self.z_dim]), tf.reshape(logvar,[-1,self.n_modes,self.z_dim]), tf.reshape(weight,[-1,self.n_modes])
        #return tf.split(self.encoder_r1(y), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)

    def encode_q(self, x=None, y=None):
        #mean, logvar = tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)
        #return tf.reshape(mean,[-1,self.z_dim]), tf.reshape(logvar,[-1,self.z_dim])
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        return tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim,self.x_dim], axis=1)
        #return mean, logvar

optimizer = tf.keras.optimizers.Adam(1e-4)

def compute_loss(model, x, y, y_normscale=1.0, ramp=1.0, boundary=1.0, noiseamp=1.0):
    noiseamp = tf.cast(noiseamp, dtype=tf.float32)
    y_normscale = tf.cast(y_normscale, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)

    y = (y + noiseamp*tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=1.0, dtype=tf.float32))/y_normscale
    mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y)
    scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
    gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=mean_r1,
            scale_diag=scale_r1))
    mean_q, logvar_q = model.encode_q(x=x,y=y)
    scale_q = EPS + tf.sqrt(tf.exp(logvar_q))
    mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=mean_q,
                          scale_diag=scale_q)
    #mvn_q = tfd.Normal(loc=mean_q,scale=scale_q)
    z_samp = mvn_q.sample()
    mean_r2, logvar_r2 = model.decode_r2(z=z_samp,y=y)
    scale_r2 = EPS + tf.sqrt(tf.exp(logvar_r2))

    # SIMPLE RECON LOSS
#    mvn_r2 = tfp.distributions.MultivariateNormalDiag(
#                          loc=tf.slice(mean_r2,[0,0],[-1,model.x_dim]),
#                          scale_diag=scale_r2)

    mvn_r2 = tfp.distributions.MultivariateNormalDiag(
                          loc=mean_r2,
                          scale_diag=scale_r2)
    x = tf.cast(x, dtype=tf.float32)
    simple_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(mvn_r2.log_prob(x)))

    # LOCALISATION
#    ra = 2*np.pi*tf.reshape(tf.boolean_mask(x,ra_mask,axis=1),[-1,1])       # convert the scaled 0->1 true RA value back to radians
#    dec = np.pi*(tf.reshape(tf.boolean_mask(x,dec_mask,axis=1),[-1,1]) - 0.5) # convert the scaled 0>1 true dec value back to radians
#    dist = tf.boolean_mask(x,dist_mask,axis=1)
#    xyz_actual = tf.reshape(tf.concat([dist*tf.cos(ra)*tf.cos(dec),dist*tf.sin(ra)*tf.cos(dec),dist*tf.sin(dec)],axis=1),[-1,3])   # construct the true parameter unit vector
#    new_x = tf.gather(tf.concat([xyz_actual, tf.boolean_mask(x,not_xyz_mask,axis=1)],axis=1),tf.constant(idx_xyz_mask),axis=1)
#    new_x = tf.cast(new_x, dtype=tf.float32)
#    simple_cost_recon = -1.0*tf.reduce_mean(mvn_r2.log_prob(new_x))

    selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
    log_r1_q = gm_r1.log_prob(z_samp)   # evaluate the log prob of r1 at the q samples
    cost_KL = selfent_q - tf.reduce_mean(log_r1_q)
    return simple_cost_recon, cost_KL

def gen_samples(model, y, ramp=1.0, nsamples=1000):
    y = y/params['y_normscale']
    y = tf.tile(y,(nsamples,1,1))
    mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y)
    scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
    gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=mean_r1,
            scale_diag=scale_r1))
    z_samp = gm_r1.sample()
    mean_r2, logvar_r2 = model.decode_r2(z=z_samp,y=y)
    scale_r2 = EPS + tf.sqrt(tf.exp(logvar_r2))

    mvn_r2 = tfp.distributions.MultivariateNormalDiag(
                          loc=tf.slice(mean_r2,[0,0],[-1,model.x_dim]),
                          scale_diag=scale_r2)
    x_sample = mvn_r2.sample()

    # LOCALISATION
    xyz = tf.reshape(tf.boolean_mask(x_sample,xyz_mask,axis=1),[-1,3])          # sample the distribution
    samp_dist = tf.reshape(tf.math.reduce_euclidean_norm(xyz,axis=1),[-1,1])        
    normed_xyz = tf.math.l2_normalize(xyz,axis=1)
    samp_ra = tf.math.floormod(tf.atan2(tf.slice(normed_xyz,[0,1],[-1,1]),tf.slice(normed_xyz,[0,0],[-1,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
    samp_dec = (tf.asin(tf.slice(normed_xyz,[0,2],[-1,1])) + 0.5*np.pi)/np.pi       
                # convert to the rescaled 0->1 dec from the unit vector
    x_samp_sky = tf.reshape(tf.concat([samp_dist,samp_ra,samp_dec],axis=1),[-1,3])  
           # group the sky samples
    new_x = tf.gather(tf.concat([x_samp_sky, tf.boolean_mask(x_sample,not_xyz_mask,axis=1)],axis=1),tf.constant(idx_xyz_mask),axis=1)
    return new_x

def gen_z_samples(model, x, y, nsamples=1000):
    y = y/params['y_normscale']
    y = tf.tile(y,(nsamples,1,1))
    x = tf.tile(x,(nsamples,1))
    mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y)
    scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
    gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=mean_r1,
            scale_diag=scale_r1))
    z_samp_r1 = gm_r1.sample()
    mean_q, logvar_q = model.encode_q(x=x,y=y)
    scale_q = EPS + tf.sqrt(tf.exp(logvar_q))
    mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=mean_q,
                          scale_diag=scale_q)
    z_samp_q = mvn_q.sample()
    return mean_r1, z_samp_r1, mean_q, z_samp_q

@tf.function
def train_step(model, x, y, optimizer, y_normscale, ramp=1.0, boundary=1.0):
    """Executes one training step and returns the loss.
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    with tf.GradientTape() as tape:
        r_loss, kl_loss = compute_loss(model, x, y, ramp=ramp)
        loss = r_loss + ramp*kl_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss_metric(loss)
    return r_loss, kl_loss

def run_vitc(params, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, y_data_test_noisefree, save_dir, truth_test, bounds, fixed_vals, bilby_samples, snrs_test=None):

    y_data_test = y_data_test[:params['r']]
    x_data_test = x_data_test[:params['r']]
    epochs = params['num_iterations']
    train_size = params['load_chunk_size']
    batch_size = params['batch_size']
    val_size = params['val_dataset_size']
    test_size = params['r']
    plot_dir = params['plot_dir']
    plot_cadence = params['plot_interval']
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "inverse_model_%s/model.ckpt" % params['run_label']
    checkpoint_dir = os.path.dirname(checkpoint_path)

    train_dataset = (tf.data.Dataset.from_tensor_slices((x_data_train,y_data_train))
                     .shuffle(train_size).batch(batch_size))
    val_dataset = (tf.data.Dataset.from_tensor_slices((x_data_val,y_data_val))
                    .shuffle(val_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices((x_data_test,y_data_test))
                    .batch(1))

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if params['resume_training']:
        model = CVAE(x_data_train.shape[1], params['ndata'],
                     y_data_train.shape[2], params['z_dimension'], params['n_modes'], params) 
        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
    else:
        model = CVAE(x_data_train.shape[1], params['ndata'],
                     y_data_train.shape[2], params['z_dimension'], params['n_modes'], params)

    # start the training loop
    train_loss = np.zeros((epochs,3))
    val_loss = np.zeros((epochs,3))
    ramp_start = params['ramp_start']
    ramp_stop = params['ramp_end']
    ramp_grad = 1.0/(ramp_stop - ramp_start)
    KL_samples = []
    for epoch in range(1, epochs + 1):

        train_loss_kl_q = 0.0
        train_loss_kl_r1 = 0.0
        start_time_train = time.time()
        ramp = tf.convert_to_tensor(np.min([(epoch-ramp_start)*ramp_grad,1.0]).astype(np.single) if epoch>ramp_start else 0.0, dtype=tf.float32)
#        boundary = tf.convert_to_tensor(np.min([(epoch-boundary_start)*boundary_grad,1.0]).astype(np.single) if epoch>boundary_start else 0.0, dtype=tf.float32)
        for step, (x_batch_train, y_batch_train) in train_dataset.enumerate():
            temp_train_r_loss, temp_train_kl_loss = train_step(model, x_batch_train, y_batch_train, optimizer, ramp=ramp)
            train_loss[epoch-1,0] += temp_train_r_loss
            train_loss[epoch-1,1] += temp_train_kl_loss
        train_loss[epoch-1,2] = train_loss[epoch-1,0] + ramp*train_loss[epoch-1,1]
        train_loss[epoch-1,:] /= float(step-1)
        end_time_train = time.time()
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss_metric.result(), step=epoch)
        train_loss_metric.reset_states()

        start_time_val = time.time()
        for step, (x_batch_val, y_batch_val) in val_dataset.enumerate():
            temp_val_r_loss, temp_val_kl_loss = compute_loss(model, x_batch_val, y_batch_val, ramp=ramp)
            val_loss[epoch-1,0] += temp_val_r_loss
            val_loss[epoch-1,1] += temp_val_kl_loss
        val_loss[epoch-1,2] = val_loss[epoch-1,0] + ramp*val_loss[epoch-1,1]
        val_loss[epoch-1,:] /= float(step-1)
        end_time_val = time.time()

        print('Epoch: {}, Run {}, Training RECON: {}, KL: {}, TOTAL: {}, time elapsed: {}'
            .format(epoch, run, train_loss[epoch-1,0], train_loss[epoch-1,1], train_loss[epoch-1,2], end_time_train - start_time_train))
        print('Epoch: {}, Run {}, Validation RECON: {}, KL: {}, TOTAL: {}, time elapsed {}'
            .format(epoch, run, val_loss[epoch-1,0], val_loss[epoch-1,1], val_loss[epoch-1,2], end_time_val - start_time_val))       
        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path)
        print('... Saved model %s ' % checkpoint_path)

        # update loss plot
        plot_losses(train_loss, val_loss, epoch, run=plot_dir)

        # generate and plot posterior samples for the latent space and the parameter space
        if epoch % plot_cadence == 0:
            for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
                mu_r1, z_r1, mu_q, z_q = gen_z_samples(model, x_batch_test, y_batch_test, nsamples=8000)
                plot_latent(mu_r1,z_r1,mu_q,z_q,epoch,step,run=plot_dir)
                start_time_test = time.time()
                samples = gen_samples(model, y_batch_test, ramp=ramp, nsamples=8000)
                end_time_test = time.time()
                if np.any(np.isnan(samples)):
                    print('Epoch: {}, found nans in samples. Not making plots'.format(epoch))
                    for k,s in enumerate(samples):
                        if np.any(np.isnan(s)):
                            print(k,s)
                    KL_est = -1.0
                else:
                    print('Epoch: {}, run {} Testing time elapsed for 8000 samples: {}'.format(epoch,run,end_time_test - start_time_test))
                    KL_est = plot_posterior(samples,x_batch_test[0,:],epoch,step,other_samples=bilby_samples[step,:],run=plot_dir)        
                    _ = plot_posterior(samples,x_batch_test[0,:],epoch,step,run=plot_dir)
                KL_samples.append(KL_est)

            # plot KL evolution
            plot_KL(np.reshape(np.array(KL_samples),[-1,params['r']]),plot_cadence,run=plot_dir)

        # load more noisefree training data back in
        if epoch % 25 == 0:
            x_data_train, y_data_train, _, snrs_train = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'],silent=True)
            train_dataset = (tf.data.Dataset.from_tensor_slices((x_data_train,y_data_train))
                     .shuffle(train_size).batch(batch_size))
        

