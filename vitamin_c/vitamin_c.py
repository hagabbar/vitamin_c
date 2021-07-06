import matplotlib.pyplot as plt
import numpy as np
from sys import exit
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow_probability as tfp
tfd = tfp.distributions
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import corner
import shutil
import h5py
import json
import sys
from sys import exit
from universal_divergence import estimate
import natsort
import plotting
from tensorflow.keras import regularizers
import time
import kerastuner as kt
import matplotlib
from matplotlib.lines import Line2D
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
import pickle

# constants
fourpisq = 4.0*np.pi*np.pi
lntwopi = tf.math.log(2.0*np.pi)
lnfourpi = tf.math.log(4.0*np.pi)

def psiphi_to_psiX(psi,phi):
    """
    Rescales the psi,phi parameters to a different space
    Input and Output in radians
    """
    X = np.remainder(psi+phi,np.pi)
    psi = np.remainder(psi,np.pi/2.0)
    return psi,X

def psiX_to_psiphi(psi,X):
    """
    Rescales the psi,X parameters back to psi, phi
    Input and Output innormalised units [0,1]
    """
    r = np.random.randint(0,2,size=(psi.shape[0],2))
    psi *= np.pi/2.0   # convert to radians 
    X *= np.pi         # convert to radians
    phi = np.remainder(X-psi + np.pi*r[:,0] + r[:,1]*np.pi/2.0,2.0*np.pi).flatten()
    psi = np.remainder(psi + np.pi*r[:,1]/2.0, np.pi).flatten()
    return psi/np.pi,phi/(np.pi*2.0)  # convert back to normalised [0,1]

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

def convert_sky(data, geocent_time, pars, single=False):
    """
    Converts right ascension to hour angle or the reverse
    """
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(geocent_time, format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return np.remainder(t - data,2.0*np.pi)

    for i,k in enumerate(pars):
        try:
            k = k.decode('utf-8')
        except:
            pass
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('...... RA is fixed. Not converting RA to hour angle or the reverse.')
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = np.remainder(t - data[i,ra_idx],2.0*np.pi)

    return data

#def convert_ra_to_hour_angle(data, params, pars, single=False):
#    """
#    Converts right ascension to hour angle and back again
#    """
#
#    greenwich = coord.EarthLocation.of_site('greenwich')
#    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
#    t = t.sidereal_time('mean', 'greenwich').radian
#
#    # compute single instance
#    if single:
#        return t - data
#
#    for i,k in enumerate(pars):
#        if k == 'ra':
#            ra_idx = i
#
#    # Check if RA exist
#    try:
#        ra_idx
#    except NameError:
#        print('...... RA is fixed. Not converting RA to hour angle.')
#    else:
#        # Iterate over all training samples and convert to hour angle
#        for i in range(data.shape[0]):
#            data[i,ra_idx] = t - data[i,ra_idx]
#
#    return data

#def convert_hour_angle_to_ra(data, params, pars, single=False):
#    """
#    Converts right ascension to hour angle and back again
#    """
#    greenwich = coord.EarthLocation.of_site('greenwich')
#    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
#    t = t.sidereal_time('mean', 'greenwich').radian
#
#    # compute single instance
#    if single:
#        return np.remainder(t - data,2.0*np.pi)
#
#    for i,k in enumerate(pars):
#        if k == 'ra':
#            ra_idx = i

    # Check if RA exist
#    try:
#        ra_idx
#    except NameError:
#        print('...... RA is fixed. Not converting RA to hour angle.')
#    else:
#        # Iterate over all training samples and convert to hour angle
#        for i in range(data.shape[0]):
#            data[i,ra_idx] = np.remainder(t - data[i,ra_idx],2.0*np.pi)

#    return data

def load_data(params,bounds,fixed_vals,input_dir,inf_pars,test_data=False,max_samples=np.inf,silent=False):
    """ Function to load either training or testing data.
    """

    # Get list of all training/testing files and define dictionary to store values in files
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
    else:
        print('ERROR: input directory not a string')
        exit(0)

    # Sort files by number index in file name using natsorted program
    filenames = sorted(os.listdir(dataLocations[0]))
    filenames = natsort.natsorted(filenames,reverse=False)

    # If loading by chunks, randomly shuffle list of training/testing filenames
    if params['load_by_chunks'] == True and not test_data:
        nfiles = np.min([int(params['load_chunk_size']/float(params['tset_split'])),len(filenames)])
        files_idx = np.random.randint(0,len(filenames),nfiles)
        filenames= np.array(filenames)[files_idx]
        if not silent:
            print('...... shuffled filenames since we are loading in by chunks')

    # Iterate over all training/testing files and store source parameters, time series and SNR info in dictionary
    data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': [], 'snrs': []}
#    const_check_samplers = ['dynesty','ptemcee','cpnest','emcee'] #params['samplers'][1:]
    const_check_samplers = params['samplers'][1:]
    for filename in filenames:
        if test_data:
            # Don't load files which are not consistent between samplers
            for samp_idx_inner in const_check_samplers:
                inner_file_existance = True
                dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
                filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],int(filename.split('_')[-1].split('.')[0]))
                # If file does not exist, skip to next file
                inner_file_existance = os.path.isfile(filename_inner)
                if inner_file_existance == False:
                    break

            if inner_file_existance == False:
                print('File not consistent beetween samplers')
                continue
        try:
            download_file = h5py.File(dataLocations[0]+'/'+filename, 'r')
            data['x_data'].append(download_file['x_data'][:])
            data['y_data_noisefree'].append(download_file['y_data_noisefree'][:])
            if test_data:
                data['y_data_noisy'].append(download_file['y_data_noisy'][:])
            data['rand_pars'] = download_file['rand_pars'][:] # shape is (15,)
            data['snrs'].append(download_file['snrs'][:]) # shape is (1000,3)
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
        data['snrs'] = np.array(data['snrs'])
    else:
        data['snrs'] = np.concatenate(np.array(data['snrs']), axis=0)

    # Get rid of weird encoding bullshit
    decoded_rand_pars = []
    for i,k in enumerate(data['rand_pars']):
        decoded_rand_pars.append(k.decode('utf-8'))

    # convert ra to hour angle
    if test_data and len(data['x_data'].shape) == 1:
        data['x_data'] = np.expand_dims(data['x_data'], axis=0)
    if test_data:
        data['x_data'] = convert_sky(data['x_data'], params['ref_geocent_time'], decoded_rand_pars)
    else:
        data['x_data'] = convert_sky(data['x_data'], params['ref_geocent_time'], params['rand_pars'])

    # convert psi and phi to reduced parameter space
    for i,k in enumerate(data['rand_pars']):
        if k.decode('utf-8')=='psi':
            psi_idx = i
        if k.decode('utf-8')=='phase':
            phi_idx = i
    # convert phi to X=phi+psi and psi on ranges [0,pi] and [0,pi/2] repsectively - both periodic and in radians
    data['x_data'][:,psi_idx], data['x_data'][:,phi_idx] = psiphi_to_psiX(data['x_data'][:,psi_idx],data['x_data'][:,phi_idx])

    # Normalise the source parameters
    for i,k in enumerate(decoded_rand_pars):
        par_min = k + '_min'
        par_max = k + '_max'

        # normalize by bounds - override psi and phase
        if par_min=='psi_min':
            data['x_data'][:,i]=data['x_data'][:,i]/(np.pi/2.0)
            #data['x_data'][:,i]=np.remainder(data['x_data'][:,i],np.pi)
        elif par_min=='phase_min':
            data['x_data'][:,i]=data['x_data'][:,i]/np.pi
        else:
            data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])

    # extract inference parameters from all source parameters loaded earlier
    idx = []
    infparlist = ''
    for k in inf_pars:
        infparlist = infparlist + k + ', '
        for i,q in enumerate(decoded_rand_pars):
            m = q
            if k==m:
                idx.append(i)
    data['x_data'] = tf.cast(data['x_data'][:,idx],dtype=tf.float32)
    if not silent:
        print('...... {} will be inferred'.format(infparlist))

    return data['x_data'], tf.cast(data['y_data_noisefree'],dtype=tf.float32), tf.cast(data['y_data_noisy'],dtype=tf.float32), data['snrs']

def load_samples(params,sampler,pp_plot=False):
    """
    read in pre-computed posterior samples
    """

    if type("%s" % params['pe_dir']) is str:
        # load generated samples back in
        dataLocations = '%s_%s' % (params['pe_dir'],sampler+'1')
        print('... looking in {} for posterior samples'.format(dataLocations))
    else:
        print('ERROR: input samples directory not a string')
        exit(0)

    # Iterate over requested number of testing samples to use
#    for i in range(params['r']):
    i = i_idx = 0
    for samp_idx,samp in enumerate(params['samplers'][1:]):
        if samp == sampler:
            samp_idx = samp_idx
            break
    while i_idx < params['r']:

        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)
#        const_check_samplers = ['dynesty','ptemcee','cpnest','emcee'] #params['samplers'][1:]
        const_check_samplers = params['samplers'][1:]
        for samp_idx_inner in const_check_samplers:
            inner_file_existance = True
            if samp_idx_inner == samp_idx:
                inner_file_existance = os.path.isfile(filename)
                if inner_file_existance == False:
                    break
                else:
                    continue

            dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
            filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
            # If file does not exist, skip to next file
            inner_file_existance = os.path.isfile(filename_inner)                
            if inner_file_existance == False:
                break

        if inner_file_existance == False:
            i+=1
            print('File does not exist for one of the samplers')
            continue
        #if not os.path.isfile(filename):
        #    print('... unable to find file {}. Exiting.'.format(filename))
        #    exit(0)

        print('... Loading test sample -> ' + filename)
        data_temp = {}
        n = 0

        # Retrieve all source parameters to do inference on
        for q in params['bilby_pars']:
            p = q + '_post'
            par_min = q + '_min'
            par_max = q + '_max'
            data_temp[p] = h5py.File(filename, 'r')[p][:]
            if p == 'psi_post':
                data_temp[p] = np.remainder(data_temp[p],np.pi)
            elif p == 'geocent_time_post':
                data_temp[p] = data_temp[p] - params['ref_geocent_time']
            # Convert samples to hour angle if doing pp plot
            if p == 'ra_post' and pp_plot:
                data_temp[p] = convert_sky(data_temp[p], params['ref_geocent_time'], None, single=True)
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
        rand_idx_posterior = np.linspace(0,Nsamp-1,num=params['n_samples'],dtype=np.int)
        np.random.shuffle(rand_idx_posterior)
        rand_idx_posterior = rand_idx_posterior[:params['n_samples']]
        
        if i == 0:
            XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
        else:
            XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))
        print('... appended {} samples to the total'.format(params['n_samples']))
        i+=1; i_idx+=1
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
    if epoch>1:
        plt.xlim([max(1,epoch-100),epoch+1])
        plt.ylim([np.min(train_loss[max(1,epoch-100):epoch,2])-1.0,np.max(train_loss[max(1,epoch-100):epoch,2])+1.0])
        plt.savefig('%s/loss_latest_total.png' % (run))
    plt.close()

    # save loss data to text file
    loss_file = '%s/loss.txt' % (run)
    data = np.concatenate([train_loss[:epoch,:],val_loss[:epoch,:]],axis=1)
    np.savetxt(loss_file,data)

def plot_KL(KL_samples, step, run='testing'):
    """
    plots the KL evolution
    """
    # arrives in shape n_kl,n_test,3
    N = KL_samples.shape[0]
#    KL_samples = np.transpose(KL_samples,[2,1,0])   # re-order axes
#    print(list(np.linspace(0,len(params['samplers'][1:])-1,num=len(params['samplers'][1:]), dtype=int))[::-1])
#    print(KL_samples.shape)
#    exit()
#    KL_samples = np.transpose(KL_samples, list(np.linspace(0,len(params['samplers'][1:])-1,num=len(params['samplers'][1:]), dtype=int))[::-1])    
    ls = ['-','--',':']
    c = ['C0','C1','C2','C3']
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(6.4,14.4))
    for i,kl_s in enumerate(KL_samples):   # loop over samplers
        for j,kl in enumerate(kl_s):     # loop over test cases
            axs[i].semilogx(np.arange(1,N+1)*step,kl,ls[i],color=c[j])
            axs[i].plot(N*step,kl[-1],'.',color=c[j])
            axs[i].grid()
    plt.xlabel('epoch')
    plt.ylabel('KL')
    plt.ylim([-0.2,1.0])
    plt.savefig('%s/kl.png' % (run))
    plt.close()


def plot_posterior(samples,x_truth,epoch,idx,snrs_test=None,noisy_sig=None,noisefree_sig=None,run='testing',all_other_samples=None):
    """
    plots the posteriors
    """

    # Check for optional basemap installation
    try:
        from mpl_toolkits.basemap import Basemap
        print("module 'basemap' is installed")
    except (ModuleNotFoundError, ImportError):
        print("module 'basemap' is not installed")
        print("Skyplotting functionality is automatically disabled.")
        skyplotting_usage = False
    else:
        skyplotting_usage = True
        try:
            from .skyplotting import plot_sky
        except:
            from skyplotting import plot_sky

    NskyPlotSamples = 200

    # check is basemap is installed
    if skyplotting_usage:
        params['Make_sky_plot'] = True
    else:
        params['Make_sky_plot'] = False

    # Turn off Make sky plot if desired
    params['Make_sky_plot'] = False

    # trim samples from outside the cube
    mask = []
    for s in samples:
        if (np.all(s>=0.0) and np.all(s<=1.0)):
            mask.append(True)
        else:
            mask.append(False)
    samples = tf.boolean_mask(samples,mask,axis=0)
    print('identified {} good samples'.format(samples.shape[0]))
    print(np.array(all_other_samples).shape)
    if samples.shape[0]<100:
        return [-1.0,-1.0,-1.0]

    # randonly convert from reduced psi-phi space to full space
    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
    psi_idx = psi_idx[0]
    phi_idx = phi_idx[0]
    samples = samples.numpy()
    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])

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
    hist_kwargs_other2 = dict(density=True,color='tab:green')
    hist_kwargs_other3 = dict(density=True, color='tab:purple')
    hist_kwargs_other4 = dict(density=True, color='tab:orange')

    custom_lines = []
    if all_other_samples is not None:
        KL_est = []
        color_wheel = ['tab:blue','tab:green','tab:purple','tab:orange']
        for i, other_samples in enumerate(all_other_samples):
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
            true_XS = convert_sky(true_XS,params['ref_geocent_time'],ol_pars)
            true_x = convert_sky(np.reshape(true_x,[1,true_XS.shape[1]]),params['ref_geocent_time'],ol_pars).flatten()
            old_true_post = true_post

            samples_file = '{}/posterior_samples_epoch_{}_event_{}_vit.txt'.format(run,epoch,idx)
            np.savetxt(samples_file,true_XS)

            # compute KL estimate
            idx1 = np.random.randint(0,true_XS.shape[0],2000)
            idx2 = np.random.randint(0,true_post.shape[0],2000)
            try:
                current_KL = -1.0 #0.5*(estimate(true_XS[idx1,:],true_post[idx2,:],n_jobs=4) + estimate(true_post[idx2,:],true_XS[idx1,:],n_jobs=4))
            except:
                current_KL = -1.0
                pass
            KL_est.append(current_KL)

            other_samples_file = '{}/posterior_samples_epoch_{}_event_{}_{}.txt'.format(run,epoch,idx,i)
            np.savetxt(other_samples_file,true_post)

            if i==0:
                figure = corner.corner(true_post, **defaults_kwargs,labels=parnames,
                           color=color_wheel[i],
                           show_titles=True, hist_kwargs=hist_kwargs_other)
            else:

                # compute KL estimate
                idx1 = np.random.randint(0,old_true_post.shape[0],2000)
                idx2 = np.random.randint(0,true_post.shape[0],2000)
                try:
                    current_KL = -1.0 #0.5*(estimate(old_true_post[idx1,:],true_post[idx2,:],n_jobs=4) + estimate(true_post[idx2,:],old_true_post[idx1,:],n_jobs=4))
                except:
                    current_KL = -1.0
                    pass
                KL_est.append(current_KL)

                if i == 1:
                    corner.corner(true_post,**defaults_kwargs,
                           color=color_wheel[i],
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other2)
                elif i == 2:
                    corner.corner(true_post,**defaults_kwargs,
                           color=color_wheel[i],
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other3) 
                elif i == 3:
                    corner.corner(true_post,**defaults_kwargs,
                           color=color_wheel[i],
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs_other4)
            custom_lines.append(Line2D([0], [0], color=color_wheel[i], lw=4))
    
        # Plot vitamin posteriors
        corner.corner(true_XS,**defaults_kwargs,
                           color='tab:red',
                           fill_contours=False, truths=true_x,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
        custom_lines.append(Line2D([0], [0], color='tab:red', lw=4))

        if params['Make_sky_plot'] == True:
            print()
            print('... Generating sky plot')
            print()
            left, bottom, width, height = [0.46, 0.6, 0.5, 0.5] # switch with waveform positioning
            ax_sky = figure.add_axes([left, bottom, width, height])

            sky_color_cycle=['blue','green','purple','orange']
            sky_color_map_cycle=['Blues','Greens','Purples','Oranges']
            for i, other_samples in enumerate(all_other_samples):
                true_post = np.zeros([other_samples.shape[0],bilby_ol_len])
                cnt = 0
                for inf_idx,bilby_idx in zip(inf_ol_idx,bilby_ol_idx):
                    bilby_par = params['bilby_pars'][bilby_idx]
                    true_post[:,cnt] = (other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
                    cnt += 1
                if i == 0:
                    ax_sky = plot_sky(true_post[:NskyPlotSamples,skymap_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[skymap_mask])
                else:
                    ax_sky = plot_sky(true_post[:NskyPlotSamples,skymap_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[skymap_mask],ax=ax_sky)
            ax_sky = plot_sky(true_XS[:NskyPlotSamples,skymap_mask],filled=True,trueloc=true_x[skymap_mask],ax=ax_sky)

        matplotlib.rc('text', usetex=True)
        if noisy_sig != None:
            # Plot GW noisy and noisefree waveform
            left, bottom, width, height = [0.67, 0.48, 0.3, 0.2] # swtiched with skymap positioning
            ax2 = figure.add_axes([left, bottom, width, height])
            ax2.plot(np.linspace(0,1,params['ndata']),noisefree_sig,color='cyan',zorder=50)
            snrs_tot = np.round(np.sqrt(np.sum(snrs_test**2)), 2)
            ax2.plot(np.linspace(0,1,params['ndata']),noisy_sig,color='darkblue', 
                     label='Network SNR: '+str(snrs_tot)+'; IndiSNRs: ['+str(np.round(snrs_test[0],2))+', '+str(np.round(snrs_test[1],2))+', '+str(np.round(snrs_test[2],2))+']')
            ax2.set_xlabel(r"$\textrm{time (seconds)}$",fontsize=16)
            ax2.yaxis.set_visible(False)
            ax2.tick_params(axis="x", labelsize=12)
            ax2.tick_params(axis="y", labelsize=12)
            ax2.set_ylim([-6,6])
            ax2.grid(False)
            ax2.margins(x=0,y=0)
            ax2.legend()

        # Add figure sampler names legend box
        corner_names = params['figure_sampler_names'][1:][:len(params['samplers'])-1] + ['VItamin']
        figure.legend(handles=custom_lines, labels=corner_names,
                      loc=(0.86,0.22), fontsize=20)

        if epoch == 'pub_plot':
            print('Saved output to %s/comp_posterior_%s_event_%d.png' % (run,epoch,idx))
            plt.savefig('%s/comp_posterior_%s_event_%d.png' % (run,epoch,idx))
        else:
            print('Saved output to %s/comp_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
            plt.savefig('%s/comp_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
        return KL_est

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
        new_samples = convert_sky(new_samples,params['ref_geocent_time'],params['inf_pars'])
        full_true_x = convert_sky(np.reshape(full_true_x,[1,samples.shape[1]]),params['ref_geocent_time'],params['inf_pars']).flatten()

        figure = corner.corner(new_samples,**defaults_kwargs,labels=parnames,
                           color='tab:red',
                           fill_contours=False, truths=full_true_x,
                           show_titles=True, hist_kwargs=hist_kwargs)
        if epoch == 'pub_plot':
            plt.savefig('%s/full_posterior_%s_event_%d.png' % (run,epoch,idx))
        else:
            plt.savefig('%s/full_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
        plt.close()
    return -1.0

def plot_latent(mu_r1, logw_r1, z_r1, mu_q, z_q, epoch, idx, run='testing'):

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

    figure = corner.corner(np.array(z_q), **defaults_kwargs,
                           color='tab:blue',
                           show_titles=True, hist_kwargs=hist_kwargs_other)
    corner.corner(np.array(z_r1),**defaults_kwargs,
                           color='tab:red',
                           fill_contours=True,
                           show_titles=True, fig=figure, hist_kwargs=hist_kwargs)
    # Extract the axes
    z_dim = z_r1.shape[1]
    axes = np.array(figure.axes).reshape((z_dim, z_dim))

    # Loop over the histograms and plot means
    for yi in range(z_dim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(mu_r1[0,:,xi], mu_r1[0,:,yi], "sr")
            ax.plot(mu_q[0,xi], mu_q[0,yi], "sb")

    if epoch == 'pub_plot':
        plt.savefig('%s/latent_%s_event_%d.png' % (run,epoch,idx))
    else:
        plt.savefig('%s/latent_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()

    # plot the weights
    plt.figure()
    plt.step(np.arange(params['n_modes']), logw_r1[0,:], where='mid')
    plt.xlabel('z dimension')
    plt.ylabel('log(w)')
    if epoch == 'pub_plot':
        plt.savefig('%s/latent_logweight_%s_event_%d.png' % (run,epoch,idx))
    else:
        plt.savefig('%s/latent_logweight_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()
    plt.figure()
    w = np.exp(logw_r1[0,:] - np.max(logw_r1[0,:]))
    w = w/np.sum(w)
    plt.step(np.arange(params['n_modes']), w, where='mid')
    plt.xlabel('z dimension')
    plt.ylabel('w')
    plt.ylim([0,np.max(w)*1.1])
    if epoch == 'pub_plot':
        plt.savefig('%s/latent_weight_%s_event_%d.png' % (run,epoch,idx))
    else:
        plt.savefig('%s/latent_weight_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()

def plot_r2hist(data, epoch, idx, name='test', run='testing'):

    # define general plotting arguments
    defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=[0.16, 0.84],
                    levels=(0.68,0.90,0.95), density=True,
                    plot_density=False, plot_datapoints=True,
                    max_n_ticks=3)

    # Get corner parnames to use in plotting labels
    parnames = []
    for k_idx,k in enumerate(params['rand_pars']):
        if np.isin(k, params['inf_pars']):
            parnames.append(params['corner_labels'][k])

    # 1-d hist kwargs for normalisation
    hist_kwargs = dict(density=True,color='tab:red')

    try:
        figure = corner.corner(np.array(data) + np.random.normal(loc=0.0,scale=1e-6,size=data.shape), **defaults_kwargs,
                           color='tab:red', #labels=parnames,
                           show_titles=True, hist_kwargs=hist_kwargs)
        plt.savefig('/data/www.astro/chrism/vitamin_c/%s/r2_%s_%d_event_%d.png' % (run,name,epoch,idx))
        plt.close()
    except:
        print('Unable to plot r2 hist data {}'.format(name))
        pass

params = './params_files/params.json'
bounds = './params_files/bounds.json'
fixed_vals = './params_files/fixed_vals.json'
run = time.strftime('%y-%m-%d-%X-%Z')
EPS = 1e-6

# Load parameters files
with open(params, 'r') as fp:
    params = json.load(fp)
with open(bounds, 'r') as fp:
    bounds = json.load(fp)
with open(fixed_vals, 'r') as fp:
    fixed_vals = json.load(fp)

# if doing hour angle, use hour angle bounds on RA
inf_ol_mask, inf_ol_idx, inf_ol_len = get_param_index(params['inf_pars'],params['bilby_pars'])
bilby_ol_mask, bilby_ol_idx, bilby_ol_len = get_param_index(params['bilby_pars'],params['inf_pars'])

# identify the indices of different sets of physical parameters
vonmise_mask, vonmise_idx_mask, vonmise_len = get_param_index(params['inf_pars'],params['vonmise_pars'])
gauss_mask, gauss_idx_mask, gauss_len = get_param_index(params['inf_pars'],params['gauss_pars'])
sky_mask, sky_idx_mask, sky_len = get_param_index(params['inf_pars'],params['sky_pars'])
skymap_mask, skymap_idx_mask, skymap_len = get_param_index(params['bilby_pars'],params['sky_pars'])
ra_mask, ra_idx_mask, ra_len = get_param_index(params['inf_pars'],['ra'])
dec_mask, dec_idx_mask, dec_len = get_param_index(params['inf_pars'],['dec'])
m1_mask, m1_idx_mask, m1_len = get_param_index(params['inf_pars'],['mass_1'])
m2_mask, m2_idx_mask, m2_len = get_param_index(params['inf_pars'],['mass_2'])
psi_mask, psi_idx_mask, psi_len = get_param_index(params['inf_pars'],['psi'])
psiphi_mask, psiphi_idx_mask, psiphi_len = get_param_index(params['inf_pars'],['psi','phase'])
periodic_mask, periodic_idx_mask, periodic_len = get_param_index(params['inf_pars'],['phase','psi','phi_12','phi_jl'])
nonperiodic_mask, nonperiodic_idx_mask, nonperiodic_len = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2'])
#idx_mask = np.argsort(gauss_idx_mask + vonmise_idx_mask + m1_idx_mask + m2_idx_mask + sky_idx_mask) # + dist_idx_mask)
idx_mask = np.argsort(m1_idx_mask + m2_idx_mask + gauss_idx_mask + vonmise_idx_mask) # + sky_idx_mask)
dist_mask, dist_idx_mask, dist_len = get_param_index(params['inf_pars'],['luminosity_distance'])
not_dist_mask, not_dist_idx_mask, not_dist_len = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])
phase_mask, phase_idx_mask, phase_len = get_param_index(params['inf_pars'],['phase'])
not_phase_mask, not_phase_idx_mask, not_phase_len = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','geocent_time','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])
geocent_mask, geocent_idx_mask, geocent_len = get_param_index(params['inf_pars'],['geocent_time'])
not_geocent_mask, not_geocent_idx_mask, not_geocent_len = get_param_index(params['inf_pars'],['mass_1','mass_2','luminosity_distance','psi','phase','theta_jn','ra','dec','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])
xyz_mask, xyz_idx_mask, xyz_len = get_param_index(params['inf_pars'],['luminosity_distance','ra','dec'])
not_xyz_mask, not_xyz_idx_mask, not_xyz_len = get_param_index(params['inf_pars'],['mass_1','mass_2','psi','phase','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'])
idx_xyz_mask = np.argsort(xyz_idx_mask + not_xyz_idx_mask)
idx_dist_mask = np.argsort(not_dist_idx_mask + dist_idx_mask)
idx_phase_mask = np.argsort(not_phase_idx_mask + phase_idx_mask)
idx_geocent_mask = np.argsort(not_geocent_idx_mask + geocent_idx_mask)
idx_periodic_mask = np.argsort(nonperiodic_idx_mask + periodic_idx_mask + ra_idx_mask + dec_idx_mask)
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

print(nonperiodic_len,periodic_len)

# define which gpu to use during training
gpu_num = str(params['gpu_num'])   
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
print('... running on GPU {}'.format(gpu_num))

# Let GPU consumption grow as needed
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
print('... letting GPU consumption grow as needed')

train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_log_dir = params['plot_dir'] + '/logs'

make_psiphi_plots = False

if make_psiphi_plots:
    #### TEST the PSI PHI stuff
    """ Choose random number between 0 and 1 for an 
    array of 6 x 2 values. Each of the 6 represent a multivariate Gaussian dist. """
    mu = np.random.uniform(0,1,(6,2))
    Xpsi = []
    """ Iterate over those 6 values in the array """
    for i in range(6):
        """ Sample from a random multivariate normal distribution with 
        random means (mu) and a standard deviation of 0.01 long diagnal. Generate 
        3000 samples for each distribution."""
        Xpsi.append(np.random.multivariate_normal(mu[i,:],0.01*np.diag(np.ones(2)),size=3000))
    Xpsi = np.array(Xpsi).reshape(18000,2)
    X1 = np.remainder(Xpsi[:,0]*np.pi,np.pi)   # X = psi + phi
    psi1 = np.remainder(Xpsi[:,1]*np.pi/2.0,np.pi/2.0)  # psi
    fig, ax = plt.subplots(2,2,figsize=(15,15))
    ax[0,0].plot(X1,psi1,',')
    ax[0,0].plot(mu[:,0]*np.pi,mu[:,1]*np.pi/2.0,'xr')
    ax[0,0].set_xlim([0,np.pi])
    ax[0,0].set_xlabel(r'$X = \psi + \phi$')
    ax[0,0].set_ylim([0,np.pi/2.0])
    ax[0,0].set_ylabel(r'$\psi$')

    ax[0,1].plot(X1-psi1,psi1,',')
    ax[0,1].set_xlim([-np.pi,3*np.pi/2.0])
    ax[0,1].set_xlabel(r'$X - \psi$')
    ax[0,1].set_ylim([-np.pi/4,3*np.pi/4.0])
    ax[0,1].set_ylabel(r'$\psi$')

    psi2,phi2 = psiX_to_psiphi(psi1/(np.pi/2.0),X1/np.pi)
    ax[1,0].plot(phi2*2*np.pi,psi2*np.pi,',')
    ax[1,0].set_xlim([0,2*np.pi])
    ax[1,0].set_xlabel(r'$\phi$')
    ax[1,0].set_ylim([0,np.pi])
    ax[1,0].set_ylabel(r'$\psi$')

    psi3,X3 = psiphi_to_psiX(psi2*np.pi,phi2*2*np.pi)
    ax[1,1].plot(X3,psi3,',')
    ax[1,1].set_xlim([0,np.pi])
    ax[1,1].set_xlabel(r'$X$')
    ax[1,1].set_ylim([0,np.pi/2.0])
    ax[1,1].set_ylabel(r'$\psi$')
    plt.savefig('{}/Xpsi.png'.format(params['plot_dir']))
    plt.close()
    exit()

    x_data_train, y_data_train, _, snrs_train = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'])

    fig, ax = plt.subplots(3,3,sharex=True,sharey=True,figsize=(15,15))
    idx = 0
    t = params['duration']*np.arange(params['ndata'])/float(params['ndata'])
    for i in range(3):
        for j in range(3):
            ax[i,j].plot(t,y_data_train[idx,:,0],'b')
            ax[i,j].plot(t,y_data_train[idx,:,1],'r')
            ax[i,j].plot(t,y_data_train[idx,:,2],'g')
            idx += 1
    plt.savefig('{}/train_signals.png'.format(params['plot_dir']))
    plt.close()

    x_data_val, y_data_val, _, snrs_val = load_data(params,bounds,fixed_vals,params['val_set_dir'],params['inf_pars'])

    fig, ax = plt.subplots(3,3,sharex=True,sharey=True,figsize=(15,15))
    idx = 0
    t = params['duration']*np.arange(params['ndata'])/float(params['ndata'])
    for i in range(3):
        for j in range(3):
            ax[i,j].plot(t,y_data_val[idx,:,0],'b')
            ax[i,j].plot(t,y_data_val[idx,:,1],'r')
            ax[i,j].plot(t,y_data_val[idx,:,2],'g')
            idx += 1
    plt.savefig('{}/val_signals.png'.format(params['plot_dir']))
    plt.close()

    x_data_test, y_data_test_noisefree, y_data_test, snrs_test = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],max_samples=params['r'],test_data=True)

    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,15))
    idx = 0
    t = params['duration']*np.arange(params['ndata'])/float(params['ndata'])
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(t,y_data_test_noisefree[idx,:,0],'b')
            ax[i,j].plot(t,y_data_test_noisefree[idx,:,1],'r')
            ax[i,j].plot(t,y_data_test_noisefree[idx,:,2],'g')
            idx += 1
    plt.savefig('{}/test_signals.png'.format(params['plot_dir']))
    plt.close()

    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(15,15))
    idx = 0
    t = params['duration']*np.arange(params['ndata'])/float(params['ndata'])
    for i in range(2):
        for j in range(2):
            ax[i,j].plot(t,y_data_test[idx,:,0],'b')
            ax[i,j].plot(t,y_data_test[idx,:,1],'r')
            ax[i,j].plot(t,y_data_test[idx,:,2],'g')
            idx += 1
    plt.savefig('{}/test_data.png'.format(params['plot_dir']))
    plt.close()

    fig,ax = plt.subplots(14,14,figsize=(15,15))
    for i in range(14):
        for j in range(14):
            if j<i:
                ax[i,j].plot(x_data_train[:,j],x_data_train[:,i],',b')
                ax[i,j].plot(x_data_val[:,j],x_data_val[:,i],',g')
                ax[i,j].plot(x_data_test[:,j],x_data_test[:,i],'xk')
                ax[i,j].set_xlabel(params['inf_pars'][j])
                ax[i,j].set_ylabel(params['inf_pars'][i])
    plt.savefig('{}/train_prior.png'.format(params['plot_dir']))
    plt.close()
    print('... Finished making phi-psi diagnostic plots!')
    exit()

def compute_loss(model, x, y, ramp=1.0, noise_ramp=1.0):

    # for mixed precision stuff
    ramp = tf.cast(ramp,dtype=tf.float32)

    # add noise and randomise distance again
    y = (y + noise_ramp*tf.random.normal(shape=tf.shape(y), mean=0.0, stddev=1.0, dtype=tf.float32))/params['y_normscale']

    # define r1 encoder
    mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y,training=True)
    gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=tf.cast(logweight_r1,dtype=tf.float32)),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.cast(mean_r1,dtype=tf.float32),
            scale_diag=tf.cast(EPS + tf.sqrt(tf.exp(logvar_r1)),dtype=tf.float32)))

    # define q encoder
    mean_q, logvar_q = model.encode_q(x=x,y=y)
    mvn_q = tfp.distributions.MultivariateNormalDiag(
            loc=mean_q,
            scale_diag=EPS + tf.sqrt(tf.exp(logvar_q)))
    z_samp = mvn_q.sample()

    # define r2 encoder
    mean_np_r2, mean_p_r2, mean_s_r2, logvar_np_r2, logvar_p_r2, logvar_s_r2 = model.decode_r2(z=z_samp,y=y)

    
    # truncated normal for non-periodic params
    tmvn_np_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.cast(mean_np_r2,dtype=tf.float32),
            scale=tf.cast(EPS + tf.sqrt(tf.exp(logvar_np_r2)),dtype=tf.float32),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)
    tmvn_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(tmvn_np_r2.log_prob(tf.boolean_mask(x,nonperiodic_mask,axis=1)),axis=1),axis=0)

    # 2D representations of periodic params
    mean_x_r2, mean_y_r2 = tf.split(mean_p_r2, num_or_size_splits=2, axis=1)   # split the means into x and y coords [size (batch,p) each]
    mean_angle_r2 = tf.math.floormod(tf.math.atan2(tf.cast(mean_y_r2,dtype=tf.float32),tf.cast(mean_x_r2,dtype=tf.float32)),2.0*np.pi)  # the mean angle (0,2pi) [size (batch,p)]
    scale_p_r2 = EPS + tf.sqrt(tf.exp(logvar_p_r2))  # define the 2D Gaussian scale [size (batch,2*p)]
    vm_r2 = tfp.distributions.VonMises(
            loc=tf.cast(mean_angle_r2,dtype=tf.float32), #tf.cast(2.0*np.pi*mean_p_r2,dtype=tf.float32), 
            concentration=tf.cast(tf.math.reciprocal(EPS + fourpisq*tf.exp(logvar_p_r2)),dtype=tf.float32)
    )
    vm_r2_cost_recon = -1.0*tf.reduce_mean(tf.reduce_sum(lntwopi + vm_r2.log_prob(2.0*np.pi*tf.boolean_mask(x,periodic_mask,axis=1)),axis=1),axis=0)

    # Fisher Von Mises
    fvm_loc = tf.reshape(tf.math.l2_normalize(mean_s_r2, axis=1),[-1,3])
    fvm_con = tf.squeeze(tf.math.reciprocal(EPS + tf.exp(logvar_s_r2)), axis=1)
    fvm_r2 = tfp.distributions.VonMisesFisher(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
    )
    ra_sky = tf.reshape(2*np.pi*tf.boolean_mask(x,ra_mask,axis=1),(-1,1))       # convert the scaled 0->1 true RA value back to radians
    dec_sky = tf.reshape(np.pi*(tf.boolean_mask(x,dec_mask,axis=1) - 0.5),(-1,1)) # convert the scaled 0>1 true dec value back to radians
    xyz_unit = tf.concat([tf.cos(ra_sky)*tf.cos(dec_sky),tf.sin(ra_sky)*tf.cos(dec_sky),tf.sin(dec_sky)],axis=1)   # construct the true parameter unit vector
    test = tf.math.l2_normalize(xyz_unit,axis=1)
    test2 = fvm_r2.log_prob(tf.reshape(test,(-1,3)))
    fvm_r2_cost_recon = -1.0*tf.reduce_mean(lnfourpi + fvm_r2.log_prob(test),axis=0)

    simple_cost_recon = tmvn_r2_cost_recon + vm_r2_cost_recon #+ fvm_r2_cost_recon

    selfent_q = -1.0*tf.reduce_mean(mvn_q.entropy())
    log_r1_q = gm_r1.log_prob(tf.cast(z_samp,dtype=tf.float32))   # evaluate the log prob of r1 at the q samples
    cost_KL = tf.cast(selfent_q,dtype=tf.float32) - tf.reduce_mean(log_r1_q)
    return simple_cost_recon, cost_KL

def ramp_func(epoch,start,ramp_length, n_cycles):
    i = (epoch-start)/(2.0*ramp_length)
    print(epoch,i)
    if i<0:
        return 0.0
    if i>=n_cycles:
        return 1.0
    return min(1.0,2.0*np.remainder(i,1.0))

def gen_z_samples(model, x, y, nsamples=1000,max_samples=1000):
        
    y = y/params['y_normscale']
    y = tf.tile(y,(max_samples,1,1))
    x = tf.tile(x,(max_samples,1))
    samp_iterations = int(nsamples/max_samples)
    for i in range(samp_iterations):
        
        mean_r1_temp, logvar_r1, logweight_r1_temp = model.encode_r1(y=y)
        scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logweight_r1_temp),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=mean_r1_temp,
            scale_diag=scale_r1))
        mean_q_temp, logvar_q = model.encode_q(x=x,y=y)
        scale_q = EPS + tf.sqrt(tf.exp(logvar_q))
        mvn_q = tfp.distributions.MultivariateNormalDiag(
                          loc=mean_q_temp,
                          scale_diag=scale_q)
        if i==0:
            z_samp_q = mvn_q.sample()
            z_samp_r1 = gm_r1.sample()
            logweight_r1 = logweight_r1_temp
            mean_r1 = mean_r1_temp
            mean_q = mean_q_temp
        else:
            z_samp_q = tf.concat([z_samp_q,mvn_q.sample()],axis=0)
            z_samp_r1 = tf.concat([z_samp_r1,gm_r1.sample()],axis=0)
            logweight_r1 = tf.concat([logweight_r1,logweight_r1_temp],axis=0)
            mean_r1 = tf.concat([mean_r1,mean_r1_temp],axis=0)
            mean_q = tf.concat([mean_q,mean_q_temp],axis=0)

    return mean_r1, logweight_r1, z_samp_r1, mean_q, z_samp_q

def paper_plots(test_dataset, y_data_test, x_data_test, model, params, plot_dir, run, bilby_samples, y_data_test_noisefree, snrs_test):
    """ Make publication plots
    """
    epoch = 'pub_plot'; ramp = 1
    plotter = plotting.make_plots(params, None, None, x_data_test)
    try:
        os.mkdir('plotting_data_%s' % params['run_label'])
    except:
        pass
    try:
        os.mkdir('%s/latest_%s' % (plot_dir,params['run_label']))
    except:
        pass

    """
    # Make corner plots
    for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
        # Make mode plots
#        mode_samples = gen_mode_samples(model,y_batch_test, ramp=ramp, nsamples=1000) # 4,1000,15
#        plot_mode_posterior(mode_samples,x_batch_test[0,:],epoch,step,run=plot_dir)
        # Make latent plots
        mu_r1, logw_r1, z_r1, mu_q, z_q = gen_z_samples(model, x_batch_test, y_batch_test, nsamples=8000)
        plot_latent(mu_r1,logw_r1,z_r1,mu_q,z_q,epoch,step,run=plot_dir)
        exit()
        if int(step) == 41 or int(step) == 184 or int(step) == 213 or int(step)==164:
            start_time_test = time.time()
            samples,_,_ = gen_samples(model, y_batch_test, ramp=ramp, nsamples=params['n_samples'])
            end_time_test = time.time()
            if np.any(np.isnan(samples)):
                print('Found nans in samples. Not making plots')
                for k,s in enumerate(samples):
                    if np.any(np.isnan(s)):
                        print(k,s)
                KL_est = [-1,-1,-1]
            else:
                print('Run {} Testing time elapsed for {} samples: {}'.format(run,params['n_samples'],end_time_test - start_time_test))
                KL_est = plot_posterior(samples,x_batch_test[0,:],epoch,step,snrs_test[int(step),:],y_batch_test[0,:,0],y_data_test_noisefree[int(step),:,0],all_other_samples=bilby_samples[:,step,:],run=plot_dir)
                _ = plot_posterior(samples,x_batch_test[0,:],epoch,step,snrs_test[int(step),:],y_batch_test[0,:,0],y_data_test_noisefree[int(step),:,0],run=plot_dir)
    print('... Finished making posterior plots! Congrats fam.')
    exit()
    """

    # Make p-p plots
#    plotter.plot_pp(model, y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
#    print('... Finished making p-p plots!')
#    exit()    

    # Make JS plots
#    plotter.gen_kl_plots(model,y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
#    print('... Finished making KL plots!')    
#    exit() 

    # JS vs SNR plot
#    JS_data = h5py.File('plotting_data_%s/JS_plot_data.h5' % params['run_label'], 'r')
#    plotting.JS_vs_SNR_plot(JS_data, snrs_test, params)
#    exit()

    # Make individual parameter JS plots
    plotting.indiPar_JS_plots(model,y_data_test, x_data_test, params, bounds, inf_ol_idx, bilby_ol_idx)
    exit()

    return


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    """
    def __init__(self, x_dim, y_dim, n_channels, z_dim, n_modes, ramp, hp):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.x_modes = 1   # hardcoded for testing
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.act = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.kern_reg = regularizers.l2(0.001)
        self.ramp = ramp
        self.hp = hp
        first_dense = self.hp.Int('first_dense', min_value=3, max_value=90, step=3)
        second_dense = self.hp.Int('second_dense', min_value=3, max_value=90, step=3)
        third_dense = self.hp.Int('third_dense', min_value=3, max_value=90, step=3)
        first_filt = self.hp.Int('first_filt', min_value=3, max_value=90, step=3)
        second_filt = self.hp.Int('second_filt', min_value=3, max_value=90, step=3)
        third_filt = self.hp.Int('third_filt', min_value=3, max_value=90, step=3)
        fourth_filt = self.hp.Int('fourth_filt', min_value=3, max_value=90, step=3)
        fifth_filt = self.hp.Int('fifth_filt', min_value=3, max_value=90, step=3)
        sixth_filt = self.hp.Int('sixth_filt', min_value=3, max_value=90, step=3)
        first_filt_size = self.hp.Int('first_filt_size', min_value=3, max_value=9, step=3)
        second_filt_size = self.hp.Int('second_filt_size', min_value=3, max_value=9, step=3)
        third_filt_size = self.hp.Int('third_filt_size', min_value=3, max_value=9, step=3)
        fourth_filt_size = self.hp.Int('fourth_filt_size', min_value=3, max_value=9, step=3)
        fifth_filt_size = self.hp.Int('fifth_filt_size', min_value=3, max_value=9, step=3)
        sixth_filt_size = self.hp.Int('sixth_filt_size', min_value=3, max_value=9, step=3)
        first_filt_stride = self.hp.Int('first_filt_stride', min_value=1, max_value=1, step=1)
        second_filt_stride = self.hp.Int('second_filt_stride', min_value=1, max_value=1, step=1)
        third_filt_stride = self.hp.Int('third_filt_stride', min_value=1, max_value=1, step=1)
        fourth_filt_stride = self.hp.Int('fourth_filt_stride', min_value=1, max_value=1, step=1)
        fifth_filt_stride = self.hp.Int('fifth_filt_stride', min_value=1, max_value=1, step=1)
        sixth_filt_stride = self.hp.Int('sixth_filt_stride', min_value=1, max_value=1, step=1)


        # the r1 encoder network
        r1_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        a = tf.keras.layers.Conv1D(filters=first_filt, kernel_size=first_filt_size, strides=first_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(r1_input_y)
        a = tf.keras.layers.Conv1D(filters=second_filt, kernel_size=second_filt_size, strides=second_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a = tf.keras.layers.Conv1D(filters=third_filt, kernel_size=third_filt_size, strides=third_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a = tf.keras.layers.Conv1D(filters=fourth_filt, kernel_size=fourth_filt_size, strides=fourth_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a = tf.keras.layers.Conv1D(filters=fifth_filt, kernel_size=fifth_filt_size, strides=fifth_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a = tf.keras.layers.Conv1D(filters=sixth_filt, kernel_size=sixth_filt_size, strides=sixth_filt_stride, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a = tf.keras.layers.Flatten()(a)
#        a = tf.keras.layers.Flatten()(r1_input_y)
        a2 = tf.keras.layers.Dense(first_dense, kernel_regularizer=self.kern_reg, activation=self.act)(a)
        a2 = tf.keras.layers.Dropout(.1)(a2)
        a2 = tf.keras.layers.Dense(second_dense, kernel_regularizer=self.kern_reg, activation=self.act)(a2)
        a2 = tf.keras.layers.Dropout(.1)(a2)
        a2 = tf.keras.layers.Dense(third_dense, kernel_regularizer=self.kern_reg, activation=self.act)(a2)
        a2 = tf.keras.layers.Dense(2*self.z_dim*self.n_modes + self.n_modes)(a2)
        self.encoder_r1 = tf.keras.Model(inputs=r1_input_y, outputs=a2)
        #print(self.encoder_r1.summary())

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        c = tf.keras.layers.Flatten()(q_input_x)
        d = tf.keras.layers.concatenate([a,c])        
        e = tf.keras.layers.Dense(first_dense, kernel_regularizer=self.kern_reg, activation=self.act)(d)
        e = tf.keras.layers.Dropout(.1)(e)
        e = tf.keras.layers.Dense(second_dense, kernel_regularizer=self.kern_reg, activation=self.act)(e)
        e = tf.keras.layers.Dropout(.1)(e)
        e = tf.keras.layers.Dense(third_dense, kernel_regularizer=self.kern_reg, activation=self.act)(e)
        e = tf.keras.layers.Dense(2*self.z_dim)(e)
        self.encoder_q = tf.keras.Model(inputs=[r1_input_y, q_input_x], outputs=e)
        #print(self.encoder_q.summary())

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        g = tf.keras.layers.Flatten()(r2_input_z)
        h = tf.keras.layers.concatenate([a,g])
        i = tf.keras.layers.Dense(first_dense, kernel_regularizer=self.kern_reg, activation=self.act)(h)
        i = tf.keras.layers.Dropout(.1)(i)
        i = tf.keras.layers.Dense(second_dense, kernel_regularizer=self.kern_reg, activation=self.act)(i)
        i = tf.keras.layers.Dropout(.1)(i)
        i = tf.keras.layers.Dense(third_dense, kernel_regularizer=self.kern_reg, activation=self.act)(i)
        j = tf.keras.layers.Dense(2*self.x_dim*self.x_modes + self.x_modes)(i)
        self.decoder_r2 = tf.keras.Model(inputs=[r1_input_y, r2_input_z], outputs=j)
        #print(self.decoder_r2.summary())
    """

    def __init__(self, x_dim_np, x_dim_p, y_dim, n_channels, z_dim, n_modes, ramp, curr_total_epoch, same_real, ramp_start, ramp_length, ramp_cycles, n_modes_r2=1):
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.n_modes = n_modes
        self.n_modes_r2 = n_modes_r2
        self.x_modes = 1   # hardcoded for testing
        self.x_dim_np = x_dim_np
        self.x_dim_p = x_dim_p
        self.x_dim = self.x_dim_np + self.x_dim_p + 2 # extra 2 for sky
        self.y_dim = y_dim
        self.n_channels = n_channels
        self.act = 'relu'
        self.kern_reg = regularizers.l2(0.001)
        self.ramp = ramp
        self.curr_total_epoch = curr_total_epoch
        self.same_real = same_real
        self.ramp_start = ramp_start
        self.ramp_length = ramp_length
        self.ramp_cycles = ramp_cycles

        # the r1 network
        all_input_y = tf.keras.Input(shape=(self.y_dim, self.n_channels))
        r1c = tf.keras.layers.Conv1D(filters=512, kernel_size=32, strides=1, activation=self.act)(all_input_y)
        r1c = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_1')(r1c)
        r1c = tf.keras.layers.Conv1D(filters=256, kernel_size=16, strides=1, activation=self.act)(r1c)
        r1c = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_2')(r1c)
        r1c = tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation=self.act)(r1c)
        r1c = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_3')(r1c)
        r1c = tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=1, activation=self.act)(r1c)
        r1c = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_4')(r1c)
        r1c = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, activation=self.act)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_5')(r1c)
#        r1c = tf.keras.layers.Conv1D(filters=96, kernel_size=16, strides=2, kernel_regularizer=self.kern_reg, activation=self.act)(r1c)
#        r1c = tf.keras.layers.BatchNormalization(name='batchnorm_6')(r1c)
        r1c = tf.keras.layers.Flatten()(r1c)
#        r1c = tf.keras.layers.Flatten()(all_input_y)
        r1 = tf.keras.layers.Dense(4096,activation=self.act)(r1c)
        r1 = tf.keras.layers.Dense(2048,activation=self.act)(r1)
        r1 = tf.keras.layers.Dense(1024,activation=self.act)(r1)
        r1mu = tf.keras.layers.Dense(self.z_dim*self.n_modes,use_bias=True,bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))(r1)
        r1logvar = tf.keras.layers.Dense(self.z_dim*self.n_modes,use_bias=True,bias_initializer=tf.keras.initializers.Constant(value=-1.0))(r1) #tf.keras.initializers.RandomUniform(minval=-4.0, maxval=0.0))(r1)
        r1w = tf.keras.layers.Dense(self.n_modes,use_bias=True,bias_initializer='Zeros')(r1)
        r1 = tf.keras.layers.concatenate([r1mu,r1logvar,r1w])
        self.encoder_r1 = tf.keras.Model(inputs=all_input_y, outputs=r1)
        print(self.encoder_r1.summary())

        # the q encoder network
        q_input_x = tf.keras.Input(shape=(self.x_dim))
        qx = tf.keras.layers.Flatten()(q_input_x)
        q = tf.keras.layers.concatenate([r1c,qx])
        q = tf.keras.layers.Dense(4096,activation=self.act)(q)
        q = tf.keras.layers.Dense(2048,activation=self.act)(q)
        q = tf.keras.layers.Dense(1024,activation=self.act)(q)
        q = tf.keras.layers.Dense(2*self.z_dim)(q)
        self.encoder_q = tf.keras.Model(inputs=[all_input_y, q_input_x], outputs=q)
        print(self.encoder_q.summary())

        # the r2 decoder network
        r2_input_z = tf.keras.Input(shape=(self.z_dim))
        r2z = tf.keras.layers.Flatten()(r2_input_z)
        r2 = tf.keras.layers.concatenate([r1c,r2z])
        r2 = tf.keras.layers.Dense(4096,activation=self.act)(r2)
        r2 = tf.keras.layers.Dense(2048,activation=self.act)(r2)
        r2 = tf.keras.layers.Dense(1024,activation=self.act)(r2)
        r2mu_np = tf.keras.layers.Dense(self.x_dim_np,activation='sigmoid')(r2)
        r2logvar_np = tf.keras.layers.Dense(self.x_dim_np,use_bias=True)(r2)
        r2mu_p = tf.keras.layers.Dense(2*self.x_dim_p,use_bias=True)(r2)
        r2logvar_p = tf.keras.layers.Dense(self.x_dim_p,use_bias=True)(r2)
        r2mu_s = tf.keras.layers.Dense(3,use_bias=True)(r2)
        r2logvar_s = tf.keras.layers.Dense(1,use_bias=True)(r2)
        r2 = tf.keras.layers.concatenate([r2mu_np,r2mu_p,r2mu_s,r2logvar_np,r2logvar_p,r2logvar_s])
        self.decoder_r2 = tf.keras.Model(inputs=[all_input_y, r2_input_z], outputs=r2)
        print(self.decoder_r2.summary())

#    def build(self, input_shape):
#        pass

#    def call():
#        return

    def encode_r1(self, y=None,training=True):
        mean, logvar, weight = tf.split(self.encoder_r1(y,training=training), num_or_size_splits=[self.z_dim*self.n_modes, self.z_dim*self.n_modes,self.n_modes], axis=1)
        mean = tf.reshape(mean,[-1,self.n_modes,self.z_dim])
        logvar = tf.reshape(logvar,[-1,self.n_modes,self.z_dim])
        weight = tf.reshape(weight,[-1,self.n_modes])
        return mean, logvar, weight

    def encode_q(self, x=None, y=None):
        return tf.split(self.encoder_q([y,x]), num_or_size_splits=[self.z_dim,self.z_dim], axis=1)

    def decode_r2(self, y=None, z=None, apply_sigmoid=False):
        return tf.split(self.decoder_r2([y,z]), num_or_size_splits=[self.x_dim_np, 2*self.x_dim_p, 3, self.x_dim_np, self.x_dim_p, 1], axis=1)

    def compile(self, optimizer, loss):
        super(CVAE, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        """Executes one training step and returns the loss.
        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        x, y = data

        with tf.GradientTape() as tape:
            r_loss, kl_loss = self.loss(self, x, y, ramp=self.ramp)
            loss = r_loss + self.ramp*kl_loss
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"r_train_loss": r_loss, "kl_train_loss": kl_loss, "total_train_loss": loss, "ramp": self.ramp}

    def test_step(self, data):
        """ Executes one validation step
        """
        x, y = data

        r_loss, kl_loss = self.loss(self, x, y, ramp=self.ramp)
        loss = r_loss + self.ramp*kl_loss
        return {"r_loss": r_loss, "kl_loss": kl_loss, "total_loss": loss}

def tf_load_dataset(filename):
    """ tf.data enabled loading function
    
    Params
    ------

    Returns
    -------
    dataset: tf.data
        tensorflow dataset object
    """
    print(filename)
    exit()

    return 

def tf_load_dataset_wrapper(filename):
    """ Load dataset wrapper
    """
    x_data_train, y_data_train = tf.py_function(
       load_data, [filename,params,bounds,fixed_vals], Tout=tf.float32) 
    return x_data_train, y_data_train

def gen_samples(model, y, ramp=1.0, nsamples=1000, max_samples=1000):

    y = y/params['y_normscale']
    y = tf.tile(y,(max_samples,1,1))
    samp_iterations = int(nsamples/max_samples)
    ramp = tf.cast(ramp,dtype=tf.float32)
    for i in range(samp_iterations):
        mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y)
        scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
        gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=tf.cast(logweight_r1,dtype=tf.float32)),
            components_distribution=tfd.MultivariateNormalDiag(
            loc=tf.cast(mean_r1,dtype=tf.float32),
            scale_diag=tf.cast(scale_r1,dtype=tf.float32)))
        z_samp = gm_r1.sample()
        mean_np_r2_temp, mean_p_r2_temp, mean_s_r2_temp, logvar_np_r2_temp, logvar_p_r2_temp, logvar_s_r2_temp = model.decode_r2(z=z_samp,y=y)
        scale_np_r2 = EPS + tf.sqrt(tf.exp(logvar_np_r2_temp))
        tmvn_np_r2 = tfp.distributions.TruncatedNormal(
            loc=tf.cast(mean_np_r2_temp,dtype=tf.float32),
            scale=tf.cast(scale_np_r2,dtype=tf.float32),
            low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)
        mean_x_r2, mean_y_r2 = tf.split(mean_p_r2_temp, num_or_size_splits=2, axis=1)
        mean_angle_r2 = tf.math.floormod(tf.math.atan2(tf.cast(mean_y_r2,dtype=tf.float32),tf.cast(mean_x_r2,dtype=tf.float32)),2.0*np.pi)


        vm_r2 = tfp.distributions.VonMises(
            loc=tf.cast(mean_angle_r2,dtype=tf.float32), #tf.cast(2.0*np.pi*mean_p_r2_temp,dtype=tf.float32),
            concentration=tf.cast(tf.math.reciprocal(EPS + fourpisq*tf.exp(logvar_p_r2_temp)),dtype=tf.float32)
        )

        fvm_loc = tf.reshape(tf.math.l2_normalize(mean_s_r2_temp, axis=1),[max_samples,3])
        fvm_con = tf.reshape(tf.math.reciprocal(EPS + tf.exp(logvar_s_r2_temp)),[max_samples])
        fvm_r2 = tfp.distributions.VonMisesFisher(
            mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
            concentration = tf.cast(fvm_con,dtype=tf.float32)
        )

        tmvn_x_sample = tmvn_np_r2.sample()
        vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)  # samples are output on range [-pi,pi] so rescale to [0,1]
        xyz = tf.reshape(fvm_r2.sample(),[max_samples,3])          # sample the distribution
        samp_ra = tf.math.floormod(tf.math.atan2(tf.slice(xyz,[0,1],[max_samples,1]),tf.slice(xyz,[0,0],[max_samples,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
        samp_dec = (tf.asin(tf.slice(xyz,[0,2],[max_samples,1])) + 0.5*np.pi)/np.pi                       # convert to the rescaled 0->1 dec from the unit vector
        fvm_x_sample = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[max_samples,2])             # group the sky samples        

        if i==0:
            x_sample = tf.gather(tf.concat([tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),idx_periodic_mask,axis=1)
            mean_r2 = tf.concat([mean_np_r2_temp,mean_p_r2_temp,mean_s_r2_temp],axis=1)
            logvar_r2 = tf.concat([logvar_np_r2_temp,logvar_p_r2_temp,logvar_s_r2_temp],axis=1)
        else:
            x_sample = tf.concat([x_sample,tf.gather(tf.concat([tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),idx_periodic_mask,axis=1)],axis=0)
            mean_r2 = tf.concat([mean_r2,tf.concat([mean_np_r2_temp,mean_p_r2_temp,mean_s_r2_temp],axis=1)],axis=0)
            logvar_r2 = tf.concat([logvar_r2,tf.concat([logvar_np_r2_temp,logvar_p_r2_temp,logvar_s_r2_temp],axis=1)],axis=0)
    return x_sample, mean_r2, logvar_r2

def gen_mode_samples(model, y, ramp=1.0, nsamples=1000, max_samples=1000):

    y = y/params['y_normscale']
    y = tf.tile(y,(max_samples,1,1))
    samp_iterations = int(nsamples/max_samples)
    ramp = tf.cast(ramp,dtype=tf.float32)
    x_sample = []
    for k in range(min(4,params['n_modes'])):
        for i in range(samp_iterations):
            mean_r1, logvar_r1, logweight_r1 = model.encode_r1(y=y)
            scale_r1 = EPS + tf.sqrt(tf.exp(logvar_r1))
            idx = np.argsort(logweight_r1[0,:])[::-1]    # get order of weights in reverse order
            weights = np.zeros(params['n_modes'])
            weights[idx[k]] = 1.0   # make the k'th highest mode the only mode   
            gm_r1 = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=tf.cast(weights,dtype=tf.float32)),
                components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.cast(mean_r1,dtype=tf.float32),
                scale_diag=tf.cast(scale_r1,dtype=tf.float32)))
            z_samp = gm_r1.sample()
            mean_np_r2, mean_p_r2, mean_s_r2, logvar_np_r2, logvar_p_r2, logvar_s_r2 = model.decode_r2(z=z_samp,y=y)
            scale_np_r2 = EPS + tf.sqrt(tf.exp(logvar_np_r2))
            tmvn_np_r2 = tfp.distributions.TruncatedNormal(
                loc=tf.cast(mean_np_r2,dtype=tf.float32),
                scale=tf.cast(scale_np_r2,dtype=tf.float32),
                low=-10.0 + ramp*10.0, high=1.0 + 10.0 - ramp*10.0)
            mean_cos_r2, mean_sin_r2 = tf.split(mean_p_r2, num_or_size_splits=2, axis=1)
            mean_angle_r2 = tf.math.floormod(tf.math.atan2(tf.cast(mean_sin_r2,dtype=tf.float32),tf.cast(mean_cos_r2,dtype=tf.float32)),2.0*np.pi)

            # rotate sample in +ve mean_angle direction
            vm_r2 = tfp.distributions.VonMises(
                loc=tf.cast(mean_angle_r2,dtype=tf.float32), #tf.cast(2.0*np.pi*mean_p_r2,dtype=tf.float32),
                concentration=tf.cast(tf.math.reciprocal(EPS + fourpisq*tf.exp(logvar_p_r2)),dtype=tf.float32)
            )
            fvm_loc = tf.reshape(tf.math.l2_normalize(mean_s_r2, axis=1),[max_samples,3])
            fvm_con = tf.reshape(tf.math.reciprocal(EPS + tf.exp(logvar_s_r2)),[max_samples])
            fvm_r2 = tfp.distributions.VonMisesFisher(
                mean_direction = tf.cast(fvm_loc,dtype=tf.float32),
                concentration = tf.cast(fvm_con,dtype=tf.float32)
            )

            tmvn_x_sample = tmvn_np_r2.sample()
            vm_x_sample = tf.math.floormod(vm_r2.sample(),(2.0*np.pi))/(2.0*np.pi)  # samples are output on range [-pi,pi] so rescale to [0,1]
            xyz = tf.reshape(fvm_r2.sample(),[max_samples,3])          # sample the distribution
            samp_ra = tf.math.floormod(tf.math.atan2(tf.slice(xyz,[0,1],[max_samples,1]),tf.slice(xyz,[0,0],[max_samples,1])),2.0*np.pi)/(2.0*np.pi)   # convert to the rescaled 0->1 RA from the unit vector
            samp_dec = (tf.asin(tf.slice(xyz,[0,2],[max_samples,1])) + 0.5*np.pi)/np.pi                       # convert to the rescaled 0->1 dec from the unit vector
            fvm_x_sample = tf.reshape(tf.concat([samp_ra,samp_dec],axis=1),[max_samples,2])             # group the sky samples
            if i==0:
                temp_x_sample = tf.gather(tf.concat([tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),idx_periodic_mask,axis=1)
            else:
                temp_x_sample = tf.concat([temp_x_sample,tf.gather(tf.concat([tmvn_x_sample,vm_x_sample,fvm_x_sample],axis=1),idx_periodic_mask,axis=1)],axis=0)
        x_sample.append(temp_x_sample)
    return x_sample

def plot_mode_posterior(samples,x_truth,epoch,idx,run='testing'):
    """
    plots the posteriors in individual latent space modes
    """

    # randonly convert from reduced psi-phi space to full space
    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
    psi_idx = psi_idx[0]
    phi_idx = phi_idx[0]

    # define general plotting arguments
    defaults_kwargs = dict(
                    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
                    title_kwargs=dict(fontsize=16),
                    truth_color='tab:orange', quantiles=None,
                    levels=[0.90], density=True,
                    plot_density=False, plot_datapoints=False,
                    max_n_ticks=3)

    col = ['tab:blue',
           'tab:orange',
           'tab:green',
           'tab:red',
           'tab:purple',
           'tab:brown',
           'tab:pink',
           'tab:gray',
           'tab:olive',
           'tab:cyan']

    par_range = inf_ol_len*[(1e16,-1e16)]
    for i, sample in enumerate(samples):
        sample = sample.numpy()
        # convert back to psi and phi space from X and psi - now on normalised ranges [0,1], and [0,1] respectively 
        sample[:,psi_idx], sample[:,phi_idx] = psiX_to_psiphi(sample[:,psi_idx], sample[:,phi_idx])
        true_x = np.zeros(inf_ol_len)
        true_XS = np.zeros([sample.shape[0],inf_ol_len])
        ol_pars = []
        cnt = 0
        for inf_idx in inf_ol_idx:
            inf_par = params['inf_pars'][inf_idx]
            true_XS[:,cnt] = (sample[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
            true_x[cnt] = (x_truth[inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
            ol_pars.append(inf_par)
            if np.min(true_XS[:,cnt])<par_range[cnt][0]:
                par_range[cnt] = (np.min(true_XS[:,cnt]),par_range[cnt][1])
            if np.max(true_XS[:,cnt])>par_range[cnt][1]:
                par_range[cnt] = (par_range[cnt][0],np.max(true_XS[:,cnt]))
            #par_range[cnt,1] = bounds[inf_par+'_max']
            cnt += 1
        parnames = []
        for k_idx,k in enumerate(params['rand_pars']):
            if np.isin(k, ol_pars):
                parnames.append(params['corner_labels'][k])

        # convert to RA
        true_XS = convert_sky(true_XS,params['ref_geocent_time'],ol_pars)
        true_x = convert_sky(np.reshape(true_x,[1,true_XS.shape[1]]),params['ref_geocent_time'],ol_pars).flatten()

        hist_kwargs = dict(density=True,color=col[np.remainder(i,len(col))])
        if i==0:
            try:
                figure = corner.corner(true_XS, **defaults_kwargs,labels=parnames,
                           color=col[np.remainder(i,len(col))], truths=true_x,
                           show_titles=True, range=par_range, hist_kwargs=hist_kwargs)
            except:
                return
        else:
            try:
                corner.corner(true_XS,**defaults_kwargs,
                           color=col[np.remainder(i,len(col))],
                           show_titles=True, range=par_range, fig=figure, hist_kwargs=hist_kwargs)
            except:
                return

    #snr_tot = np.sqrt(np.sum(snrs**2))
    #plt.annotate('SNR = {:.3f} [{:.3f},{:.3f},{:.3f}]'.format(snr_tot,snrs[0],snrs[1],snrs[2]),(0.6,0.95),xycoords='figure fraction',fontsize=18)
    plt.savefig('%s/modes_posterior_epoch_%d_event_%d.png' % (run,epoch,idx))
    plt.close()

def run_vitc(params, x_data_train, y_data_train, x_data_val, y_data_val, x_data_test, y_data_test, y_data_test_noisefree, save_dir, truth_test, bounds, fixed_vals, bilby_samples, snrs_test=None):

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
    hyper_par_tune = False
    """ Skip doing any plotting involving test samples"""
    skip_test_plotting = params['use_real_events']
    initial_learning_rate=1e-4
    optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    """ Currently following: https://stackoverflow.com/questions/55363728/how-to-feed-h5-files-in-tf-data-pipeline-in-tensorflow-model
    and https://www.tensorflow.org/guide/data#applying_arbitrary_python_logic
    """
#    list_ds = tf.data.Dataset.list_files(params['train_set_dir'])
#    file_path = next(iter(list_ds))
    # test load a training set sample
#    dataset = tf_load_dataset_wrapper(file_path)
#    print(dataset)
#    exit()    

    # load the training data
#    if not make_paper_plots:
    x_data_train, y_data_train, _, snrs_train = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'])
    x_data_val, y_data_val, _, snrs_val = load_data(params,bounds,fixed_vals,params['val_set_dir'],params['inf_pars'])
    if not skip_test_plotting:
        x_data_test, y_data_test_noisefree, y_data_test, snrs_test = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],test_data=True)
        y_data_test = y_data_test[:params['r'],:,:]; x_data_test = x_data_test[:params['r'],:]

        # load precomputed samples
        bilby_samples = []
        for sampler in params['samplers'][1:]:
#        for sampler in ['dynesty']:
            bilby_samples.append(load_samples(params,sampler))
        bilby_samples = np.array(bilby_samples)
    # If deadling with real signals
    else:
        #x_data_test, y_data_test_noisefree, y_data_test, snrs_test = load_data(params,bounds,fixed_vals,params['test_set_dir'],params['inf_pars'],test_data=True)
        #y_data_test = y_data_test[:params['r'],:,:]; x_data_test = x_data_test[:params['r'],:]
        print('Was able to load real event waveform by itself')
        exit()

    # Convert datasets to tensorflow
#    if not make_paper_plots:
    if not hyper_par_tune:
        train_dataset = (tf.data.Dataset.from_tensor_slices((x_data_train,y_data_train))
                         .shuffle(train_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE))
    else:
        train_dataset = (tf.data.Dataset.from_tensor_slices(({'image': y_data_train},{'label': x_data_train}))
                         .shuffle(train_size))#.batch(batch_size))
    val_dataset = (tf.data.Dataset.from_tensor_slices((x_data_val,y_data_val))
                    .shuffle(val_size).batch(batch_size))
    if not skip_test_plotting:
        test_dataset = (tf.data.Dataset.from_tensor_slices((x_data_test,y_data_test))
                        .batch(1))

    train_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # start the training loop
    train_loss = np.zeros((epochs,3))
    val_loss = np.zeros((epochs,3))
    ramp_start = params['ramp_start']
    ramp_length = params['ramp_end']
    ramp_cycles = 1
    KL_samples = []

    #####################
    # Callbacks section #
    #####################
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
#            current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
#            print("current decayed lr: {:0.7f}".format(current_decayed_lr))

            # apply ramp
            ramp = tf.convert_to_tensor(ramp_func((self.model.curr_total_epoch+tf.cast(epoch, tf.float32)),self.model.ramp_start,self.model.ramp_length,self.model.ramp_cycles), dtype=tf.float32)
            self.model.ramp.assign(tf.Variable(ramp, trainable=False))


    #reduce_lr = ReduceLROnPlateau(monitor='train_loss', factor=0.2,
    #                          patience=5, min_lr=1e-5)

    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard_logs/%s' % params['run_label'],
                                 histogram_freq=0,
                                 write_graph=True,
                                 write_images=False,
                                 update_freq='epoch',
                                 profile_batch=2,
                                 embeddings_freq=0,
                                 embeddings_metadata=None,
                                 )
    TerminateOnNan = tf.keras.callbacks.TerminateOnNaN()
    EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                monitor='val_total_loss', min_delta=0, patience=3, verbose=0,
                                mode='auto', baseline=None, restore_best_weights=False
                                )
    #####################
    # Callbacks section #
    #####################

    # Keras hyperparameter optimization
    hyper_par_tune = False
    ramp = tf.Variable(1.0, trainable=False)
    if hyper_par_tune:
        def model_builder(hp):
          model = CVAE(x_data_test.shape[1], params['ndata'],
                     y_data_test.shape[2], params['z_dimension'], params['n_modes'], ramp, hp)
          # Choose an optimal value from 0.01, 0.001, or 0.0001
          hp_learning_rate = hp.Choice('learning_rate', values=[5e-4, 1e-4, 5e-5])

          model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss=compute_loss
                        )

          return model

        tuner = kt.Hyperband(model_builder,
                     objective=kt.Objective("val_total_loss", direction="min"),
                     max_epochs=50,
                     directory='hyperpar_tuning_testing',
                     project_name='%s' % params['run_label'])
        tuner.search(train_dataset, epochs=50, validation_data=val_dataset, 
                     callbacks=[EarlyStopping])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(best_hps.values)       
 
        # Save current best hyperparameters
        #np.savetxt('hyperpar_tuning_testing/%s' % params['run_label'], best_hps)

        print(f"""
        The hyperparameter search is complete.
        """)
        exit()

    # log params used for this run
    path = params['plot_dir']
    shutil.copy('./vitamin_c.py',path)
    shutil.copy('./params_files/params.json',path)
    decay_steps=100000
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=0.96,
    staircase=True)
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
    initial_learning_rate, decay_steps)
    same_real = 4

    # Make publication plots
    make_paper_plots = params['make_paper_plots']
    if make_paper_plots:
        print('... Making plots for publication.')
        # Load the previously saved weights
        model = CVAE(int(nonperiodic_len), int(periodic_len), params['ndata'],
                         y_data_test.shape[2], params['z_dimension'], params['n_modes'], ramp,
                         tf.Variable(0, trainable=False, dtype=tf.float32), same_real, ramp_start, ramp_length, ramp_cycles)    
        
        ramp = tf.convert_to_tensor(1.0)
        model.ramp.assign(tf.Variable(ramp, trainable=False))
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        """
        # Load optimizer weights from previously saved file
        with open('inverse_model_%s/opt_weights.pkl' % params['run_label'], 'rb') as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)
        print('... Loaded opimizer weights inverse_model_%s ' % params['run_label'])

        # Load and set batchnorm weights from previously trained model
        with open('inverse_model_%s/batchnorm_weights.pkl' % params['run_label'], 'rb') as f:
            batchnorm_weights = pickle.load(f)
        batchnorm_idx = 0
        for net in model.layers:
            if net.name == 'model' or net.name=='model_1' or net.name=='model_2':
                for layer in net.layers:
                    if layer.name == 'batchnorm':
                        layer.set_weights(batchnorm_weights[batchnorm_idx][0])
                        if batchnorm_idx == 0:
                            print()
                            print(layer.get_weights())
                        batchnorm_idx+=1
        """

        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
        paper_plots(test_dataset, y_data_test, x_data_test, model, params, plot_dir, run, bilby_samples, y_data_test_noisefree, snrs_test)
        return

    model = CVAE(int(nonperiodic_len), int(periodic_len), params['ndata'],
                         y_data_train.shape[2], params['z_dimension'], params['n_modes'], ramp,
                         tf.Variable(0, trainable=False, dtype=tf.float32), same_real, ramp_start, ramp_length, ramp_cycles)
    model.compile(
            optimizer=optimizer,
            loss=compute_loss,
        )

    if params['resume_training']:
        """
        # Load optimizer weights from previously saved file
        with open('inverse_model_%s/opt_weights.pkl' % params['run_label'], 'rb') as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)
        print('... Loaded opimizer weights inverse_model_%s ' % params['run_label'])

        # Load and set batchnorm weights from previously trained model
        with open('inverse_model_%s/batchnorm_weights.pkl' % params['run_label'], 'rb') as f:
            batchnorm_weights = pickle.load(f)
        batchnorm_idx = 0
        for net in model.layers:
            if net.name == 'model' or net.name=='model_1' or net.name=='model_2':
                for layer in net.layers:
                    if layer.name == 'batchnorm':
                        layer.set_weights(batchnorm_weights[batchnorm_idx][0])
                        if batchnorm_idx == 0:
                            print()
                            print(layer.get_weights())
                        batchnorm_idx+=1
        """

        # Load the previously saved weights
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model.load_weights(latest)
        print('... loading in previous model %s' % checkpoint_path)
    else:
        print('... Training new model.')


    for epoch in range(1, epochs + 1):

        if params['resume_training'] or not params['ramp']:
            ramp = tf.convert_to_tensor(1.0)
            model.ramp.assign(tf.Variable(ramp, trainable=False))
            print('... Not using ramp.')
        else:
#        ramp = tf.convert_to_tensor(ramp_func((epoch)*same_real,ramp_start,ramp_length,ramp_cycles), dtype=tf.float32)
#        model.ramp.assign(tf.Variable(ramp, trainable=False)) 
            model.curr_total_epoch.assign(tf.Variable((epoch-1)*same_real, trainable=False, dtype=tf.float32))

        # Fit the model
        if params['resume_training']:
            history = model.fit(train_dataset, epochs=same_real, batch_size=params['batch_size'],
                            callbacks=[CustomCallback(), TerminateOnNan, tboard_callback], 
                            validation_data=val_dataset) # CustomCallback may cause problem.
        else:
            history = model.fit(train_dataset, epochs=same_real, batch_size=params['batch_size'], 
                            callbacks=[CustomCallback(), TerminateOnNan, tboard_callback], 
                            validation_data=val_dataset)

        train_loss[((epoch-1)*same_real):((epoch)*same_real), 0] = history.history['r_train_loss'] 
        train_loss[((epoch-1)*same_real):((epoch)*same_real), 1] = history.history['kl_train_loss']
        train_loss[((epoch-1)*same_real):((epoch)*same_real), 2] = history.history['total_train_loss']
        val_loss[((epoch-1)*same_real):((epoch)*same_real), 0] = history.history['val_r_loss']
        val_loss[((epoch-1)*same_real):((epoch)*same_real), 1] = history.history['val_kl_loss']
        val_loss[((epoch-1)*same_real):((epoch)*same_real), 2] = history.history['val_total_loss']

        # Plot loss
        plot_losses(train_loss, val_loss, epoch*same_real, run=plot_dir)

        # generate and plot posterior samples for the latent space and the parameter space 
        plot_cadence = int(params['plot_interval']/same_real)
        #plot_cadence = int(1)
        if epoch % plot_cadence == 0 and not skip_test_plotting:
            for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
                # Make mode plots
                mode_samples = gen_mode_samples(model,y_batch_test, ramp=ramp, nsamples=1000)
                plot_mode_posterior(mode_samples,x_batch_test[0,:],epoch,step,run=plot_dir)
                # Make latent plots
                mu_r1, logw_r1, z_r1, mu_q, z_q = gen_z_samples(model, x_batch_test, y_batch_test, nsamples=8000)
                plot_latent(mu_r1,logw_r1,z_r1,mu_q,z_q,epoch*same_real,step,run=plot_dir)
                start_time_test = time.time()
                samples, mu_r2, logvar_r2 = gen_samples(model, y_batch_test, ramp=ramp, nsamples=params['n_samples'])
                # Make r2 histogram plots
                #plot_r2hist(tf.squeeze(mu_r2),epoch,step,name='mu',run=run)
                #plot_r2hist(tf.squeeze(logvar_r2),epoch,step,name='logvar',run=run)
                end_time_test = time.time()
                if np.any(np.isnan(samples)):
                    print('Epoch: {}, found nans in samples. Not making plots'.format(epoch*same_real))
                    for k,s in enumerate(samples):
                        if np.any(np.isnan(s)):
                            print(k,s)
                    KL_est = [-1,-1,-1]
                else:
                    print('Epoch: {}, run {} Testing time elapsed for {} samples: {}'.format(epoch*same_real,run,params['n_samples'],end_time_test - start_time_test))
                    KL_est = plot_posterior(samples,x_batch_test[0,:],epoch*same_real,step,all_other_samples=bilby_samples[:,step,:],run=plot_dir)
                    _ = plot_posterior(samples,x_batch_test[0,:],epoch*same_real,step,run=plot_dir)
                KL_samples.append(KL_est)

        # Load new data
        print()
        print('Total iteration %d' % (epoch*same_real))
        print()
        x_data_train, y_data_train, _, snrs_train = load_data(params,bounds,fixed_vals,params['train_set_dir'],params['inf_pars'],silent=True)

        # augment the data for this chunk - distqnce
        print('... Adding extra distance marginalization.')
        old_d = bounds['luminosity_distance_min'] + tf.boolean_mask(x_data_train,dist_mask,axis=1)*(bounds['luminosity_distance_max'] - bounds['luminosity_distance_min'])
        new_x = tf.random.uniform(shape=tf.shape(old_d), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        new_d = bounds['luminosity_distance_min'] + new_x*(bounds['luminosity_distance_max'] - bounds['luminosity_distance_min'])
        x_data_train = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x_data_train,not_dist_mask,axis=1),[-1,tf.shape(x_data_train)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(idx_dist_mask),axis=1)
        dist_scale = tf.tile(tf.expand_dims(old_d/new_d,axis=1),(1,tf.shape(y_data_train)[1],1))

        # augment the data for this chunk - randomize phase and arrival time
        print('... Adding extra phase randomization.')
        old_phase = tf.boolean_mask(x_data_train,phase_mask,axis=1)*np.pi   # this is the X parameters X = psi+phi and has range [0,pi]
        new_x = tf.random.uniform(shape=tf.shape(old_phase), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        new_phase = new_x*np.pi   # this is the X parameters X = psi+phi and has range [0,pi]
        x_data_train = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x_data_train,not_phase_mask,axis=1),[-1,tf.shape(x_data_train)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(idx_phase_mask),axis=1)

        if np.any([r=='geocent_time' for r in params['inf_pars']]):

            old_geocent = bounds['geocent_time_min'] + tf.boolean_mask(x_data_train,geocent_mask,axis=1)*(bounds['geocent_time_max'] - bounds['geocent_time_min'])
            new_x = tf.random.uniform(shape=tf.shape(old_geocent), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
            new_geocent = bounds['geocent_time_min'] + new_x*(bounds['geocent_time_max'] - bounds['geocent_time_min'])
            x_data_train = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x_data_train,not_geocent_mask,axis=1),[-1,tf.shape(x_data_train)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(idx_geocent_mask),axis=1)
            phase_correction = -1.0*tf.complex(tf.cos(2.0*(new_phase-old_phase)),tf.sin(2.0*(new_phase-old_phase)))
            phase_correction = tf.tile(tf.expand_dims(phase_correction,axis=1),(1,len(params['det']),tf.shape(y_data_train)[1]/2 + 1))
            fvec = tf.range(params['ndata']/2 + 1)/params['duration']
            time_correction = -2.0*np.pi*fvec*(new_geocent-old_geocent)
            time_correction = tf.complex(tf.cos(time_correction),tf.sin(time_correction))
            time_correction = tf.tile(tf.expand_dims(time_correction,axis=1),(1,len(params['det']),1))

        y_data_train_fft = tf.signal.rfft(tf.transpose(y_data_train,[0,2,1]))*phase_correction*time_correction
        y_data_train = tf.transpose(tf.signal.irfft(y_data_train_fft),[0,2,1])*dist_scale

        train_dataset = (tf.data.Dataset.from_tensor_slices((x_data_train,y_data_train))
                 .shuffle(train_size).batch(batch_size))

        # Save model weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path)
        print('... Saved model %s ' % checkpoint_path)

        # Save optimizer weights using the `checkpoint_path` format
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open('inverse_model_%s/opt_weights.pkl' % params['run_label'], 'wb') as f:
            pickle.dump(weight_values, f)
        print('... Saved opimizer weights inverse_model_%s ' % params['run_label'])

        # Get all batch normalization weight information
        batch_norm_weights = []
        for net in model.layers:
            if net.name == 'model' or net.name=='model_1' or net.name=='model_2':
                for layer in net.layers:
                    if layer.name.split('_')[0] == 'batchnorm':
                        batch_norm_weights.append([layer.get_weights()])
        # Save batch normalization weight information
        with open('inverse_model_%s/batchnorm_weights.pkl' % params['run_label'], 'wb') as f:
            pickle.dump(batch_norm_weights, f)
        print('... Saved batchnorm weights inverse_model_%s ' % params['run_label'])
