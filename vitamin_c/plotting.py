""" Script containing a suite of plotting-related functions used by vitamin_b
"""

from __future__ import division
from decimal import *
import os, shutil
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import numpy as np
from scipy.stats import uniform, norm, gaussian_kde, ks_2samp, anderson_ksamp
from scipy import stats
import scipy
from scipy.integrate import dblquad
import h5py
#from ligo.skymap.plot import PPPlot
import bilby
from universal_divergence import estimate
import pandas as pd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FixedLocator,
                               AutoMinorLocator)
import matplotlib.ticker as ticker
from lal import GreenwichMeanSiderealTime
from vitamin_c import gen_samples, load_samples, convert_sky, get_param_index, psiX_to_psiphi

def sky_JS_decade_plots(params, JS_data, test_dataset, model, all_other_samples, bilby_ol_idx, bilby_ol_len, bounds, sky_mask, inf_ol_idx, inf_ol_len):
    """ Make sky only plots at specific JS divergence 
    decadel intervals (i.e. 10^-2, 10^-1, 10^0).
    """
    ramp = 1; NskyPlotSamples = 200
    JS_data = np.array(JS_data['dynesty-vitamin'])
    matplotlib.rc('text', usetex=True)

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

    zero_found = minus1_found = minus2_found = False
    for JS_ele_idx,JS_ele in enumerate(JS_data):
        # Check for 10^0
        if JS_ele >= (1e0-1e-1) and JS_ele <= (1e0+1e-1) and not zero_found:
            #if JS_ele_idx == 36:
            #    continue
            zero_found=True
            zero_idx = JS_ele_idx
            print('... Found JS for 10^zero: %.6f' % JS_ele)

            for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
                if int(step) == zero_idx:
                    samples,_ = gen_samples(model, y_batch_test, ramp=ramp, nsamples=params['n_samples'])
                    # trim samples from outside the cube
                    mask = []
                    for s in samples:
                        if (np.all(s>=0.0) and np.all(s<=1.0)):
                            mask.append(True)
                        else:
                            mask.append(False)
                    samples = tf.boolean_mask(samples,mask,axis=0)
                    print('identified {} good samples'.format(samples.shape[0]))
                    if samples.shape[0]<100:
                        return [-1.0,-1.0,-1.0]

                    # randonly convert from reduced psi-phi space to full space
                    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
                    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
                    psi_idx = psi_idx[0]
                    phi_idx = phi_idx[0]
                    samples = samples.numpy()
                    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])
                    print()
                    print('... Generating sky plot')
                    print()
                    fig, ax = plt.subplots(figsize=(12,6))

                    sky_color_cycle=['blue','green','purple','orange']
                    sky_color_map_cycle=['Blues','Greens','Purples','Oranges']
                    for i, other_samples in enumerate(all_other_samples[:,step,:]):
                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])

                        true_x = np.zeros(inf_ol_len)
                        true_XS = np.zeros([samples.shape[0],inf_ol_len])
                        cnt= 0; ol_pars=[]
                        for inf_idx in inf_ol_idx:
                            inf_par = params['inf_pars'][inf_idx]
                            true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                            true_x[cnt] = (x_batch_test[0,:][inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
                            ol_pars.append(inf_par)
                            cnt += 1

                        # convert to RA
                        true_XS = convert_sky(true_XS,params['ref_geocent_time'],ol_pars)
                        true_x = convert_sky(np.reshape(true_x,[1,true_XS.shape[1]]),params['ref_geocent_time'],ol_pars).flatten()

                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])
                        cnt = 0
                        for bilby_idx in bilby_ol_idx:
                            bilby_par = params['bilby_pars'][bilby_idx]
                            true_post[:,cnt] = (other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
                            cnt += 1
                        if i == 0:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask])
                        else:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask],ax=ax)
                    ax = plot_sky(true_XS[:NskyPlotSamples,sky_mask],filled=True,trueloc=true_x[sky_mask],ax=ax)
                    plt.savefig('%s/latest_%s/sky_10toZero.png' % (params['plot_dir'],params['run_label']),dpi=360)
                    plt.close(fig)
                    exit()

    for JS_ele_idx,JS_ele in enumerate(JS_data):
        if JS_ele >= (1e-1-1e-2) and JS_ele <= (1e-1+1e-2) and not minus1_found:
            continue
            minus1_found = True
            minus1_idx = JS_ele_idx
            print('... Found JS for 10^-1: %.6f' % JS_ele)

            for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
                if int(step) == minus1_idx:
                    samples,_ = gen_samples(model, y_batch_test, ramp=ramp, nsamples=params['n_samples'])
                    # trim samples from outside the cube
                    mask = []
                    for s in samples:
                        if (np.all(s>=0.0) and np.all(s<=1.0)):
                            mask.append(True)
                        else:
                            mask.append(False)
                    samples = tf.boolean_mask(samples,mask,axis=0)
                    print('identified {} good samples'.format(samples.shape[0]))
                    if samples.shape[0]<100:
                        return [-1.0,-1.0,-1.0]

                    # randonly convert from reduced psi-phi space to full space
                    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
                    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
                    psi_idx = psi_idx[0]
                    phi_idx = phi_idx[0]
                    samples = samples.numpy()
                    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])
                    print()
                    print('... Generating sky plot')
                    print()
                    fig, ax = plt.subplots(figsize=(12,6))

                    sky_color_cycle=['blue','green','purple','orange']
                    sky_color_map_cycle=['Blues','Greens','Purples','Oranges']
                    for i, other_samples in enumerate(all_other_samples[:,step,:]):
                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])

                        true_x = np.zeros(inf_ol_len)
                        true_XS = np.zeros([samples.shape[0],inf_ol_len])
                        cnt= 0; ol_pars=[]
                        for inf_idx in inf_ol_idx:
                            inf_par = params['inf_pars'][inf_idx]
                            true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                            true_x[cnt] = (x_batch_test[0,:][inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
                            ol_pars.append(inf_par)
                            cnt += 1

                        # convert to RA
                        true_XS = convert_sky(true_XS,params['ref_geocent_time'],ol_pars)
                        true_x = convert_sky(np.reshape(true_x,[1,true_XS.shape[1]]),params['ref_geocent_time'],ol_pars).flatten()

                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])
                        cnt = 0
                        for bilby_idx in bilby_ol_idx:
                            bilby_par = params['bilby_pars'][bilby_idx]
                            true_post[:,cnt] = (other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
                            cnt += 1
                        if i == 0:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask])
                        else:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask],ax=ax)
                    ax = plot_sky(true_XS[:NskyPlotSamples,sky_mask],filled=True,trueloc=true_x[sky_mask],ax=ax)
                    plt.savefig('%s/latest_%s/sky_10toMinus1.png' % (params['plot_dir'],params['run_label']), dpi=360)
                    plt.close(fig)



    for JS_ele_idx,JS_ele in enumerate(JS_data):
        if JS_ele >= (1e-2-5e-2) and JS_ele <= (1e-2+5e-2) and not minus2_found:
            continue
            minus2_found = True
            minus2_idx = JS_ele_idx
            print('... Found JS for 10^-2: %.6f' % JS_ele)

            for step, (x_batch_test, y_batch_test) in test_dataset.enumerate():
                if int(step) == minus2_idx:
                    samples,_ = gen_samples(model, y_batch_test, ramp=ramp, nsamples=params['n_samples'])
                    # trim samples from outside the cube
                    mask = []
                    for s in samples:
                        if (np.all(s>=0.0) and np.all(s<=1.0)):
                            mask.append(True)
                        else:
                            mask.append(False)
                    samples = tf.boolean_mask(samples,mask,axis=0)
                    print('identified {} good samples'.format(samples.shape[0]))
                    if samples.shape[0]<100:
                        return [-1.0,-1.0,-1.0]

                    # randonly convert from reduced psi-phi space to full space
                    _, psi_idx, _ = get_param_index(params['inf_pars'],['psi'])
                    _, phi_idx, _ = get_param_index(params['inf_pars'],['phase'])
                    psi_idx = psi_idx[0]
                    phi_idx = phi_idx[0]
                    samples = samples.numpy()
                    samples[:,psi_idx], samples[:,phi_idx] = psiX_to_psiphi(samples[:,psi_idx], samples[:,phi_idx])
                    print()
                    print('... Generating sky plot')
                    print()
                    fig, ax = plt.subplots(figsize=(12,6))

                    sky_color_cycle=['blue','green','purple','orange']
                    sky_color_map_cycle=['Blues','Greens','Purples','Oranges']
                    for i, other_samples in enumerate(all_other_samples[:,step,:]):
                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])

                        true_x = np.zeros(inf_ol_len)
                        true_XS = np.zeros([samples.shape[0],inf_ol_len])
                        cnt= 0; ol_pars=[]
                        for inf_idx in inf_ol_idx:
                            inf_par = params['inf_pars'][inf_idx]
                            true_XS[:,cnt] = (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                            true_x[cnt] = (x_batch_test[0,:][inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par + '_min']
                            ol_pars.append(inf_par)
                            cnt += 1

                        # convert to RA
                        true_XS = convert_sky(true_XS,params['ref_geocent_time'],ol_pars)
                        true_x = convert_sky(np.reshape(true_x,[1,true_XS.shape[1]]),params['ref_geocent_time'],ol_pars).flatten()

                        true_post = np.zeros([other_samples.shape[0],bilby_ol_len])
                        cnt = 0
                        for bilby_idx in bilby_ol_idx:
                            bilby_par = params['bilby_pars'][bilby_idx]
                            true_post[:,cnt] = (other_samples[:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par + '_min']
                            cnt += 1
                        if i == 0:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask])
                        else:
                            ax = plot_sky(true_post[:NskyPlotSamples,sky_mask],filled=False,cmap=sky_color_map_cycle[i],col=sky_color_cycle[i],trueloc=true_x[sky_mask],ax=ax)
                    ax = plot_sky(true_XS[:NskyPlotSamples,sky_mask],filled=True,trueloc=true_x[sky_mask],ax=ax)
                    plt.savefig('%s/latest_%s/sky_10toMinus2.png' % (params['plot_dir'],params['run_label']), dpi=360)
                    plt.close(fig)

             
    
    print(samples.shape)
    print(zero_idx,minus1_idx,minus2_idx)
    exit()
    print('... Finished making sky JS plots!')

    return

def indiPar_JS_plots(model,sig_test,par_test,params,bounds,inf_ol_idx,bilby_ol_idx):
    """ Make individual parameter JS divergence plots

    """
    matplotlib.rc('text', usetex=True)
    grey_colors = ['#00FFFF','#FF00FF','#800000','#23F035']
    def extract_correct_sample_idx(samples, inf_ol_idx, params):
        true_XS = np.zeros([samples.shape[0],len(inf_ol_idx)])
        cnt_rm_inf = 0
        for inf_idx in inf_ol_idx:
            inf_par = params['inf_pars'][inf_idx]
            true_XS[:,cnt_rm_inf] = samples[:,inf_idx] # (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
            cnt_rm_inf += 1

        # Apply mask
        true_XS = true_XS.T
        sampset_1 = true_XS
        del_cnt = 0
        # iterate over each sample   during inference training
        for i in range(sampset_1.shape[1]):
            # iterate over each parameter
            for k,q in enumerate(params['bilby_pars']):
                # if sample out of range, delete the sample                                              the y data (size changes by factor of  n_filter/(2**n_redsteps) )
                if sampset_1[k,i] < 0.0 or sampset_1[k,i] > 1.0:
                    true_XS = np.delete(true_XS,del_cnt,axis=1)
                    del_cnt-=1
                    break
                # check m1 > m2
                elif q == 'mass_1' or q == 'mass_2':
                    m1_idx = np.argwhere(params['bilby_pars']=='mass_1')
                    m2_idx = np.argwhere(params['bilby_pars']=='mass_2')
                    if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                        true_XS = np.delete(true_XS,del_cnt,axis=1)
                        del_cnt-=1
                        break
        # transpose
        true_XS = np.zeros([samples.shape[0],len(inf_ol_idx)])
        for i in range(true_XS.shape[1]):
            true_XS[:,i] = sampset_1[i,:]
        return true_XS

    def compute_indi_kl(sampset_1,sampset_2,samplers,params,one_D=False):
        """
        Compute JS divergence for one test case.
        """

        # Remove samples outside of the prior mass distribution           
        cur_max = params['n_samples']
        
        # Iterate over parameters and remove samples outside of prior
        if samplers[0] == 'vitamin1' or samplers[1] == 'vitamin2':

            # Apply mask
            sampset_1 = sampset_1.T
            sampset_2 = sampset_2.T
            set1 = sampset_1
            set2 = sampset_2
            del_cnt_set1 = 0
            del_cnt_set2 = 0
            params_to_infer = params['inf_pars']
            for i in range(set1.shape[1]):

                # iterate over each parameter in first set
                for k,q in enumerate(params_to_infer):
                    # if sample out of range, delete the sample
                    if set1[k,i] < 0.0 or set1[k,i] > 1.0:
                        sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                        del_cnt_set1-=1
                        break
                    # check m1 > m2
                    elif q == 'mass_1' or q == 'mass_2':
                        m1_idx = np.argwhere(params_to_infer=='mass_1')
                        m2_idx = np.argwhere(params_to_infer=='mass_2')
                        if set1[m1_idx,i] < set1[m2_idx,i]:
                            sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                            del_cnt_set1-=1
                            break

                del_cnt_set1+=1

            # iterate over each sample
            for i in range(set2.shape[1]):

                # iterate over each parameter in second set
                for k,q in enumerate(params_to_infer):
                    # if sample out of range, delete the sample
                    if set2[k,i] < 0.0 or set2[k,i] > 1.0:
                        sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                        del_cnt_set2-=1
                        break
                    # check m1 > m2
                    elif q == 'mass_1' or q == 'mass_2':
                        m1_idx = np.argwhere(params_to_infer=='mass_1')
                        m2_idx = np.argwhere(params_to_infer=='mass_2')
                        if set2[m1_idx,i] < set2[m2_idx,i]:
                            sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                            del_cnt_set2-=1
                            break

                del_cnt_set2+=1

            del_final_idx = np.min([del_cnt_set1,del_cnt_set2])
            set1 = sampset_1[:,:del_final_idx]
            set2 = sampset_2[:,:del_final_idx]

        else:

            set1 = sampset_1.T
            set2 = sampset_2.T
  
        # Iterate over number of randomized sample slices
        SMALL_CONSTANT = 1e-162 # 1e-4 works best for some reason
        def my_kde_bandwidth(obj, fac=1.0):

            """We use Scott's Rule, multiplied by a constant factor."""

            return np.power(obj.n, -1./(obj.d+4)) * fac
            
        kl_result = []
        set1 = set1.T
        set2 = set2.T
        #for kl_idx in range(10):
        rand_idx_kl = np.linspace(0,params['n_samples']-1,num=params['n_samples'],dtype=np.int)
        np.random.shuffle(rand_idx_kl)
        rand_idx_kl = rand_idx_kl[:1000]
        rand_idx_kl_2 = np.linspace(0,params['n_samples']-1,num=params['n_samples'],dtype=np.int)
        np.random.shuffle(rand_idx_kl_2)
        rand_idx_kl_2 = rand_idx_kl_2[:1000]
        kl_result = np.zeros((len(params['bilby_pars'])))
        for indi_par_idx,_ in enumerate(params['bilby_pars']):
            kl_result[indi_par_idx] = estimate(np.expand_dims(set1[rand_idx_kl,indi_par_idx],axis=1),np.expand_dims(set2[rand_idx_kl_2,indi_par_idx],axis=1),k=10,n_jobs=10) + estimate(np.expand_dims(set2[rand_idx_kl_2,indi_par_idx],axis=1),np.expand_dims(set1[rand_idx_kl,indi_par_idx],axis=1),k=10,n_jobs=10)
        
        return kl_result

    # Define variables 
    usesamps = params['samplers']
    samplers = params['samplers']
    fig_samplers = params['figure_sampler_names']
    indi_fig_kl, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(6,6))  
    indi_axis_kl = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
  
    # Compute kl divergence on all test cases with preds vs. benchmark
    # Iterate over samplers
    tmp_idx=len(usesamps)
    print_cnt = 0
    runtime = {}
    CB_color_cycle = ['orange', 'purple', 'green',
              'blue', '#a65628', '#984ea3',
              '#e41a1c', '#dede00', 
              '#004d40','#d81b60','#1e88e5',
              '#ffc107','#1aff1a','#377eb8',
              '#fefe62','#d35fb7','#dc3220']
    label_idx = 0
    vi_pred_made = None

    # Get the xaxis labels for the scatter plot
    JS_par_labels=[]
    for par in params['bilby_pars']:
        JS_par_labels.append(params['corner_labels'][par])
    JS_par_labels = JS_par_labels*params['r']
    
    if params['load_plot_data'] == False:
        try:
            hf = h5py.File('plotting_data_%s/JS_indiPar_plot_data.h5' % params['run_label'], 'w')
        except:
            os.remove('plotting_data_%s/JS_indiPar_plot_data.h5' % params['run_label'])
            hf = h5py.File('plotting_data_%s/JS_indiPar_plot_data.h5' % params['run_label'], 'w')
    else:
        hf = h5py.File('plotting_data_%s/JS_indiPar_plot_data.h5' % params['run_label'], 'r')
   
    # Individual plots (not subplots)
    fig_dict = {}
    axis_dict = {}
    for k in range(len(usesamps)-1):
        fig_dict[usesamps[1:][k]], axis_dict[usesamps[1:][k]] = plt.subplots(figsize=(8,6))

    # This will go through values of 0 to 3 in steps of 1
    for k in range(len(usesamps)-1):

        print_cnt = 0
        label_idx = 0
        tmp_idx = len(usesamps)
        if k <= 1:
           kl_idx_1 = 0
           kl_idx_2 = k
        elif k > 1:
           kl_idx_1 = 1
           kl_idx_2 = (k-2)

        tot_kl_grey = np.array([])
        # run through values from 0 to 4 (total of 5 elements)
        #grey_cnt=1
        for i in range(len(usesamps)):
            for j in range(tmp_idx):

                sampler1, sampler2 = samplers[i], samplers[::-1][j]

                # Check if sampler results already exists in hf directories
                h5py_node1 = '%s-%s' % (sampler1,sampler2)
                h5py_node2 = '%s-%s' % (sampler2,sampler1)
                node_exists = False
                if h5py_node1 in hf.keys() or h5py_node2 in hf.keys():
                    node_exists=True

                # Get KL results
                if samplers[i] == samplers[::-1][j]: # Skip samplers with themselves
                    print_cnt+=1
                    continue
                elif params['load_plot_data'] == True or node_exists: # load pre-generated JS results
                    tot_kl = np.array(hf['%s-%s' % (sampler1,sampler2)])
                else:
                    if params['load_plot_data'] == False: # Make new JS results if needed
                        if sampler1 != 'vitamin':
                            set1 = load_samples(params, sampler1, pp_plot=True)
                        elif sampler1 == 'vitamin' and vi_pred_made == None:
                            set1 = np.zeros((params['r'], params['n_samples'], len(params['bilby_pars'])))
                            for sig_test_idx in range(params['r']):
                                samples,_ = np.array(gen_samples(model, np.expand_dims(sig_test[sig_test_idx], axis=0), ramp=1, nsamples=params['n_samples'])) 
                                set1[sig_test_idx,:,:] = extract_correct_sample_idx(samples, inf_ol_idx, params)
                            vi_pred_made = [set1]
                        if sampler2 != 'vitamin':
                            set2 = load_samples(params, sampler2, pp_plot=True)
                        elif sampler1 == 'vitamin' and vi_pred_made == None:
                            set2 = np.zeros((params['r'], params['n_samples'], len(params['bilby_pars'])))
                            for sig_test_idx in range(params['r']):
                                samples,_ = np.array(gen_samples(model, np.expand_dims(sig_test[sig_test_idx], axis=0), ramp=1, nsamples=params['n_samples']))
                                set2[sig_test_idx,:,:] = extract_correct_sample_idx(samples, inf_ol_idx, params)
                            vi_pred_made = [set2]

                    # Iterate over test cases
                    tot_kl = np.zeros((params['r'],len(params['bilby_pars'])))  # total JS over all infered parameters
                    #tot_kl = []
                    for r in range(params['r']):
                        # Iterate over indi source parameters
                        tot_kl[r,:] = compute_indi_kl(set1[r,:],set2[r,:],[sampler1,sampler2],params)
                        print()
                        print('... Completed JS for set %s-%s and test sample %s with mean JS: %.6f' % (sampler1,sampler2,str(r), np.mean(tot_kl)))
                        print()

                # Try saving both sampler permutations of the JS results
                try:
                    if params['load_plot_data'] == False:
                        # Save results to h5py file
                        hf.create_dataset('%s-%s' % (sampler1,sampler2), data=tot_kl)
                except Exception as e:
                    pass
                try:
                    if params['load_plot_data'] == False:
                        # Save results to h5py file
                        hf.create_dataset('%s-%s' % (sampler2,sampler1), data=tot_kl)
                except Exception as e:
                    pass

                # plot colored hist
                if samplers[i] == 'vitamin' and samplers[::-1][j] == samplers[1:][k]:
                    cur_bayesian_sampler = fig_samplers[::-1][j]
                    
                    axis_dict[usesamps[1:][k]].scatter(JS_par_labels,tot_kl.flatten(),s=1,c=CB_color_cycle[print_cnt],label=r'$\mathrm{%s \ vs. \ %s}$' % (fig_samplers[::-1][j],fig_samplers[i]))
                # record non-colored hists
                elif samplers[i] != 'vitamin' and samplers[::-1][j] != 'vitamin':
                    if samplers[i] == samplers[1:][k] or samplers[::-1][j] == samplers[1:][k]:
                        if cur_bayesian_sampler == fig_samplers[i]:
                            second_bayesian_sampler = fig_samplers[::-1][j]
                            grey_cnt = fig_samplers.index(fig_samplers[::-1][j]) - 1
                        else:
                            second_bayesian_sampler = fig_samplers[i]
                            grey_cnt = fig_samplers.index(fig_samplers[i]) - 1 
                        axis_dict[usesamps[1:][k]].scatter(JS_par_labels,tot_kl.flatten(),s=1,c=grey_colors[grey_cnt],label=r'$\mathrm{%s \ vs. \ %s}$' % (cur_bayesian_sampler,second_bayesian_sampler))
                        print()
                        print('... Grey Mean total JS between %s-%s: %s' % ( sampler1, sampler2, str(np.mean(tot_kl)) ) )
                print()
                print_cnt+=1
            tmp_idx-=1
        # plot JS histograms
        axis_dict[usesamps[1:][k]].set_xlabel(r'$\mathrm{Source \ Parameters}$',fontsize=14)
        axis_dict[usesamps[1:][k]].set_ylabel(r'$p(\mathrm{JS})$',fontsize=14)
        #axis_dict[usesamps[1:][k]].set_xticks(rotation=45)
        #plt.xticks(rotation=45)
        axis_dict[usesamps[1:][k]].set_xticklabels(JS_par_labels[:len(params['bilby_pars'])], rotation=45, ha='right')
        leg = axis_dict[usesamps[1:][k]].legend(loc='upper right',  fontsize=6) #'medium')
        for l in leg.legendHandles:
            l.set_alpha(1.0)

        # Make horizontal line for Gregory criterion
#        axis_dict[usesamps[1:][k]].plot(JS_par_labels[:len(params['bilby_pars'])],[0.002]*len(params['bilby_pars']), color='red', ls='dashed')
        axis_dict[usesamps[1:][k]].axhline(y=0.002, color='red', ls='dashed')
        print()
        print('... Made JS IndiPar plot %d' % k)
        print()

        # Save figure
        fig_dict[usesamps[1:][k]].canvas.draw()
        plt.tight_layout()
        fig_dict[usesamps[1:][k]].savefig('%s/latest_%s/JS_IndiPar_%s.png' % (params['plot_dir'],params['run_label'],usesamps[1:][k]),dpi=360)
        plt.close(fig_dict[usesamps[1:][k]])
        print()
        print('... Saved JS IndiPar plot to -> %s/latest_%s/JS_IndiPar_%s.png' % (params['plot_dir'],params['run_label'],usesamps[1:][k]))
        print()
                
    hf.close()
    return

def JS_vs_SNR_plot(JS_data, snrs_test, params):
    """ Plot the JS as a function of SNR for 
    all test signals.

    """

    # Get dynesty vs. Vitamin results
    dyn_vs_vit = np.array(JS_data['dynesty-vitamin'])
    
    # Plot JS vs. SNR
    matplotlib.rc('text', usetex=True)
    for idx, det in enumerate(params['det']):
        plt.scatter(snrs_test[:,idx],dyn_vs_vit, s=6, linewidth=0.5, marker="+", label=r'$\textrm{%s}$' % det)
    plt.xlabel(r'$\textrm{SNR}$')
    plt.ylabel(r'$\textrm{JS Divergence}$')
    plt.legend()
    plt.savefig('%s/latest_%s/JS_vs_SNR.png' % (params['plot_dir'], params['run_label']), dpi=360)
    plt.close()

    print('... Made JS vs. SNR plot!')
    return

def prune_samples(chain_file_loc,params):
    """ Function to remove bad likelihood emcee chains 
   
    Parameters
    ----------
    chain_file_loc: str
        location of emcee chain file
    params: dict
        general run parameters

    Returns
    -------
    XS: array_like
        pruned emcee samples
    """
    nsteps = 14000
    nburnin = 4000
    nwalkers = 250
    thresh_num = 30
    ndim=len(params['inf_pars'])
    chain_file = h5py.File(chain_file_loc, 'r')
    exit_flag=False

    while exit_flag == False:
        # Iterate over all parameters in chain file
        XS = np.array([])
        for idx in range(ndim):
            chains_before = np.array(chain_file[params['inf_pars'][idx]+'_post']).reshape((nsteps-nburnin,nwalkers))
            logL = np.array(chain_file['log_like_eval']).reshape((nsteps-nburnin,nwalkers))
            logL_max = np.max(logL)

            XS = np.append(XS,np.expand_dims(chains_before,0))

        # data starts as (nsteps*nwalkers) x ndim -> 2D
        XS = XS.transpose()                                     # now ndim x (nsteps*nwalkers) -> 2D
        XS = XS.reshape(ndim,nwalkers,nsteps-nburnin)                      # now ndim x nwalkers x nsteps -> 3D
        XSex = XS[:,0,:].squeeze().transpose()        # take one walker nsteps x ndim -> 2D
        XS = XS.transpose((2,1,0))                          # now nsteps x nwalkers x ndim -> 3D

        # identify good walkers
        # logL starts off with shape (nsteps*nwalkers) -> 1D
        thresh = logL_max - thresh_num                                # define log likelihood threshold
        idx_walkers = np.argwhere([np.all(logL[:,i]>thresh) for i in range(nwalkers)])       # get the indices of good chains
        Nsamp = len(idx_walkers)*(nsteps-nburnin)                                 # redefine total number of good samples 

        # select good walkers
        XS = np.array([XS[:,i,:] for i in idx_walkers]).squeeze()     # just pick out good walkers

        XS = XS.reshape(-1,ndim)                                    # now back to original shape (but different order) (walkers*nstep) x 
        try:
            #idx = np.random.choice(Nsamp,10000)          # choose 10000 random indices for corner plots
            if Nsamp < params['n_samples']:
                raise Exception
            else:
                rand_idx_posterior = np.linspace(0,Nsamp-1,num=params['n_samples'],dtype=np.int)
                np.random.shuffle(rand_idx_posterior)
                idx = rand_idx_posterior[:params['n_samples']]
        except Exception:
            thresh_num+=10
            print('... increasing threshold value to %d' % thresh_num)
        else:
            exit_flag = True

    # pick out random samples from clean set
    XS = XS[idx,:]                                                  # select 10000 random samples

    return XS

def make_dirs(params,out_dir):
    """ Make directories to store plots. Directories that already exist will be overwritten.
    
    Parameters
    ----------
    params: dict
        general run parameters
    out_dir: str
        output directory for test plots
    """

    ## If file exists, delete it ##
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    else:    ## Show a message ##
        print("Attention: %s directory not found, making new directory" % out_dir)

    ## If file exists, delete it ##
    if os.path.exists('plotting_data_%s' % params['run_label']):
        print('plotting data directory already exits.')
    else:    ## Show a message ##
        print("Attention: plotting_data_%s directory not found, making new directory ..." % params['run_label'])
        # setup output directory - if it does not exist
        os.makedirs('plotting_data_%s' % params['run_label'])
        print('Created directory: plotting_data_%s' % params['run_label'])

    os.makedirs('%s' % out_dir)
    os.makedirs('%s/latest' % out_dir)
    os.makedirs('%s/animations' % out_dir)
   
    print('Created directory: %s' % out_dir)
    print('Created directory: %s' % (out_dir+'/latest'))
    print('Created directory: %s' % (out_dir+'/animations'))

    return

def factorial(n):
   if n == 1: #base case
       return n
   else:
       return n*factorial(n-1) # recursive case

class make_plots:
    """ Class to generate a suite of testing plots
    """
    
    def __init__(self,params,samples,rev_x,pos_test):
        """ Add variables here later if need be
        """
        self.params = params       # general run parameters
        self.samples = samples     # Bayesian posterior samples
        self.rev_x = rev_x         # pre-generated NN posteriors
        self.pos_test = pos_test   # scalar truths of test samples

        def load_test_set(model,sig_test,par_test,y_normscale,bounds,sampler='dynesty1',vitamin_pred_made=None):
            """ load requested test set

            Parameters
            ----------
            model: tensorflow object
                pre-trained tensorflow neural network
            sig_test: array_like
                test signal time series
            par_test: array_like
                test signal source parameter values
            y_normscale: float
                arbitrary normalization factor to scale time series
            bounds: dict
                bounds allowed for GW waveforms
            sampler: str
                sampler results to load
            vitamin_pred_made: bool
                if True, use pre-generated vitamin predictions

            Returns
            -------
            VI_pred_all: array_like
                predictions made by the vitamin box
            XS_all: array_like
                predictions made by the Bayesian samplers
            dt: array_like
                time to compute posterior predictions 
            """

            if sampler=='vitamin1' or sampler=='vitamin2':

                # check if vitamin test posteriors have already been generated
                if vitamin_pred_made != None:
                    return vitamin_pred_made[0], vitamin_pred_made[1]

                VI_pred_all = []
                for i in range(params['r']):
                    # The trained inverse model weights can then be used to infer a probability density of solutions given new measurements
                    VI_pred,dt, _,_,_,_,_,_  = model.run(params, par_test[i], np.expand_dims(sig_test[i],axis=0),
                                              "inverse_model_dir_%s/inverse_model.ckpt" % params['run_label'],
                                              wrmp=np.ones(params['n_modes']))

                    """
                    # convert RA to hour angle for test set validation cost if both ra and geo time present
                    if np.isin('ra', params['inf_pars']) and  np.isin('geocent_time', params['inf_pars']):
                        # get geocenttime index
                        for k_idx,k in enumerate(params['inf_pars']):
                            if k == 'geocent_time':
                                geo_idx = k_idx
                            elif k == 'ra':
                                ra_idx = k_idx

                        # unnormalize and get gps time
                        VI_pred[:,ra_idx] = (VI_pred[:,ra_idx] * (bounds['ra_max'] - bounds['ra_min'])) + bounds['ra_min']   

                        gps_time_arr = (VI_pred[:,geo_idx] * (bounds['geocent_time_max'] - bounds['geocent_time_min'])) + bounds['geocent_time_min']
                        # convert to RA
                        # Iterate over all training samples and convert to hour angle
                        for k in range(VI_pred.shape[0]):
#                            VI_pred[k,ra_idx]=np.mod(GreenwichMeanSiderealTime(float(params['ref_geocent_time']+gps_time_arr[k]))-VI_pred[k,ra_idx], 2.0*np.pi)
                            VI_pred[k,ra_idx]=np.mod(GreenwichMeanSiderealTime(params['ref_geocent_time'])-VI_pred[k,ra_idx], 2.0*np.pi)
                        # normalize
                        VI_pred[:,ra_idx]=(VI_pred[:,ra_idx] - bounds['ra_min']) / (bounds['ra_max'] - bounds['ra_min'])
                    """                    

                    VI_pred_all.append(VI_pred)

                    print('Generated vitamin preds %d/%d' % (int(i),int(params['r'])))

                VI_pred_all = np.array(VI_pred_all)

                return VI_pred_all, dt


            # load up the posterior samples (if they exist)
            # load generated samples back in
            post_files = []

            # choose directory with lowest number of total finished posteriors
            num_finished_post = int(1e8)
            for i in self.params['samplers']:
                if i == 'vitamin':
                    continue
                for j in range(1):
                    input_dir = '%s_%s%d/' % (self.params['pe_dir'],i,j+1)
                    if type("%s" % input_dir) is str:
                        dataLocations = ["%s" % input_dir]

                    filenames = sorted(os.listdir(dataLocations[0]), key=lambda x: int(x.split('.')[0].split('_')[-1]))      
                    if len(filenames) < num_finished_post:
                        sampler_loc = i + str(j+1)
                        num_finished_post = len(filenames)

            dataLocations_try = '%s_%s' % (self.params['pe_dir'],sampler_loc)
            
            dataLocations = '%s_%s' % (self.params['pe_dir'],sampler)

            #for i,filename in enumerate(glob.glob(dataLocations[0])):
            i_idx = 0
            i = 0
            i_idx_use = []
            dt = []
            while i_idx < self.params['r']:

#                filename_try = '%s/%s_%d.h5py' % (dataLocations_try,self.params['bilby_results_label'],i)
                filename = '%s/%s_%d.h5py' % (dataLocations,self.params['bilby_results_label'],i)

                for samp_idx_inner in params['samplers'][1:]:
                    inner_file_existance = True
                    if samp_idx_inner == sampler+'1':
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
#                    print('File does not exist for one of the samplers')
                    continue

                # If file does not exist, skip to next file
#                try:
#                    h5py.File(filename_try, 'r')
#                except Exception as e:
#                    i+=1
#                    continue

                print(filename)
                dt.append(np.array(h5py.File(filename, 'r')['runtime']))

                post_files.append(filename)
                if sampler == 'emcee1':
                    emcee_pruned_samples = prune_samples(filename,self.params)
                data_temp = {}
                n = 0
                for q_idx,q in enumerate(self.params['inf_pars']):
                     p = q + '_post'
                     par_min = q + '_min'
                     par_max = q + '_max'
                     if p == 'psi_post':
                         data_temp[p] = np.remainder(data_temp[p],np.pi)
                     if sampler == 'emcee1':
                         data_temp[p] = emcee_pruned_samples[:,q_idx]
                     else:
                         data_temp[p] = h5py.File(filename, 'r')[p][:]
                     if p == 'geocent_time_post' or p == 'geocent_time_post_with_cut':
                         data_temp[p] = data_temp[p] - self.params['ref_geocent_time']
                     data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
                     Nsamp = data_temp[p].shape[0]
                     n = n + 1

                XS = np.zeros((Nsamp,n))
                j = 0
                for p,d in data_temp.items():
                    XS[:,j] = d
                    j += 1

                rand_idx_posterior = np.linspace(0,XS.shape[0]-1,num=self.params['n_samples'],dtype=np.int)
                np.random.shuffle(rand_idx_posterior)
                rand_idx_posterior = rand_idx_posterior[:self.params['n_samples']]
                if i_idx == 0:
                    XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
                    #XS_all = np.expand_dims(XS[:self.params['n_samples'],:], axis=0)
                else:
                    # save all posteriors in array
                    max_allow_idx = np.min([XS_all.shape[1],np.expand_dims(XS[rand_idx_posterior,:], axis=0).shape[1]])
                    XS_all = np.vstack((XS_all[:,:max_allow_idx,:],np.expand_dims(XS[rand_idx_posterior,:], axis=0)[:,:max_allow_idx,:]))
                    #XS_all = np.vstack((XS_all[:,:max_allow_idx,:],np.expand_dims(XS[:self.params['n_samples'],:], axis=0)[:,:max_allow_idx,:]))

                i_idx_use.append(i)
                i+=1
                i_idx+=1

            # save time per sample
            dt = np.array(dt)
            dt = np.array([np.min(dt),np.max(dt),np.median(dt)])

            return XS_all, dt

        def confidence_bd(samp_array):
            """ compute confidence bounds for a given array
            
            Parameters
            ----------
            samp_array: array_like
                posterior samples
 
            Returns
            -------
            list:
                lower and upper bounds of the confidence interval
            """
            cf_bd_sum_lidx = 0
            cf_bd_sum_ridx = 0
            cf_bd_sum_left = 0
            cf_bd_sum_right = 0
            cf_perc = 0.05

            cf_bd_sum_lidx = np.sort(samp_array)[int(len(samp_array)*cf_perc)]
            cf_bd_sum_ridx = np.sort(samp_array)[int(len(samp_array)*(1.0-cf_perc))]

            return [cf_bd_sum_lidx, cf_bd_sum_ridx]

        # Store above declared functions to be used later
        self.load_test_set = load_test_set
        self.confidence_bd = confidence_bd

    def pp_plot(self,truth,samples):
        """ generates the pp plot data given samples and truth values
        
        Parameters
        ----------
        truth: float
            true scalar value of source parameter
        samples: array_like
            posterior samples of source parameter

        Returns
        -------
        r: float
            pp value
        """

        Nsamp = samples.shape[0]

        r = np.sum(samples>truth)/float(Nsamp)

        return r

    def plot_pp(self, model, sig_test, par_test, params, bounds, inf_ol_idx, bilby_ol_idx):
        """ make p-p plots using in-house methods
        
        Parameters
        ----------
        model: tensorflow object
            pre-trained tensorflow neural network model
        sig_test: array_like
            array of test time series
        par_test: array_like
            array of test time series source parameters
            

        """
        normscales=params['y_normscale']

        matplotlib.rc('text', usetex=True)
        Npp = int(self.params['r']) # number of test GW waveforms to use to calculate PP plot
        ndim_y = self.params['ndata']
        conf_color_wheel = ['#D8D8D8','#A4A4A4','#6E6E6E']
        confidence = [0.9,0.5]
        x_values = np.linspace(0, 1, 1001)
        N = int(self.params['r'])
       
 
        # publication plot
        fig, axis = plt.subplots(1,1,figsize=(6,6))

        # diagnostic indi par plot
        fig_indi, axis_indi = plt.subplots(1,1,figsize=(6,6))
        NUM_COLORS = len(self.params['bilby_pars'])
        new_colors = [plt.get_cmap('nipy_spectral')(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

        if self.params['load_plot_data'] == True:
            # Create dataset to save PP results for later plotting
            hf = h5py.File('plotting_data_%s/pp_plot_data.h5' % self.params['run_label'], 'r')
        else:
            # Create dataset to save PP results for later plotting
            try:
                os.remove('plotting_data_%s/pp_plot_data.h5' % self.params['run_label'])
            except:
                pass
            hf = h5py.File('%s/pp_plot_data.h5' % self.params['plot_dir'], 'w')

        if self.params['load_plot_data'] == False:
            pp = np.zeros(((self.params['r'])+2,len(self.params['bilby_pars']))) 
            for cnt in range(Npp):

                # generate Vitamin samples
                y = np.expand_dims(sig_test[cnt,:], axis=0)
                 # The trained inverse model weights can then be used to infer a probability density of solutions 
#given new measurements
                samples,_ = np.array(gen_samples(model, y, ramp=1, nsamples=params['n_samples']))

                true_XS = np.zeros([samples.shape[0],len(inf_ol_idx)])
                true_x = np.zeros([len(inf_ol_idx)])
                ol_pars = []
                cnt_rm_inf = 0
                for inf_idx in inf_ol_idx:
                    inf_par = params['inf_pars'][inf_idx]
                    true_XS[:,cnt_rm_inf] = samples[:,inf_idx] # (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                    true_x[cnt_rm_inf] = par_test[cnt,inf_idx]
                    ol_pars.append(inf_par)
                    cnt_rm_inf += 1

                # Apply mask
                x = true_XS
                x = x.T
                sampset_1 = x   
                del_cnt = 0
                # iterate over each sample   during inference training
                for i in range(sampset_1.shape[1]):
                    # iterate over each parameter
                    for k,q in enumerate(self.params['bilby_pars']):
                        # if sample out of range, delete the sample                                              the y data (size changes by factor of  n_filter/(2**n_redsteps) )
                        if sampset_1[k,i] < 0.0 or sampset_1[k,i] > 1.0:
                            x = np.delete(x,del_cnt,axis=1)   
                            del_cnt-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(self.params['bilby_pars']=='mass_1')
                            m2_idx = np.argwhere(self.params['bilby_pars']=='mass_2')
                            if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                                x = np.delete(x,del_cnt,axis=1)
                                del_cnt-=1
                                break    
                    del_cnt+=1
                for j in range(len(self.params['bilby_pars'])):
                    pp[0,j] = 0.0
                    pp[1,j] = 1.0
                    pp[cnt+2,j] = self.pp_plot(true_x[j],x[j,:])
                    print()
                    print('... Computed param %d p-p plot iteration %d/%d' % (j,int(cnt)+1,int(Npp)))
                    print()

            # Save VItamin pp curves
            hf.create_dataset('vitamin_pp_data', data=pp)

        else:
            pp = hf['vitamin_pp_data']
            print()
            print('... Loaded VItamin pp curves')
            print()
        confidence_pp = np.zeros((len(self.params['samplers'])-1,int(self.params['r'])+2))
        # plot the pp plot
        for j in range(len(self.params['bilby_pars'])):        
            if j == 0:
                axis.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp[:,j]),'-',color='red',linewidth=1,zorder=50,label=r'$\textrm{%s}$' % self.params['figure_sampler_names'][0],alpha=0.5)
                axis_indi.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp[:,j]),'-',linewidth=1,
                          zorder=50,alpha=1.0,label=repr('${%s}$') % self.params['bilby_pars'][j],color=new_colors[j])
            elif j == 10:
                print(np.sort(pp[:,j]))
            else:
                axis.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp[:,j]),'-',color='red',linewidth=1,zorder=50,alpha=0.5)
                axis_indi.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp[:,j]),'-',linewidth=1,
                          zorder=50,alpha=1.0,label=repr('${%s}$') % self.params['bilby_pars'][j],color=new_colors[j])
        
        # Credibility intervals for indi plot       
        matplotlib.rc('text', usetex=True)
        axis_indi.margins(x=0,y=0)
        axis_indi.plot([0,1],[0,1],'--k')
        for ci,j in zip(confidence,range(len(confidence))):
            edge_of_bound = (1. - ci) / 2.
            lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
            upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
            lower[0] = 0
            upper[0] = 0
            axis_indi.fill_between(x_values, lower, upper, facecolor=conf_color_wheel[j],alpha=0.5)        

        axis_indi.set_xlim([0,1])
        axis_indi.set_ylim([0,1])
        axis_indi.set_ylabel(r'$\textrm{Fraction of events within the Credible Interval}$',fontsize=14)
        axis_indi.set_xlabel(r'$\textrm{Probability within the Credible Interval}$',fontsize=14)
        axis_indi.tick_params(axis="x", labelsize=14)
        axis_indi.tick_params(axis="y", labelsize=14)
        leg = axis_indi.legend(loc='lower right', fontsize=7)
        for l in leg.legendHandles:
            l.set_alpha(1.0)
        plt.tight_layout()
        fig_indi.savefig('%s/latest_%s/vitmain_indi_pp_plot.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        print()
        print('... Saved pp plot to -> %s/latest_%s/vitamin_indi_pp_plot.png' % (self.params['plot_dir'],self.params['run_label']))
        print()
        plt.close(fig_indi); del axis_indi
        print('Made individual par pp plot!')
        
        # make bilby p-p plots
        samplers = self.params['samplers']
        CB_color_cycle=['blue','green','purple','orange']

        for i in range(len(self.params['samplers'])):
            if samplers[i] == 'vitamin': continue

            # diagnostic indi par plot
            fig_indi, axis_indi = plt.subplots(1,1,figsize=(6,6))

            if self.params['load_plot_data'] == False:
                # load bilby sampler samples
                #samples,time = self.load_test_set(model,sig_test,par_test,normscales,bounds,sampler=samplers[i]+'1')
                samples = load_samples(self.params, samplers[i], pp_plot=True)
                true_XS = np.zeros([samples.shape[0],samples.shape[1],len(bilby_ol_idx)])
                true_x = np.zeros([samples.shape[0],len(bilby_ol_idx)])
                ol_pars = []; cnt_rm_inf = 0
                for inf_idx,bilby_idx in zip(inf_ol_idx,bilby_ol_idx):
                    bilby_par = params['bilby_pars'][bilby_idx]
                    inf_par = params['inf_pars'][inf_idx]
                    true_XS[:,:,cnt_rm_inf] = samples[:,:,bilby_idx] #(samples[:,:,bilby_idx] * (bounds[bilby_par+'_max'] - bounds[bilby_par+'_min'])) + bounds[bilby_par+'_min']
                    true_x[:,cnt_rm_inf] = par_test[:,inf_idx]
                    ol_pars.append(bilby_par)
                    cnt_rm_inf += 1
                samples = true_XS

                if samples.shape[0] == self.params['r']:
                    samples = samples[:,:,-self.params['n_samples']:]
                else:
                    samples = samples[:self.params['n_samples'],:]

            for j in range(len(self.params['bilby_pars'])):
                pp_bilby = np.zeros((self.params['r'])+2)
                pp_bilby[0] = 0.0
                pp_bilby[1] = 1.0
                if self.params['load_plot_data'] == False:
                    for cnt in range(self.params['r']):
                        pp_bilby[cnt+2] = self.pp_plot(true_x[cnt, j], samples[cnt,:,j])
                        print()
                        print('... Computed %s, param %d p-p plot iteration %d/%d' % (samplers[i],j,int(cnt)+1,int(self.params['r'])))
                        print()
                    hf.create_dataset('%s_param%d_pp' % (samplers[i],j), data=pp_bilby)           
                else:
                    pp_bilby = hf['%s_param%d_pp' % (samplers[i],j)]
                    print()
                    print('... Loaded Bilby sampler pp curve')
                    print()
                # plot bilby sampler results
                if j == 0:
                    axis.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp_bilby),'-',color=CB_color_cycle[i-1],linewidth=1,label=r'$\textrm{%s}$' % self.params['figure_sampler_names'][i],alpha=0.5)
                    axis_indi.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp_bilby),'-',
                                              color=new_colors[j],linewidth=1,label=repr('${%s}$') % self.params['bilby_pars'][j],alpha=1.0)
                else:
                    axis.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp_bilby),'-',color=CB_color_cycle[i-1],linewidth=1,alpha=0.5)
                    axis_indi.plot(np.arange((self.params['r'])+2)/((self.params['r'])+1.0),np.sort(pp_bilby),'-',
                                   label=repr('${%s}$') % self.params['bilby_pars'][j],color=new_colors[j],linewidth=1,alpha=1.0)


            # Bilby individual pp plots       
            matplotlib.rc('text', usetex=True)
            axis_indi.margins(x=0,y=0)
            axis_indi.plot([0,1],[0,1],'--k')
            for ci,j in zip(confidence,range(len(confidence))):
                edge_of_bound = (1. - ci) / 2.
                lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
                upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
                lower[0] = 0
                upper[0] = 0
                axis_indi.fill_between(x_values, lower, upper, facecolor=conf_color_wheel[j],alpha=0.5)
            axis_indi.set_xlim([0,1])
            axis_indi.set_ylim([0,1])
            axis_indi.set_ylabel(r'$\textrm{Fraction of events within the Credible Interval}$',fontsize=14)
            axis_indi.set_xlabel(r'$\textrm{Probability within the Credible Interval}$',fontsize=14)
            axis_indi.tick_params(axis="x", labelsize=14)
            axis_indi.tick_params(axis="y", labelsize=14)
            leg = axis_indi.legend(loc='lower right', fontsize=7)
            for l in leg.legendHandles:
                l.set_alpha(1.0)
            plt.tight_layout()
            fig_indi.savefig('%s/latest_%s/%s_indi_pp_plot.png' % (self.params['plot_dir'],self.params['run_label'],self.params['samplers'][i]),dpi=360)
            print()
            print('... Saved pp plot to -> %s/latest_%s/%s_indi_pp_plot.png' % (self.params['plot_dir'],self.params['run_label'],self.params['samplers'][i]))
            print()
            plt.close(fig_indi); del axis_indi
            print('Made individual par pp plot!')

            confidence_pp[i-1,:] = np.sort(pp_bilby)

        matplotlib.rc('text', usetex=True) 
        # Remove whitespace on x-axis in all plots
        axis.margins(x=0,y=0)
        axis.plot([0,1],[0,1],'--k')

        
        # Add credibility interals
        for ci,j in zip(confidence,range(len(confidence))):
            edge_of_bound = (1. - ci) / 2.
            lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
            upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
            # The binomial point percent function doesn't always return 0 @ 0,
            # so set those bounds explicitly to be sure
            lower[0] = 0
            upper[0] = 0
            axis.fill_between(x_values, lower, upper, facecolor=conf_color_wheel[j],alpha=0.5)
        
        axis.set_xlim([0,1])
        axis.set_ylim([0,1])
        axis.set_ylabel(r'$\textrm{Fraction of events within the Credible Interval}$',fontsize=14)
        axis.set_xlabel(r'$\textrm{Probability within the Credible Interval}$',fontsize=14)
        axis.tick_params(axis="x", labelsize=14)
        axis.tick_params(axis="y", labelsize=14)
        leg = axis.legend(loc='lower right', fontsize=14)
        for l in leg.legendHandles:
            l.set_alpha(1.0)
        plt.tight_layout()
        fig.savefig('%s/latest_%s/latest_pp_plot.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        print()
        print('... Saved pp plot to -> %s/latest_%s/latest_pp_plot.png' % (self.params['plot_dir'],self.params['run_label']))
        print()
        plt.close(fig)
        hf.close()
        return

    def plot_loss(self):
        """ Regenerate previously made loss plot
        """
        matplotlib.rc('text', usetex=True)

        # Load old plot data
        plotdata = np.loadtxt("inverse_model_dir_%s/loss_data.txt" % self.params['run_label'])

        # Make loss plot
        plt.figure()
        xvec = self.params['report_interval']*np.arange(np.array(plotdata).shape[0])
        plt.semilogx(xvec,np.array(plotdata)[:,0],label=r'$\mathrm{Recon}(L)$',color='blue',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,1],label=r'$\mathrm{KL}$',color='orange',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,2],label=r'$\mathrm{Total}(H)$',color='green',alpha=0.5)
        plt.semilogx(xvec,np.array(plotdata)[:,3],color='blue',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,4],color='orange',linestyle='dotted')
        plt.semilogx(xvec,np.array(plotdata)[:,5],color='green',linestyle='dotted')
        plt.xlim([3e3,np.max(xvec)])
        plt.ylim([-25,15])
        plt.xlabel(r'$\mathrm{Iteration}$')
        plt.ylabel(r'$\mathrm{Cost}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/latest_%s/cost_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']),dpi=360)
        print()
        print('... Saved cost unzoomed plot to -> %s/latest_%s/cost_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']))
        plt.ylim([np.min(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,0]), np.max(np.array(plotdata)[-int(0.9*np.array(plotdata).shape[0]):,1])])
        plt.savefig('%s/latest_%s/cost_zoom_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']),dpi=360)
        print('... Saved cost zoomed plot to -> %s/latest_%s/cost_zoom_%s.png' % (self.params['plot_dir'],self.params['run_label'],self.params['run_label']))
        print()
        plt.close('all')

        return

    def gen_kl_plots(self,model,sig_test,par_test,params,bounds,inf_ol_idx,bilby_ol_idx):
        """  Make kl corner histogram plots.
        
        Parameters
        ----------
        model: tensorflow object
            pre-trained tensorflow model
        sig_test: array_like
            test sample time series
        par_test: array_like
            test sample source parameter values
        normscales: float
            arbitrary normalization factor for time series
        bounds: dict
            allowed bounds for GW waveform source parameters
        snrs_test: array_like
            Optimal SNR values for every test sample
        """
        matplotlib.rc('text', usetex=True)

        def extract_correct_sample_idx(samples, inf_ol_idx):
            true_XS = np.zeros([samples.shape[0],len(inf_ol_idx)])
            cnt_rm_inf = 0
            for inf_idx in inf_ol_idx:
                inf_par = params['inf_pars'][inf_idx]
                true_XS[:,cnt_rm_inf] = samples[:,inf_idx] # (samples[:,inf_idx] * (bounds[inf_par+'_max'] - bounds[inf_par+'_min'])) + bounds[inf_par+'_min']
                cnt_rm_inf += 1

            # Apply mask
            true_XS = true_XS.T
            sampset_1 = true_XS
            del_cnt = 0
            # iterate over each sample   during inference training
            for i in range(sampset_1.shape[1]):
                # iterate over each parameter
                for k,q in enumerate(self.params['bilby_pars']):
                    # if sample out of range, delete the sample                                              the y data (size changes by factor of  n_filter/(2**n_redsteps) )
                    if sampset_1[k,i] < 0.0 or sampset_1[k,i] > 1.0:
                        true_XS = np.delete(true_XS,del_cnt,axis=1)
                        del_cnt-=1
                        break
                    # check m1 > m2
                    elif q == 'mass_1' or q == 'mass_2':
                        m1_idx = np.argwhere(self.params['bilby_pars']=='mass_1')
                        m2_idx = np.argwhere(self.params['bilby_pars']=='mass_2')
                        if sampset_1[m1_idx,i] < sampset_1[m2_idx,i]:
                            true_XS = np.delete(true_XS,del_cnt,axis=1)
                            del_cnt-=1
                            break
            # transpose
            true_XS = np.zeros([samples.shape[0],len(inf_ol_idx)])
            for i in range(true_XS.shape[1]):
                true_XS[:,i] = sampset_1[i,:]
            return true_XS

        
        def compute_kl(sampset_1,sampset_2,samplers,one_D=False):
            """
            Compute JS divergence for one test case.
            """

            # Remove samples outside of the prior mass distribution           
            cur_max = self.params['n_samples']
            
            # Iterate over parameters and remove samples outside of prior
            if samplers[0] == 'vitamin1' or samplers[1] == 'vitamin2':

                # Apply mask
                sampset_1 = sampset_1.T
                sampset_2 = sampset_2.T
                set1 = sampset_1
                set2 = sampset_2
                del_cnt_set1 = 0
                del_cnt_set2 = 0
                params_to_infer = self.params['inf_pars']
                for i in range(set1.shape[1]):

                    # iterate over each parameter in first set
                    for k,q in enumerate(params_to_infer):
                        # if sample out of range, delete the sample
                        if set1[k,i] < 0.0 or set1[k,i] > 1.0:
                            sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                            del_cnt_set1-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(params_to_infer=='mass_1')
                            m2_idx = np.argwhere(params_to_infer=='mass_2')
                            if set1[m1_idx,i] < set1[m2_idx,i]:
                                sampset_1 = np.delete(sampset_1,del_cnt_set1,axis=1)
                                del_cnt_set1-=1
                                break

                    del_cnt_set1+=1

                # iterate over each sample
                for i in range(set2.shape[1]):

                    # iterate over each parameter in second set
                    for k,q in enumerate(params_to_infer):
                        # if sample out of range, delete the sample
                        if set2[k,i] < 0.0 or set2[k,i] > 1.0:
                            sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                            del_cnt_set2-=1
                            break
                        # check m1 > m2
                        elif q == 'mass_1' or q == 'mass_2':
                            m1_idx = np.argwhere(params_to_infer=='mass_1')
                            m2_idx = np.argwhere(params_to_infer=='mass_2')
                            if set2[m1_idx,i] < set2[m2_idx,i]:
                                sampset_2 = np.delete(sampset_2,del_cnt_set2,axis=1)
                                del_cnt_set2-=1
                                break

                    del_cnt_set2+=1

                del_final_idx = np.min([del_cnt_set1,del_cnt_set2])
                set1 = sampset_1[:,:del_final_idx]
                set2 = sampset_2[:,:del_final_idx]

            else:

                set1 = sampset_1.T
                set2 = sampset_2.T
      
            # Iterate over number of randomized sample slices
            SMALL_CONSTANT = 1e-162 # 1e-4 works best for some reason
            def my_kde_bandwidth(obj, fac=1.0):

                """We use Scott's Rule, multiplied by a constant factor."""

                return np.power(obj.n, -1./(obj.d+4)) * fac
            if one_D:
                kl_result_all = np.zeros((1,len(self.params['inf_pars'])))
                for r in range(len(self.params['inf_pars'])):
                    if self.params['gen_indi_KLs'] == True:
                        p = gaussian_kde(set1[r],bw_method=my_kde_bandwidth)#'scott') # 7.5e0 works best ... don't know why. Hope it's not over-smoothing results.
                        q = gaussian_kde(set2[r],bw_method=my_kde_bandwidth)#'scott')#'silverman') # 7.5e0 works best ... don't know why.   
                        # Compute JS Divergence
                        log_diff = np.log((p(set1[r])+SMALL_CONSTANT)/(q(set1[r])+SMALL_CONSTANT))
                        kl_result = (1.0/float(set1.shape[1])) * np.sum(log_diff)

                        # compute symetric kl
                        anti_log_diff = np.log((q(set2[r])+SMALL_CONSTANT)/(p(set2[r])+SMALL_CONSTANT))
                        anti_kl_result = (1.0/float(set1.shape[1])) * np.sum(anti_log_diff)
                        kl_result_all[:,r] = kl_result + anti_kl_result
#                        kl_result = estimate(np.expand_dims(set1[r],1),np.expand_dims(set2[r],1))
#                        kl_result_all[:,r] = kl_result 
                    else:
                        kl_result_all[:,r] = 0   

                return kl_result_all
            else:
                
                kl_result = []
                set1 = set1.T
                set2 = set2.T
                #for kl_idx in range(10):
                rand_idx_kl = np.linspace(0,self.params['n_samples']-1,num=self.params['n_samples'],dtype=np.int)
                np.random.shuffle(rand_idx_kl)
                rand_idx_kl = rand_idx_kl[:1000]
                rand_idx_kl_2 = np.linspace(0,self.params['n_samples']-1,num=self.params['n_samples'],dtype=np.int)
                np.random.shuffle(rand_idx_kl_2)
                rand_idx_kl_2 = rand_idx_kl_2[:1000]
                kl_result.append(estimate(set1[rand_idx_kl,:],set2[rand_idx_kl_2,:],k=10,n_jobs=10) + estimate(set2[rand_idx_kl_2,:],set1[rand_idx_kl,:],k=10,n_jobs=10))
                
                kl_result = np.mean(kl_result)
                return kl_result

                """
                # TODO: comment this out when not doing dynesty kl with itself. Use above expression instead.
                kl_result_mean = []
                kl_result_std = []
                set1 = set1.T
                set2 = set2.T
                for kl_idx in range(10):
                    rand_idx_kl = np.random.choice(np.linspace(0,set1.shape[0]-1,dtype=np.int),size=100)
                    rand_idx_kl_2 = np.random.choice(np.linspace(0,set2.shape[0]-1,dtype=np.int),size=100)
                    new_kl = estimate(set1[rand_idx_kl,:],set2[rand_idx_kl_2,:]) + estimate(set2[rand_idx_kl_2,:],set1[rand_idx_kl,:])
                    kl_result_mean.append(new_kl)
                    kl_result_std.append(new_kl**2)
                kl_result_mean = np.mean(kl_result_mean)
                kl_result_std = np.sqrt(np.mean(kl_result_std))
 
                return kl_result_std, kl_result_mean
                """   

        # Define variables 
        usesamps = params['samplers']
        samplers = params['samplers']
        fig_samplers = params['figure_sampler_names']
        indi_fig_kl, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(6,6))  
        indi_axis_kl = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
      
        # Compute kl divergence on all test cases with preds vs. benchmark
        # Iterate over samplers
        tmp_idx=len(usesamps)
        print_cnt = 0
        runtime = {}
        CB_color_cycle = ['orange', 'purple', 'green',
                  'blue', '#a65628', '#984ea3',
                  '#e41a1c', '#dede00', 
                  '#004d40','#d81b60','#1e88e5',
                  '#ffc107','#1aff1a','#377eb8',
                  '#fefe62','#d35fb7','#dc3220']
        label_idx = 0
        vi_pred_made = None

        
        if params['load_plot_data'] == False:
            try:
                hf = h5py.File('plotting_data_%s/JS_plot_data.h5' % params['run_label'], 'w')
            except:
                os.remove('plotting_data_%s/JS_plot_data.h5' % params['run_label'])
                hf = h5py.File('plotting_data_%s/JS_plot_data.h5' % params['run_label'], 'w')
        else:
            hf = h5py.File('plotting_data_%s/JS_plot_data.h5' % params['run_label'], 'r')
       
        # 4 pannel JS approach
        fig_kl, axis_kl = plt.subplots(2,2,figsize=(8,6),sharey=True,sharex=True)
        grey_colors = ['#00FFFF','#FF00FF','#800000','#23F035']
        # This will go through values of 0 to 3 in steps of 1
        for k in range(len(usesamps)-1):
            print_cnt = 0
            label_idx = 0
            tmp_idx = len(usesamps)
            if k <= 1:
               kl_idx_1 = 0
               kl_idx_2 = k
            elif k > 1:
               kl_idx_1 = 1
               kl_idx_2 = (k-2)

            tot_kl_grey = np.array([])
            # run through values from 0 to 4 (total of 5 elements)
            #grey_cnt=1
            for i in range(len(usesamps)):
                for j in range(tmp_idx):

                    sampler1, sampler2 = samplers[i], samplers[::-1][j]

                    # Check if sampler results already exists in hf directories
                    h5py_node1 = '%s-%s' % (sampler1,sampler2)
                    h5py_node2 = '%s-%s' % (sampler2,sampler1)
                    node_exists = False
                    if h5py_node1 in hf.keys() or h5py_node2 in hf.keys():
                        node_exists=True

                    # Get KL results
                    if samplers[i] == samplers[::-1][j]: # Skip samplers with themselves
                        print_cnt+=1
                        continue
                    elif self.params['load_plot_data'] == True or node_exists: # load pre-generated JS results
                        tot_kl = np.array(hf['%s-%s' % (sampler1,sampler2)])
                    else:
                        if self.params['load_plot_data'] == False: # Make new JS results if needed
                            if sampler1 != 'vitamin':
                                set1 = load_samples(params, sampler1, pp_plot=True)
                            elif sampler1 == 'vitamin' and vi_pred_made == None:
                                set1 = np.zeros((params['r'], params['n_samples'], len(params['bilby_pars'])))
                                for sig_test_idx in range(params['r']):
                                    samples,_ = np.array(gen_samples(model, np.expand_dims(sig_test[sig_test_idx], axis=0), ramp=1, nsamples=params['n_samples'])) 
                                    set1[sig_test_idx,:,:] = extract_correct_sample_idx(samples, inf_ol_idx)
                                vi_pred_made = [set1]
                            if sampler2 != 'vitamin':
                                set2 = load_samples(params, sampler2, pp_plot=True)
                            elif sampler1 == 'vitamin' and vi_pred_made == None:
                                set2 = np.zeros((params['r'], params['n_samples'], len(params['bilby_pars'])))
                                for sig_test_idx in range(params['r']):
                                    samples,_ = np.array(gen_samples(model, np.expand_dims(sig_test[sig_test_idx], axis=0), ramp=1, nsamples=params['n_samples']))
                                    set2[sig_test_idx,:,:] = extract_correct_sample_idx(samples, inf_ol_idx)
                                vi_pred_made = [set2]

                        # Iterate over test cases
                        tot_kl = []  # total JS over all infered parameters

                        for r in range(self.params['r']):
                            tot_kl.append(compute_kl(set1[r],set2[r],[sampler1,sampler2]))
                            print()
                            print('... Completed JS for set %s-%s and test sample %s with mean JS: %.6f' % (sampler1,sampler2,str(r), np.mean(tot_kl)))
                            print()
                        tot_kl = np.array(tot_kl)

                    # Try saving both sampler permutations of the JS results
                    try:
                        if self.params['load_plot_data'] == False:
                            # Save results to h5py file
                            hf.create_dataset('%s-%s' % (sampler1,sampler2), data=tot_kl)
                    except Exception as e:
                        pass
                    try:
                        if self.params['load_plot_data'] == False:
                            # Save results to h5py file
                            hf.create_dataset('%s-%s' % (sampler2,sampler1), data=tot_kl)
                    except Exception as e:
                        pass

                    logbins = np.logspace(-3,2.5,50)
                    logbins_indi = np.logspace(-3,3,50)

                    # plot colored hist
                    if samplers[i] == 'vitamin' and samplers[::-1][j] == samplers[1:][k]:
                        cur_bayesian_sampler = fig_samplers[::-1][j]
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,alpha=0.5,histtype='stepfilled',density=True,color=CB_color_cycle[print_cnt],label=r'$\mathrm{%s \ vs. \ %s}$' % (fig_samplers[::-1][j],fig_samplers[i]),zorder=2)
                        axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor=CB_color_cycle[print_cnt],zorder=10)
                    # record non-colored hists
                    elif samplers[i] != 'vitamin' and samplers[::-1][j] != 'vitamin':
                        if samplers[i] == samplers[1:][k] or samplers[::-1][j] == samplers[1:][k]:
                            if cur_bayesian_sampler == fig_samplers[i]:
                                second_bayesian_sampler = fig_samplers[::-1][j]
                                grey_cnt = fig_samplers.index(fig_samplers[::-1][j]) - 1
                            else:
                                second_bayesian_sampler = fig_samplers[i]
                                grey_cnt = fig_samplers.index(fig_samplers[i]) - 1 
                            axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,alpha=0.8,histtype='stepfilled',density=True,color=grey_colors[grey_cnt],label=r'$\mathrm{%s \ vs. \ %s}$' % (cur_bayesian_sampler,second_bayesian_sampler),zorder=1)
                            axis_kl[kl_idx_1,kl_idx_2].hist(tot_kl,bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor='grey',zorder=1)
                            #grey_cnt+=1
                            #if grey_cnt==4:
                            #    grey_cnt = 1
                            #tot_kl_grey = np.append(tot_kl_grey,tot_kl)
                            print()
                            print('... Grey Mean total JS between %s-%s: %s' % ( sampler1, sampler2, str(np.mean(tot_kl)) ) )
                    print()
                    print_cnt+=1
                tmp_idx-=1
            """
            # make segmented grey histograms for Chris and I
            grey_cnt = 0
            for sampler_grey in range(1, len(self.params['samplers'][1:])):
                start_idx = grey_cnt*125; end_idx = (grey_cnt+1)*125
                print(start_idx, end_idx)
                axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze()[start_idx:end_idx],bins=logbins,alpha=0.8,histtype='stepfilled',density=True,color=grey_colors[sampler_grey],label=r'$\mathrm{%s \ vs. \ other \ samplers}$' % fig_samplers[1:][k],zorder=1)
                axis_kl[kl_idx_1,kl_idx_2].hist(np.array(tot_kl_grey).squeeze()[start_idx:end_idx],bins=logbins,histtype='step',density=True,facecolor='None',ls='-',lw=2,edgecolor='grey',zorder=1)
                grey_cnt+=1
            """
            # plot JS histograms
            if kl_idx_1 == 1:
                axis_kl[kl_idx_1,kl_idx_2].set_xlabel(r'$\mathrm{JS-Statistic}$',fontsize=14)
            if kl_idx_2 == 0:
                axis_kl[kl_idx_1,kl_idx_2].set_ylabel(r'$p(\mathrm{JS})$',fontsize=14)
            leg = axis_kl[kl_idx_1,kl_idx_2].legend(loc='upper right',  fontsize=6) #'medium')
            for l in leg.legendHandles:
                l.set_alpha(1.0)

            axis_kl[kl_idx_1,kl_idx_2].set_xlim(left=1e-3,right=100)
            axis_kl[kl_idx_1,kl_idx_2].set_xscale('log')
            locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=25)
            axis_kl[kl_idx_1,kl_idx_2].xaxis.set_minor_locator(locmin)
            locmaj = matplotlib.ticker.LogLocator(base=10, numticks=25)
            axis_kl[kl_idx_1,kl_idx_2].xaxis.set_major_locator(locmaj)
            axis_kl[kl_idx_1,kl_idx_2].set_yscale('log')
            axis_kl[kl_idx_1,kl_idx_2].grid(False)
            print()
            print('... Made hist plot %d' % k)
            print()

        # Save figure
        fig_kl.canvas.draw()
        plt.tight_layout()
        fig_kl.savefig('%s/latest_%s/hist-JS.png' % (self.params['plot_dir'],self.params['run_label']),dpi=360)
        plt.close(fig_kl)
        hf.close()
        print()
        print('... Saved JS plot to -> %s/latest_%s/hist-JS.png' % (self.params['plot_dir'],self.params['run_label']))
        print()
        return 
