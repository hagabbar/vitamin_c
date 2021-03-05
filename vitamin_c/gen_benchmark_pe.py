#! /usr/bin/env python

""" Script to generate training and testing samples
"""

from __future__ import division, print_function

import numpy as np
import bilby
from sys import exit
import os, glob, shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import scipy
import lalsimulation
import lal
import time
import h5py
from scipy.ndimage.interpolation import shift
import argparse
from gwpy.timeseries import TimeSeries

def parser():
    """ Parses command line arguments

    Returns
    -------
        arguments
    """

    #TODO: complete help sections
    parser = argparse.ArgumentParser(prog='bilby_pe.py', description='script for generating bilby samples/posterior')

    # arguments for data
    parser.add_argument('-samplingfrequency', type=float, help='sampling frequency of signal')
    parser.add_argument('-samplers', nargs='+', type=str, help='list of samplers to use to generate')
    parser.add_argument('-duration', type=float, help='duration of signal in seconds')
    parser.add_argument('-Ngen', type=int, help='number of samples to generate')
    parser.add_argument('-refgeocenttime', type=float, help='reference geocenter time')
    parser.add_argument('-bounds', type=str, help='dictionary of the bounds')
    parser.add_argument('-fixedvals', type=str, help='dictionary of the fixed values')
    parser.add_argument('-randpars', nargs='+', type=str, help='list of pars to randomize')
    parser.add_argument('-infpars', nargs='+', type=str, help='list of pars to infer')
    parser.add_argument('-label', type=str, help='label of run')
    parser.add_argument('-outdir', type=str, help='output directory')
    parser.add_argument('-training', type=str, help='boolean for train/test config')
    parser.add_argument('-seed', type=int, help='random seed')
    parser.add_argument('-dope', type=str, help='boolean for whether or not to do PE')
    

    return parser.parse_args()

def gen_real_noise(duration,
                 sampling_frequency,
                 det,
                 ref_geocent_time, psd_files=[],
                 real_noise_seg =[None,None]
                 ):
    """ pull real noise samples
    """

    # compute the number of time domain samples
    Nt = int(sampling_frequency*duration)

    # Get ifos bilby variable
    ifos = bilby.gw.detector.InterferometerList(det)

    start_open_seg, end_open_seg = real_noise_seg # 1 sec noise segments
    for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
        time_series = TimeSeries.find('%s:GDS-CALIB_STRAIN' % det[ifo_idx],
                      start_open_seg, end_open_seg) # pull timeseries data using gwpy
        ifo.set_strain_data_from_gwpy_timeseries(time_series=time_series) # input new ts into bilby ifo

    noise_sample = ifos[0].strain_data.frequency_domain_strain # get frequency domain strain
    noise_sample /= ifos[0].amplitude_spectral_density_array # assume default psd from bilby
    noise_sample = np.sqrt(2.0*Nt)*np.fft.irfft(noise_sample) # convert frequency to time domain

    return noise_sample
    

def gen_template(duration,
                 sampling_frequency,
                 pars,
                 ref_geocent_time, psd_files=[],
                 use_real_det_noise=False,
                 real_noise_seg =[None,None]
                 ):
    """ Generates a whitened waveforms in Gaussian noise.

    Parameters
    ----------
    duration: float
        duration of the signal in seconds
    sampling_frequency: float
        sampling frequency of the signal
    pars: dict
        values of source parameters for the waveform
    ref_geocent_time: float
        reference geocenter time of injected signal
    psd_files: list
        list of psd files to use for each detector (if other than default is wanted)
    use_real_det_noise: bool
        if True, use real ligo noise around ref_geocent_time
    real_noise_seg: list
        list containing the starting and ending times of the real noise 
        segment

    Returns
    -------
    whitened noise-free signal: array_like
    whitened noisy signal: array_like
    injection_parameters: dict
        source parameter values of injected signal
    ifos: dict
        interferometer properties
    waveform_generator: bilby function
        function used by bilby to inject signal into noise 
    """

    if sampling_frequency>4096:
        print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
        exit(0)

    # compute the number of time domain samples
    Nt = int(sampling_frequency*duration)

    # define the start time of the timeseries
    start_time = ref_geocent_time-duration/2.0

    # fix parameters here
    injection_parameters = dict(
        mass_1=pars['mass_1'],mass_2=pars['mass_2'], a_1=pars['a_1'], a_2=pars['a_2'], tilt_1=pars['tilt_1'], tilt_2=pars['tilt_2'],
        phi_12=pars['phi_12'], phi_jl=pars['phi_jl'], luminosity_distance=pars['luminosity_distance'], theta_jn=pars['theta_jn'], psi=pars['psi'],
        phase=pars['phase'], geocent_time=pars['geocent_time'], ra=pars['ra'], dec=pars['dec'])

    # Fixed arguments passed into the source model
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                              reference_frequency=20., minimum_frequency=20.)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=start_time)

    # create waveform
    wfg = waveform_generator

    # extract waveform from bilby
    wfg.parameters = injection_parameters
    freq_signal = wfg.frequency_domain_strain()
    time_signal = wfg.time_domain_strain()

    # Set up interferometers. These default to their design
    # sensitivity

    ifos = bilby.gw.detector.InterferometerList(pars['det'])

    # If user is specifying PSD files
    if len(psd_files) > 0:
        type_psd = psd_files[0].split('/')[-1].split('_')[-1].split('.')[0]
        for int_idx,ifo in enumerate(ifos):
            if type_psd == 'psd':
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_files[0])
            elif type_psd == 'asd':
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=psd_files[0])
            else:
                print('Could not determine whether psd or asd ...')
                exit()

    ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=start_time)

    # inject signal
    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters)

    print('... Injected signal')
    whitened_signal_td_all = []
    whitened_h_td_all = [] 
    # iterate over ifos
    for i in range(len(pars['det'])):
        # get frequency domain noise-free signal at detector
        signal_fd = ifos[i].get_detector_response(freq_signal, injection_parameters) 

        # whiten frequency domain noise-free signal (and reshape/flatten)
        whitened_signal_fd = signal_fd/ifos[i].amplitude_spectral_density_array
        #whitened_signal_fd = whitened_signal_fd.reshape(whitened_signal_fd.shape[0])    

        # get frequency domain signal + noise at detector
        h_fd = ifos[i].strain_data.frequency_domain_strain

        # inverse FFT noise-free signal back to time domain and normalise
        whitened_signal_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)

        # whiten noisy frequency domain signal
        whitened_h_fd = h_fd/ifos[i].amplitude_spectral_density_array
    
        # inverse FFT noisy signal back to time domain and normalise
        whitened_h_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_h_fd)
        
        whitened_h_td_all.append([whitened_h_td])
        whitened_signal_td_all.append([whitened_signal_td])

    print('... Whitened signals')
    return np.squeeze(np.array(whitened_signal_td_all),axis=1),np.squeeze(np.array(whitened_h_td_all),axis=1),injection_parameters,ifos,waveform_generator

def run(sampling_frequency=256.0,
           duration=1.,
           N_gen=1000,
           bounds=None,
           fixed_vals=None,
           rand_pars=[None],
           inf_pars=[None],
           ref_geocent_time=1126259642.5,
           training=True,
           do_pe=False,
           label='test_results',
           out_dir='bilby_output',
           seed=None,
           samplers=['vitamin','dynesty'],
           condor_run=False,
           params=None,
           det=['H1','L1','V1'],
           psd_files=[],
           use_real_det_noise=False,
           use_real_events=False,
           samp_idx=False,
           ):
    """ Main function to generate both training sample time series 
    and test sample time series/posteriors.

    Parameters
    ----------
    sampling_frequency: float
        sampling frequency of the signals
    duration: float
        duration of signals in seconds
    N_gen: int
        number of test/training timeseries to generate
    bounds: dict
        allowed bounds of timeseries source parameters
    fixed_vals: dict
        fixed values of source parameters not randomized
    rand_pars: list
        source parameters to randomize
    inf_pars: list
        source parameters to infer
    ref_geocent_time: float
        reference geocenter time of injected signals
    training: bool
        if true, generate training timeseries
    do_pe: bool
        if true, generate posteriors in addtion to test sample time series
    label: string
        label to give to saved files
    out_dir: string
        output directory of saved files
    seed: float
        random seed for generating timeseries and posterior samples
    samplers: list
        samplers to use when generating posterior samples
    condor_run: bool
        if true, use setting to make condor jobs run properly
    params: dict
        general script run parameters
    det: list
        detectors to use
    psd_files
        optional list of psd files to use for each detector
    """

    # Set up a random seed for result reproducibility.  This is optional!
    if seed is not None:
        np.random.seed(seed)

    # Set up a PriorDict, which inherits from dict.
    priors = bilby.gw.prior.BBHPriorDict()
    priors.pop('chirp_mass')
    priors['mass_ratio'] = bilby.gw.prior.Constraint(minimum=0.125, maximum=1, name='mass_ratio', latex_label='$q$', unit=None)
    if np.any([r=='geocent_time' for r in rand_pars]):
        priors['geocent_time'] = bilby.core.prior.Uniform(
                minimum=ref_geocent_time + bounds['geocent_time_min'],
                maximum=ref_geocent_time + bounds['geocent_time_max'],
                name='geocent_time', latex_label='$t_c$', unit='$s$')
    else:
        priors['geocent_time'] = fixed_vals['geocent_time']

    if np.any([r=='mass_1' for r in rand_pars]):
        print('inside m1')
        priors['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=bounds['mass_1_min'], maximum=bounds['mass_1_max'],unit='$M_{\odot}$')
    else:
        priors['mass_1'] = fixed_vals['mass_1']

    if np.any([r=='mass_2' for r in rand_pars]):
        priors['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=bounds['mass_2_min'], maximum=bounds['mass_2_max'],unit='$M_{\odot}$')
    else:
        priors['mass_2'] = fixed_vals['mass_2']

    if np.any([r=='a_1' for r in rand_pars]):
        priors['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=bounds['a_1_min'], maximum=bounds['a_1_max'])
    else:
        priors['a_1'] = fixed_vals['a_1']

    if np.any([r=='a_2' for r in rand_pars]):
        priors['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=bounds['a_2_min'], maximum=bounds['a_2_max'])
    else:
        priors['a_2'] = fixed_vals['a_2']

    if np.any([r=='tilt_1' for r in rand_pars]):
        priors['tilt_1'] = bilby.gw.prior.Uniform(name='tilt_1', minimum=bounds['tilt_1_min'], maximum=bounds['tilt_1_max'])
    else:
        priors['tilt_1'] = fixed_vals['tilt_1']

    if np.any([r=='tilt_2' for r in rand_pars]):
       priors['tilt_2'] = bilby.gw.prior.Uniform(name='tilt_2', minimum=bounds['tilt_2_min'], maximum=bounds['tilt_2_max'])
    else:
        priors['tilt_2'] = fixed_vals['tilt_2']

    if np.any([r=='phi_12' for r in rand_pars]):
        priors['phi_12'] = bilby.gw.prior.Uniform(name='phi_12', minimum=bounds['phi_12_min'], maximum=bounds['phi_12_max'], boundary='periodic')
    else:
        priors['phi_12'] = fixed_vals['phi_12']

    if np.any([r=='phi_jl' for r in rand_pars]):
        priors['phi_jl'] = bilby.gw.prior.Uniform(name='phi_jl', minimum=bounds['phi_jl_min'], maximum=bounds['phi_jl_max'], boundary='periodic')
    else:
        priors['phi_jl'] = fixed_vals['phi_jl']

    if np.any([r=='ra' for r in rand_pars]):
        priors['ra'] = bilby.gw.prior.Uniform(name='ra', minimum=bounds['ra_min'], maximum=bounds['ra_max'], boundary='periodic')
    else:
        priors['ra'] = fixed_vals['ra']

    if np.any([r=='dec' for r in rand_pars]):
        pass
    else:
        priors['dec'] = fixed_vals['dec']

    if np.any([r=='psi' for r in rand_pars]):
        priors['psi'] = bilby.gw.prior.Uniform(name='psi', minimum=bounds['psi_min'], maximum=bounds['psi_max'], boundary='periodic')
    else:
        priors['psi'] = fixed_vals['psi']

    if np.any([r=='theta_jn' for r in rand_pars]):
        pass
    else:
        priors['theta_jn'] = fixed_vals['theta_jn']

    if np.any([r=='phase' for r in rand_pars]):
        priors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=bounds['phase_min'], maximum=bounds['phase_max'], boundary='periodic')
    else:
        priors['phase'] = fixed_vals['phase']

    if np.any([r=='luminosity_distance' for r in rand_pars]):
        priors['luminosity_distance'] =  bilby.gw.prior.Uniform(name='luminosity_distance', minimum=bounds['luminosity_distance_min'], maximum=bounds['luminosity_distance_max'], unit='Mpc')
    else:
        priors['luminosity_distance'] = fixed_vals['luminosity_distance']

    # generate training samples
    if training == True:
        train_samples = real_noise_array = []
        snrs = []
        train_pars = np.zeros((N_gen,len(rand_pars)))
        for i in range(N_gen):
            
            # sample from priors
            pars = priors.sample()
            pars['det'] = det
           
            # store the params
            temp = []
            for p_idx,p in enumerate(rand_pars):
                for q,qi in pars.items():
                    if p==q:
                        #temp.append(qi-ref_geocent_time) if p=='geocent_time' else temp.append(qi)
                        if p == 'geocent_time':
                            train_pars[i,p_idx] = qi-ref_geocent_time
                        else:
                            train_pars[i,p_idx] = qi
            #train_pars.append([temp])

            # make the data - shift geocent time to correct reference
            train_samp_noisefree, train_samp_noisy,_,ifos,_ = gen_template(duration,sampling_frequency,
                                                                           pars,ref_geocent_time,psd_files,
                                                                           use_real_det_noise=use_real_det_noise
                                                                           )
            train_samples.append([train_samp_noisefree,train_samp_noisy])
            small_snr_list = [ifos[j].meta_data['optimal_SNR'] for j in range(len(pars['det']))]
            snrs.append(small_snr_list)
            #train_samples.append(gen_template(duration,sampling_frequency,pars,ref_geocent_time)[0:2])
            print('Made waveform %d/%d' % (i,N_gen)) 

        train_samples_noisefree = np.array(train_samples)[:,0,:]
        snrs = np.array(snrs) 
#        train_pars = np.array(train_pars)
        return train_samples_noisefree,train_pars,snrs

    # otherwise we are doing test data 
    else:
       
        # generate parameters
        pars = priors.sample()
        pars['det'] = det
        temp = []
        for p in rand_pars:
            for q,qi in pars.items():
                if p==q:
                    temp.append(qi-ref_geocent_time) if p=='geocent_time' else temp.append(qi)      

        # inject signal - shift geocent time to correct reference
        test_samples_noisefree,test_samples_noisy,injection_parameters,ifos,waveform_generator = gen_template(duration,sampling_frequency,
                                   pars,ref_geocent_time,psd_files)

        # get test sample snr
        snr = np.array([ifos[j].meta_data['optimal_SNR'] for j in range(len(pars['det']))])

        # if not doing PE then return signal data
        if not do_pe:
            return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr

        try:
            bilby.core.utils.setup_logger(outdir=out_dir, label=label)
        except Exception as e:
            print(e)
            pass

        # Initialise the likelihood by passing in the interferometer data (ifos) and
        # the waveform generator
        phase_marginalization=True
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, phase_marginalization=phase_marginalization,
            priors=priors)

        # save test waveform information
        try:
            os.mkdir('%s' % (out_dir+'_waveforms'))
        except Exception as e:
            print(e)
            pass

        if params != None:
            hf = h5py.File('%s/data_%d.h5py' % (out_dir+'_waveforms',int(label.split('_')[-1])),'w')
            for k, v in params.items():
                try:
                    hf.create_dataset(k,data=v)
                except:
                    pass

            hf.create_dataset('x_data', data=np.array([temp]))
            for k, v in bounds.items():
                hf.create_dataset(k,data=v)
            hf.create_dataset('y_data_noisefree', data=test_samples_noisefree)
            hf.create_dataset('y_data_noisy', data=test_samples_noisy)
            hf.create_dataset('rand_pars', data=np.string_(params['rand_pars']))
            hf.create_dataset('snrs', data=snr)
            hf.close()

        # look for dynesty sampler option
        if np.any([r=='dynesty1' for r in samplers]) or np.any([r=='dynesty2' for r in samplers]) or np.any([r=='dynesty' for r in samplers]):

            run_startt = time.time()
            # Run sampler dynesty 1 sampler

            result = bilby.run_sampler(
                likelihood=likelihood, priors=priors, sampler='dynesty', npoints=1000, nact=50, npool=8, dlogz=27.0,
                injection_parameters=injection_parameters, outdir=out_dir+'_'+samplers[-1], label=label,
                save='hdf5', plot=True)
            run_endt = time.time()

            # save test sample waveform
            hf = h5py.File('%s_%s/%s.h5py' % (out_dir,samplers[-1],label), 'w')
            hf.create_dataset('noisy_waveform', data=test_samples_noisy)
            hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)

            # loop over randomised params and save samples
            #for p in inf_pars:
            for q,qi in result.posterior.items():
            #if p==q:
                name = q + '_post'
                print('saving PE samples for parameter {}'.format(q))
                hf.create_dataset(name, data=np.array(qi))
            hf.create_dataset('runtime', data=(run_endt - run_startt))
            hf.close()

            # return samples if not doing a condor run
            if condor_run == False:
                # Make a corner plot.
                result.plot_corner()
                # remove unecessary files
                png_files=glob.glob("%s_dynesty1/*.png*" % (out_dir))
                hdf5_files=glob.glob("%s_dynesty1/*.hdf5*" % (out_dir))
                pickle_files=glob.glob("%s_dynesty1/*.pickle*" % (out_dir))
                resume_files=glob.glob("%s_dynesty1/*.resume*" % (out_dir))
                filelist = [png_files,hdf5_files,pickle_files,resume_files]
                for file_type in filelist:
                    for file in file_type:
                        os.remove(file)
                print('finished running pe')
                return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr

            run_startt = time.time()

        # look for cpnest sampler option
        if np.any([r=='cpnest1' for r in samplers]) or np.any([r=='cpnest2' for r in samplers]) or np.any([r=='cpnest' for r in samplers]):

            # run cpnest sampler 1 
            run_startt = time.time()
            result = bilby.run_sampler(
                likelihood=likelihood, priors=priors, sampler='cpnest',
                nlive=2048,maxmcmc=1000, seed=1994, nthreads=10,
                injection_parameters=injection_parameters, outdir=out_dir+'_'+samplers[-1], label=label,
                save='hdf5', plot=True)
            run_endt = time.time()

            # save test sample waveform
            hf = h5py.File('%s_%s/%s.h5py' % (out_dir,samplers[-1],label), 'w')
            hf.create_dataset('noisy_waveform', data=test_samples_noisy)
            hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)

            # loop over randomised params and save samples
            #for p in inf_pars:
            for q,qi in result.posterior.items():
            #if p==q:
                name = q + '_post'
                print('saving PE samples for parameter {}'.format(q))
                hf.create_dataset(name, data=np.array(qi))
            hf.create_dataset('runtime', data=(run_endt - run_startt))
            hf.close()

            # return samples if not doing a condor run
            if condor_run == False:
                # remove unecessary files
                png_files=glob.glob("%s_cpnest1/*.png*" % (out_dir))
                hdf5_files=glob.glob("%s_cpnest1/*.hdf5*" % (out_dir))
                other_files=glob.glob("%s_cpnest1/*cpnest_*" % (out_dir))
                resume_files=glob.glob("%s_cpnest1/*.resume*" % (out_dir))
                filelist = [png_files,hdf5_files,pickle_files]
                for file_idx,file_type in enumerate(filelist):
                    for file in file_type:
                        if file_idx == 2:
                            shutil.rmtree(file)
                        else:
                            os.remove(file)
                print('finished running pe')
                return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr

        n_ptemcee_walkers = 250
        n_ptemcee_steps = 5000
        n_ptemcee_burnin = 4000
        # look for ptemcee sampler option
        if np.any([r=='ptemcee1' for r in samplers]) or np.any([r=='ptemcee2' for r in samplers]) or np.any([r=='ptemcee' for r in samplers]):

            # run ptemcee sampler 1
            run_startt = time.time()
            result = bilby.run_sampler(
                likelihood=likelihood, priors=priors, sampler='ptemcee',
#                nwalkers=n_ptemcee_walkers, nsteps=n_ptemcee_steps, nburn=n_ptemcee_burnin, plot=True, ntemps=8,
                nsamples=10000, nwalkers=n_ptemcee_walkers, ntemps=8, plot=True, threads=10,
                injection_parameters=injection_parameters, outdir=out_dir+'_'+samplers[-1], label=label,
                save=False)
            run_endt = time.time()

            # save test sample waveform
            os.mkdir('%s_%s_h5py_files' % (out_dir,samplers[-1]))
            hf = h5py.File('%s_%s_h5py_files/%s.h5py' % (out_dir,samplers[-1],label), 'w')
            hf.create_dataset('noisy_waveform', data=test_samples_noisy)
            hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)

            # throw away samples with "bad" liklihood values
            all_lnp = result.log_likelihood_evaluations
            hf.create_dataset('log_like_eval', data=all_lnp) # save log likelihood evaluations
#            max_lnp = np.max(all_lnp)
#            idx_keep = np.argwhere(all_lnp>max_lnp-12.0).squeeze()
#            all_lnp = all_lnp.reshape((n_ptemcee_steps - n_ptemcee_burnin,n_ptemcee_walkers)) 

#            print('Identified bad liklihood points')

            # loop over randomised params and save samples
            #for p in inf_pars:
            for q,qi in result.posterior.items():
            #if p==q:
                name = q + '_post'
                print('saving PE samples for parameter {}'.format(q))
#                        old_samples = np.array(qi).reshape((n_ptemcee_steps - n_ptemcee_burnin,n_ptemcee_walkers))
#                        new_samples = np.array([])
#                        for m in range(old_samples.shape[0]):
#                            new_samples = np.append(new_samples,old_samples[m,np.argwhere(all_lnp[m,:]>max_lnp-12.0).squeeze()])
                hf.create_dataset(name, data=np.array(qi))
#                        hf.create_dataset(name+'_with_cut', data=np.array(new_samples))
            hf.create_dataset('runtime', data=(run_endt - run_startt))
            hf.close()

            # return samples if not doing a condor run
            if condor_run == False:
                # remove unecessary files
                png_files=glob.glob("%s_ptemcee1/*.png*" % (out_dir))
                hdf5_files=glob.glob("%s_ptemcee1/*.hdf5*" % (out_dir))
                other_files=glob.glob("%s_ptemcee1/*ptemcee_*" % (out_dir))
                resume_files=glob.glob("%s_ptemcee1/*.resume*" % (out_dir))
                filelist = [png_files,hdf5_files,other_files]
                for file_idx,file_type in enumerate(filelist):
                    for file in file_type:
                        if file_idx == 2:
                            shutil.rmtree(file)
                        else:
                            os.remove(file)
                print('finished running pe')
                return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr

        n_emcee_walkers = 250
        n_emcee_steps = 14000
        n_emcee_burnin = 4000
        # look for emcee sampler option
        if np.any([r=='emcee1' for r in samplers]) or np.any([r=='emcee2' for r in samplers]) or np.any([r=='emcee' for r in samplers]):

            # run emcee sampler 1
            run_startt = time.time()
            result = bilby.run_sampler(
            likelihood=likelihood, priors=priors, sampler='emcee', a=1.4, thin_by=10, store=False,
            nwalkers=n_emcee_walkers, nsteps=n_emcee_steps, nburn=n_emcee_burnin,
            injection_parameters=injection_parameters, outdir=out_dir+samplers[-1], label=label,
            save=False,plot=True)
            run_endt = time.time()

            # save test sample waveform
            os.mkdir('%s_h5py_files' % (out_dir+samplers[-1]))
            hf = h5py.File('%s_h5py_files/%s.h5py' % ((out_dir+samplers[-1]),label), 'w')
            hf.create_dataset('noisy_waveform', data=test_samples_noisy)
            hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)

            # throw away samples with "bad" liklihood values
            all_lnp = result.log_likelihood_evaluations
            hf.create_dataset('log_like_eval', data=all_lnp) # save log likelihood evaluations
            max_lnp = np.max(all_lnp)
#            idx_keep = np.argwhere(all_lnp>max_lnp-12.0).squeeze()
            all_lnp = all_lnp.reshape((n_emcee_steps - n_emcee_burnin,n_emcee_walkers))

            print('Identified bad liklihood points')

            print

            # loop over randomised params and save samples  
            #for p in inf_pars:
            for q,qi in result.posterior.items():
            #if p==q:
                name = q + '_post'
                print('saving PE samples for parameter {}'.format(q))
                old_samples = np.array(qi).reshape((n_emcee_steps - n_emcee_burnin,n_emcee_walkers))
                new_samples = np.array([])
                for m in range(old_samples.shape[0]):
                    new_samples = np.append(new_samples,old_samples[m,np.argwhere(all_lnp[m,:]>max_lnp-12.0).squeeze()])
                hf.create_dataset(name, data=np.array(qi))
                hf.create_dataset(name+'_with_cut', data=np.array(new_samples))
                        
            hf.create_dataset('runtime', data=(run_endt - run_startt))
            hf.close()

            # return samples if not doing a condor run
            if condor_run == False:
                # remove unecessary files
                png_files=glob.glob("%s_emcee1/*.png*" % (out_dir))
                hdf5_files=glob.glob("%s_emcee1/*.hdf5*" % (out_dir))
                other_files=glob.glob("%s_emcee1/*emcee_*" % (out_dir))
                resume_files=glob.glob("%s_emcee1/*.resume*" % (out_dir))
                filelist = [png_files,hdf5_files,other_files]
                for file_idx,file_type in enumerate(filelist):
                    for file in file_type:
                        if file_idx == 2:
                            shutil.rmtree(file)
                        else:
                            os.remove(file)
                print('finished running pe')
                return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr

    print('finished running pe')

def main(args):
     
    def get_params():
        params = dict(
           sampling_frequency=args.samplingfrequency,
           duration=args.duration,
           N_gen=args.Ngen,
           bounds=args.bounds,
           fixed_vals=args.fixedvals,
           rand_pars=list(args.randpars[0].split(',')),
           inf_pars=list(args.infpars[0].split(',')),
           ref_geocent_time=args.refgeocenttime,
           training=eval(args.training),
           do_pe=eval(args.dope),
           label=args.label,
           out_dir=args.outdir,
           seed=args.seed,
           samplers=list(args.samplers[0].split(',')),
           condor_run=True

        )

        return params

    params = get_params()
    run(sampling_frequency=args.samplingfrequency,
           duration=args.duration,
           N_gen=args.Ngen,
           bounds=args.bounds,
           fixed_vals=args.fixedvals,
           rand_pars=list(args.randpars[0].split(',')),
           inf_pars=list(args.infpars[0].split(',')),
           ref_geocent_time=args.refgeocenttime,
           training=eval(args.training),
           do_pe=eval(args.dope),
           label=args.label,
           out_dir=args.outdir,
           seed=args.seed,
           samplers=list(args.samplers[0].split(',')),
           condor_run=True,
           params=params)

if __name__ == '__main__':
    args = parser()
    main(args)

