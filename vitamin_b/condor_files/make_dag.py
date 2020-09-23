import sys
import numpy as np

# Defining the list of parameter that need to be fed into the models
def get_params():
    ndata = 1024
    rand_pars='mass_1,mass_2,luminosity_distance,geocent_time,phase,ra,dec,theta_jn,psi,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl'
    r = 2 # total number of samples to make
    bilby_results_label = '1024Hz_full_15par'
    ref_geocent_time=1126259642.5   # reference gps time
    params = dict(
        start_job_idx=0,
        ndata = ndata,
        bilby_results_label=bilby_results_label, # label given to results for bilby posteriors
        print_values=True,            # optionally print values every report interval
        duration = 1.0,               # the timeseries length in seconds
        r = r,                                # the grid dimension for the output tests
        rand_pars=rand_pars,
        ref_geocent_time=ref_geocent_time,            # reference gps time
        testing_data_seed=44,
        inf_pars='mass_1,mass_2,luminosity_distance,phase,geocent_time,ra,dec,theta_jn,psi,a_1,a_2,tilt_1,tilt_2,phi_12,phi_jl',#,'geocent_time','phase','theta_jn','psi'], # parameter names
#        pe_dir='/scratch/hunter.gabbard/condor_runs/%s/test' % bilby_results_label,  # location of bilby PE results
        pe_dir = 'test',
        samplers='vitamin,emcee1,ptemcee1,dynesty1,cpnest1,emcee2,ptemcee2,dynesty2,cpnest2',          # samplers to use when plotting
        doPE = True,                          # if True then do bilby PE
        )
    return params

def add_job(the_file, job_number, **kwargs):

    job_id="%s%.6u" % ('GW', job_number)
    the_file.write("JOB %s %s.sub\n" % (job_id, 'GW'))
    vars_line=" ".join(['%s="%s"'%(arg,str(val))
                        for arg,val in kwargs.items()])
    the_file.write("VARS %s %s\n" % (job_id, vars_line))
    the_file.write("\n")

if __name__ == "__main__":

    # Get training/test data and parameters of run
    params=get_params()
    f = open("params_%s.txt" % params['bilby_results_label'],"w")
    f.write( str(params) )
    f.close()

    r = params['r']
    test_samples=int(r)

    fdag = open("my.dag",'w')

    # iterate over each job
    
    # first iterate over the number of test samples
    job_id_idx = 0 + params['start_job_idx']
    for idx in range(test_samples):
        for idx_samp in range(len(params['samplers'].split(","))-1):
            samplers_to_use = str(params['samplers'].split(",")[0]+','+params['samplers'].split(",")[1:][idx_samp])
            add_job(fdag, job_id_idx, samplingfrequency='%s' % str(params['ndata']/params['duration']),
                                                      duration='%s' % str(params['duration']),
                                                      Ngen='%s' % str(1),
                                                      refgeocenttime='%s' % str(params['ref_geocent_time']),
                                                      bounds='%s' % str('condor_run'),
                                                      fixedvals='%s' % str('condor_run'),
                                                      randpars='%s' % (params['rand_pars']),
                                                      infpars='%s' % (params['inf_pars']),
                                                      label='%s' % str(params['bilby_results_label'] + '_' + str(idx)),
                                                      outdir='%s' % str(params['pe_dir']),
                                                      training='%s' % str(False),
                                                      seed='%s' % str(params['testing_data_seed']+idx),
                                                      dope='%s' % str(params['doPE']),
                                                      samplers='%s' % (samplers_to_use))
            job_id_idx += 1
