#%%
import os, sys
os.chdir('D:\Downloads\@Books\@TPM\@PDPC\@data_analysis\model_build')
sys.path.append('D:\Downloads\@Books\@TPM\@PDPC\@data_analysis\model_build')
# ^ just a sanity thing: console starts from project root, but pycharm 'run in console' runs with 'model_build' as working directory

import OFM_model as ofm
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cmasher as cmr
import colorcet as cet
from importlib import reload  ## for reloading packages
import pickle
import scipy
# from importlib import reload

#%%
imports_path = "data_input/omi_prevalence_flights_not_inflated.pickletable"

imports_df = pd.read_pickle(imports_path)
# flightbans_set1 = [-2, 0, 2, 7, 14, 28, 56]
# flightbans_set2 = range(-7, 15, 1)  # 1-day increment since -7 till 14 (note python count)
# p_flightbans = set(flightbans_set1) | set(flightbans_set2)
p_flightbans = tuple(range(-7, 57, 1))  # 1-day increments since -7 (week earlier) till 56 days
s_flightbans = tuple(range(-7, 8, 1))
s_imports = tuple(range(15, 121, 15))

# R_range_set1 = np.arange(20, 31, 5) / 10
# R_range_set2 = np.arange(11, 20, 1) /10  # 1.1 till 1.9
# R_range = sorted(list(set(R_range_set2) | set(R_range_set1)))
R_range = tuple(np.arange(11,31,1)/10)
model_vers = 'v2'
pop_NL = int(17.48e6)  # 17.48 million

#%%
# generate infectious distribution (from An der Heiden's serial interval) for v2
available_distributions = {
    'alpha': (2.0, 0.44),
    'delta': (2.11, 0.50),
    'omicron': (2.14, 0.59)
}
# TODO: mention which paper to cite
a = 2.14
rate = .59
x = range(0, 13, 1)  # ends on the 12th day Todo document why
func_serial = scipy.stats.gamma.pdf(x=x, a=a, scale=1 / rate, loc=-1)  # loc '-1' is the cutoff
func_serial_cdf = scipy.stats.gamma.cdf(x=x, a=a, scale=1 / rate, loc=-1)  # for illustration
# Note: truncation of day 0 entry
#%%
# run one experiment set for sensitivity with day0 policy, another for flightban policy choice
#
sim_sens = True
sim_pols = False

if sim_sens:
    experiments_sens = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_df,
        p_FlightBans=s_flightbans,
        u_Rzero=R_range,
        u_ImportsFlights=s_imports,
        u_Func_Infectious=(func_serial,), # v2
        c_SimTime=500,
        c_import_scaling_mode=2, # v1.2
        c_model_engine='step_v2', # v2
    )
    experiments_sens.run_experiments()
    with open(f'data_output/experiment_sens_rev2_model{model_vers}.pickleobject', 'wb') as exp_file:
        pickle.dump(experiments_sens, exp_file)

if sim_pols:
    experiments_pols = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_df,
        p_FlightBans=p_flightbans,
        u_Rzero=(2.,),
        u_ImportsFlights=(15,),
        u_Func_Infectious=(func_serial,), # NEW
        c_SimTime=300,
        c_import_scaling_mode=2,
        c_model_engine='step_v2'
    )
    experiments_pols.run_experiments()
    with open(f'data_output/experiment_pols_model{model_vers}.pickleobject', 'wb') as exp_file:
        pickle.dump(experiments_pols, exp_file)

# ** import_scaling_mode:
# mode 1 = calibrate import function for N total imports prior to (not including) reference date
# mode 2 = calibrate import function for N total imports only on the reference date

#%%
# v_sum = pd.DataFrame.from_dict(experiments_sens.imports_inflated, orient='columns')
# reload(ofm)

# try function call
# experiments_sens.sort_variables(input_params=experiments_sens.PARAMS_RAW)

#%%

#%%
# save the model object (may depreciate with further additions)

    # pickle.dump(experiments.params_raw, exp_file)


#%%