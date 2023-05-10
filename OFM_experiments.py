# %%
import os, sys

os.chdir('D:\Downloads\@Books\@TPM\@PDPC\@data_analysis\model_build')
sys.path.append('D:\Downloads\@Books\@TPM\@PDPC\@data_analysis\model_build')
# ^ just a sanity thing: console starts from project root, but pycharm 'run in console' runs with 'model_build' as working directory

from typing import Mapping
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

# %%
imports_path = "data_input/omi_prevalence_flights_not_inflated.pickletable"

imports_df = pd.read_pickle(imports_path)
imports_func = imports_df['import_pax']
# flightbans_set1 = [-2, 0, 2, 7, 14, 28, 56]
# flightbans_set2 = range(-7, 15, 1)  # 1-day increment since -7 till 14 (note python count)
# p_flightbans = set(flightbans_set1) | set(flightbans_set2)
p_flightbans = tuple(range(-7, 57, 1))  # 1-day increments since -7 (week earlier) till 56 days
s_flightbans = tuple(range(-7, 8, 1))
s_imports = tuple(range(15, 61, 15))

# R_range_set1 = np.arange(20, 31, 5) / 10
# R_range_set2 = np.arange(11, 20, 1) /10  # 1.1 till 1.9
# R_range = sorted(list(set(R_range_set2) | set(R_range_set1)))
# R_range = tuple(np.arange(11, 31, 1) / 10)
R_range = (1.3, 1.5, 2., 2.5, 3.)
pop_NL = int(17.48e6)  # 17.48 million


def setup_run_save_experiments(
        exps_set: ofm.ISIR_PolicyExperiments,
        save_name: str,
        save_results: bool = True,
        save_dir: str = "data_output/",
        model_vers=None,
) -> None:
    exps_set.run_experiments()

    if save_results:
        with open(f'{save_dir}experiments_{save_name}_model{model_vers}.pickleobject', 'wb') as save_file:
            pickle.dump(exps_set, save_file)



# %%
# generate infectious distribution (from An der Heiden's serial interval) for v2
"""
Infectious distribution is derived from An der Heiden & Buchholz (2022)
source: https://doi.org/10.1017/S0950268822001248

It uses the serial interval distribution for the Omicron infections to represent the infectivity of Omicron over time, fitted to a gamma distribution.

*serial interval = time interval between symptom onset of index case, to symptom onset of subsequent case
"""
# available_distributions = {
#     'alpha': (2.0, 0.44),
#     'delta': (2.11, 0.50),
#     'omicron': (2.14, 0.59)
# }
a = 2.14
rate = .59
x = range(0, 13, 1)  # ends on the 13th day in Pythonic count

func_serial = scipy.stats.gamma.pdf(x=x, a=a, scale=1 / rate, loc=-1)  # loc '-1' is the cutoff
func_serial_cdf = scipy.stats.gamma.cdf(x=x, a=a, scale=1 / rate, loc=-1)  # for illustration
# Note: truncation of day 0 entry since model calculation is for 'next day cases'.
# %%
# run one experiment set for sensitivity with day0 policy, another for flightban policy choice
#
model_vers = 'v4.1'
sim_reference = False
sim_import_scaling = True
sim_policies = False
# sim_import_indirect = False
# sim_preexisting_case = False
sim_constant_beta = False # should be equivalent to not have beta function

if sim_reference:
    exps_reference = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_func,
        p_FlightBans=s_flightbans,
        u_Rzero=R_range,
        u_ImportsFlights=(0, 15, 30, 45, 60, 75, 90),
        u_Func_Infectious=(func_serial,),  # v2
        c_SimTime=500,
        c_model_engine='step_v2_variable_beta',  # v2
    )
    setup_run_save_experiments(exps_reference, save_name='Reference', model_vers=model_vers, save_results=True)

if sim_import_scaling:
    exps_imp_scaling = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_func,
        p_FlightBans=(0,),
        u_Rzero=R_range,
        u_ImportsFlights=(0, 15, 30, 60),
        u_ImportsIndirect=(0, 0.5, 1, 2),
        u_Func_Infectious=(func_serial,),  # v2
        c_SimTime=500,
        c_model_engine='step_v2_variable_beta',  # v2
        save_constants=True,
    )
    setup_run_save_experiments(exps_imp_scaling, save_name='ImportScale_smallindirect', model_vers=model_vers, save_results=True)

if sim_policies:
    exps_policies = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_func,
        p_FlightBans=s_flightbans,
        u_Rzero=(1.3, 1.5, 2., 2.5, 3.),
        u_ImportsFlights=(15, 30,),
        u_ImportsIndirect= range(0, 5, 1),
        u_Func_Infectious=(func_serial,),
        c_SimTime=500,
        c_model_engine='step_v2_variable_beta'
    )
    setup_run_save_experiments(exps_policies, save_name='Policies_small indirect', model_vers=model_vers,save_results=True)

if sim_constant_beta: # variation of exps_policies but with older infectivity function (constant)
    exps_constant_beta = ofm.ISIR_PolicyExperiments(
        u_ImportsFunc=imports_func,
        p_FlightBans=p_flightbans,
        u_Rzero=(2.,),
        u_ImportsFlights=(15,),
        u_TIncub=(3,),
        c_model_engine='step_v1_constant_beta',
        c_SimTime=300,
    )
    setup_run_save_experiments(exps_constant_beta, save_name='ConstantBeta', model_vers=model_vers,save_results=True)

# if sim_import_indirect: # variation of exps_import_scaling, with indirect flights that are not blocked
#     exps_imp_indir = ofm.ISIR_PolicyExperiments(
#         u_ImportsFunc=imports_func,
#         p_FlightBans=s_flightbans,
#         u_Rzero= R_range,
#         u_ImportsFlights=s_imports,
#         u_ImportsIndirect=(15,30),
#         u_Func_Infectious=(func_serial,),
#         c_SimTime=500,
#         c_model_engine='step_v2_variable_beta',
#     )
#     setup_run_save_experiments(exps_imp_indir, save_name='ImportIndirect', model_vers=model_vers, save_results=True)



# ** import_scaling_mode:
# mode 1 = calibrate import function for N total imports prior to (not including) reference date
# mode 2 = calibrate import function for N total imports only on the reference date
