# %%
import os, sys

os.chdir('E:\Downloads (E)\@Books\@TPM\@PDPC\@data_analysis\model_build')
sys.path.append('E:\Downloads (E)\@Books\@TPM\@PDPC\@data_analysis\model_build')
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
imports_path = "data_input/infectious_func_rw_rev3.pickletable"

imports_df = pd.read_pickle(imports_path)
imports_func = imports_df['infect_presence']
ref_day = imports_df.loc[imports_df['date'] == '2021-11-26', :].index[0]  # get reference day (26 Nov 2021) from time
# axis of imports function (for reliability)


p_flightbans = tuple([*range(-7, 15, 1), None])  # + no policy
# s_imports = tuple(range(15, 61, 15))

# R_range_set1 = np.arange(20, 31, 5) / 10
# R_range_set2 = np.arange(11, 20, 1) /10  # 1.1 till 1.9
# R_range = sorted(list(set(R_range_set2) | set(R_range_set1)))
# R_range = tuple(np.arange(11, 31, 1) / 10)
# R_range = (1.3, 1.5, 2., 2.5, 3.)
R_range = list(np.arange(12, 18, 1) / 10)  + [2., 2.5]
pop_NL = int(17.48e6)  # 17.48 million


def setup_run_save_experiments(
        exps_set: ofm.ISIR_PolicyExperiments,
        save_name: str,
        save_results: bool = True,
        save_dir: str = "data_output",
        model_vers=None,
        n_workers=5,
) -> None:
    # exps_set.run_experiments()
    # TODO: check if MP mode could be run with normal set?
    # TODO: consider that normal MP output is not postprocessed as normal. Should save the output as a results_verbose item, with the metadata in the results_metadata (see line 307 in model level)
    mp_done = False  # initially False for now?
    mp_done: bool = exps_set.run_experiments_multiprocess(n_workers=n_workers)
    while not mp_done:
        # wait till multiprocessing loop is done?
        pass
    if save_results:
        with open(f'{save_dir}/model{model_vers}_exps_{save_name}_.pickleobject', 'wb') as save_file:
            pickle.dump(exps_set, save_file)

    print(f'saved experiment outputs: model{model_vers}_exps_{save_name}')



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
model_vers = 'v5.2b'  # v5 for multiprocessing and adjustments,  v5.1 for change in indirect importation application
sim_reference = False
sim_import_scaling = False
sim_policies = False
sim_policies_indirect = False

if __name__ == "__main__":
    ## Reference case: not deemed important anymore
    # if sim_reference:
    #     exps_reference = ofm.ISIR_PolicyExperiments(
    #         u_ImportsFunc=imports_func,
    #         p_FlightBans=p_flightbans,
    #         u_Rzero=R_range,
    #         u_ImportsFlights=(15, 30,),
    #         u_ImportsIndirect=(0, 0.2),
    #         u_Func_Infectious=(func_serial,),  # v2
    #         c_SimTime=150,
    #         c_model_engine='step_v2_variable_beta',  # v2
    #         c_nominal_ref_date=ref_day,
    #     )
    #     setup_run_save_experiments(exps_reference, save_name='Reference', model_vers=model_vers, save_results=True)

    if sim_import_scaling:
        exps_imp_scaling = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=(0,),
            u_Rzero=R_range,
            u_ImportsFlights=(15, 30, 60),
            u_ImportsIndirect=(0, 0.1, 0.2, 0.3),
            u_Func_Infectious=(func_serial,),  # v2
            c_SimTime=150,
            c_model_engine='step_v2_variable_beta',  # v2
            save_constants=True,
            c_nominal_ref_date=ref_day,
        )
        setup_run_save_experiments(exps_imp_scaling, save_name='ImportScale', model_vers=model_vers, save_results=True)

    if sim_policies:
        exps_policies = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=(-2, -1, 0, 1, 2, 7, 500),
            u_Rzero=(1.3,),
            u_ImportsFlights=(15,),
            u_ImportsIndirect=(0.118,),
            u_Func_Infectious=(func_serial,),
            c_SimTime=150,
            c_model_engine='step_v2_variable_beta',
            c_nominal_ref_date=ref_day,
        )

        setup_run_save_experiments(exps_policies, save_name='Policies', model_vers=model_vers,save_results=True)

    # if sim_constant_beta: # variation of exps_policies but with older infectivity function (constant)
    #     exps_constant_beta = ofm.ISIR_PolicyExperiments(
    #         u_ImportsFunc=imports_func,
    #         p_FlightBans=p_flightbans,
    #         u_Rzero=(2.,),
    #         u_ImportsFlights=(15,),
    #         u_TIncub=(3,),
    #         c_model_engine='step_v1_constant_beta',
    #         c_SimTime=300,
    #     )
    #     setup_run_save_experiments(exps_constant_beta, save_name='ConstantBeta', model_vers=model_vers,save_results=True)

    if sim_policies_indirect: # variation of exps_import_scaling, with indirect flights that are not blocked
        exps_imp_indir = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans= list(range(-2, 28, 1))+ [500],  # 500 = no policy
            u_Rzero=(1.1, 1.3,1.5, 1.7, 2.0),
            u_ImportsFlights=(15,),
            u_ImportsIndirect=(0.0, 0.02, 0.118, 0.216),
            u_Func_Infectious=(func_serial,),
            c_SimTime=150,
            c_model_engine='step_v2_variable_beta',
            save_constants=True,
            c_nominal_ref_date=ref_day,
        )
        # exps_imp_indir.run_experiments_multiprocess(n_workers=5)
        # with open(f'data_output/modelv5_exps_PoliciesIndirect.pickleobject', 'wb') as savefile:
        #     pickle.dump(exps_imp_indir, savefile)
        setup_run_save_experiments(exps_imp_indir, save_name='Policies and Indirect', model_vers=model_vers, save_results=True)

# ** import_scaling_mode:
# mode 1 = calibrate import function for N total imports prior to (not including) reference date
# mode 2 = calibrate import function for N total imports only on the reference date
