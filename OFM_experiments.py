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
import pickle
import scipy

# from importlib import reload

# %%
imports_path = "data_input/infectious_func_rw_rev3.pickletable"

imports_df = pd.read_pickle(imports_path)
imports_func = imports_df['infect_presence']
ref_day = imports_df.loc[imports_df['date'] == '2021-11-26', :].index[0]  # get reference day (26 Nov 2021) from time
# axis of imports function (for reliability)


# p_flightbans = tuple([*range(-7, 15, 1), None])  # + no policy

R_range = list(np.arange(12, 18, 1) / 10) + [2., 2.5]
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
available_distributions = {
    'alpha': (2.0, 0.44),  # *
    'delta': (2.11, 0.50),  # *
    'omicron': (2.14, 0.59),  # *
    'omi_high1': (2.29, 0.6676),
    'omi_high2': (2.45, 0.7453),
    'omi_high3':(2.59,0.8229),
    # 'omi_low1': (1.99, 0.5124),
    # 'omi_low2': (1.85, 0.4347)
} #*list of gamma function parameters from An der Heiden and Buchholz 2022
a = 2.14
rate = .59
x = range(0, 14, 1)  # default: ends on the 13th day in Pythonic count

func_serial = 'omicron',scipy.stats.gamma.pdf(x=x, a=a, scale=1 / rate, loc=-1)  # loc '-1' is the cutoff
func_serial_cdf = scipy.stats.gamma.cdf(x=x, a=a, scale=1 / rate, loc=-1)  # for illustration
# Note: truncation of day 0 entry since model calculation is for 'next day cases'.
# create alternative infection functions (derived from SA)
func_serial_variable_pdf = tuple((name, scipy.stats.gamma.pdf(x=x, a=params[0], scale=1/params[1], loc=-1)) for name, params in available_distributions.items())
func_serial_variable_cdf = tuple(scipy.stats.gamma.cdf(x=x, a=ax, scale=1/rx, loc=-1) for ax,rx in available_distributions.values())
# %%
# run one experiment set for sensitivity with day0 policy, another for flightban policy choice
#
model_vers = 'v6'
# naming for organisation
# v6: added simplified calculation for Model (step_v2b) that does not reduce infectivity from remaining proportion of susceptible population
# v5.2d: test effect of different truncation point for infectious function
# v5.2c2: test change to infection function math: no normalisation done
# v5.2c1: temporary build for a minor Model-level math optimisation
# v5.2c: minor math correction for 2% leakage factor (2% of direct imports, not all).
# v5 for multiprocessing and adjustments,  v5.1 for change in indirect importation application

# ** import_scaling_mode:
# mode 1 = calibrate import function for N total imports prior to (not including) reference date
# mode 2 = calibrate import function for N total imports only on the reference date
model_engine = 'step_v2b_variable_beta_constant_R'  # addition for v6
sim_import_scaling = True
sim_policies = True
sim_policies_indirect = True
sim_different_scaling = False
sim_different_Ifunc = False

if __name__ == "__main__":
    # make directory if model_version directory not present
    save_path = f'output_figures/{model_vers}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created new directory in 'data_output' with label {model_vers}")

    if sim_import_scaling:
        exps_imp_scaling = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=(0,),
            u_Rzero=R_range,
            u_ImportsFlights=(15, 30, 60),
            u_ImportsIndirect=(0, 0.118, 0.216, 0.314),
            u_Func_Infectious=(func_serial,),  # v2
            c_SimTime=150,
            c_model_engine=model_engine,  # v2
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
            c_model_engine=model_engine,
            c_nominal_ref_date=ref_day,
        )

        setup_run_save_experiments(exps_policies, save_name='Policies', model_vers=model_vers, save_results=True)

    if sim_policies_indirect:  # variation of exps_import_scaling, with indirect flights that are not blocked
        exps_imp_indir = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=list(range(-2, 28, 1)) + [500],  # 500 = no policy
            u_Rzero=(1.1, 1.3, 1.5, 1.7, 2.0),
            u_ImportsFlights=(15,),
            u_ImportsIndirect=(0.0, 0.02, 0.118, 0.216),
            u_Func_Infectious=(func_serial,),
            c_SimTime=150,
            c_model_engine=model_engine,
            save_constants=True,
            c_nominal_ref_date=ref_day,
        )
        setup_run_save_experiments(exps_imp_indir, save_name='Policies and Indirect', model_vers=model_vers,
                                   save_results=True)


    if sim_different_scaling:  # variation of exps_import_scaling, with indirect flights that are not blocked
        exps_imp_indir = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=list(range(-2, 28, 1)) + [500],  # 500 = no policy
            u_Rzero=(1.1, 1.3, 1.5, 1.7, 2.0),
            u_ImportsFlights=(5, 10, 15, 20, 25, 30),
            u_ImportsIndirect=(0.118,),
            u_Func_Infectious=(func_serial,),
            c_SimTime=200,
            c_model_engine=model_engine,
            save_constants=True,
            c_nominal_ref_date=ref_day,
        )
        setup_run_save_experiments(exps_imp_indir, save_name='Policies+R+Scaling', model_vers=model_vers,
                                   save_results=True)

    if sim_different_Ifunc:  # variation of exps_import_scaling, with indirect flights that are not blocked
        exps_imp_indir = ofm.ISIR_PolicyExperiments(
            u_ImportsFunc=imports_func,
            p_FlightBans=list(range(-2, 28, 1)) + [500],
            u_Rzero=(1.1, 1.3, 1.5, 1.7, 2.0),
            u_ImportsFlights=(15,),
            u_ImportsIndirect=(0.118,),
            u_Func_Infectious=func_serial_variable_pdf,
            c_SimTime=200,
            c_model_engine=model_engine,
            save_constants=True,
            c_nominal_ref_date=ref_day,
        )
        setup_run_save_experiments(exps_imp_indir, save_name='IFunc', model_vers=model_vers,
                                   save_results=True)

