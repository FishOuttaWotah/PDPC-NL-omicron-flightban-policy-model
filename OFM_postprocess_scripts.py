from typing import Sequence
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker ## special scaling stuff
import cmasher as cmr
import colorcet as cet
from importlib import reload  ## for reloading packages
import pickle
from OFM_model import ISIR_PolicyExperiments


# TODO trim out unnecessary imports later

def read_experiments_from_pickle(exps_path:str):
    with open(exps_path, 'rb') as exp_file:
        exps_obj = pickle.load(exp_file)

    return exps_obj

def add_extra_metrics(exp_obj: ISIR_PolicyExperiments,
                      ) -> pd.DataFrame:
    # convenience function
    # metadata = pd.DataFrame.from_dict(exp_obj.results_metadata, orient='index')
    results: pd.DataFrame = exp_obj.results_postprocess
    results = results.reset_index(names=['exp_id', 'day'])

    results['cum_infected'] = results['infected_total'] + results['isolated']
    results['cum_infected_pct'] = results['cum_infected'] / exp_obj.c_pop_total
    # elapsed day from nominal day (not policy start day!)
    results['elapsed_nominal'] = results['day'] - exp_obj.c_nominal_ref_date
    results['elapsed_policy'] = results['day'] - results.p_FlightBans
    results['p_FlightBans_rel'] = results['p_FlightBans'].map(exp_obj.p_flightbans_map)
    return results


def add_threshold_time_search(df_to_search: pd.DataFrame,
                              df_to_add: pd.DataFrame,
                             threshold):
    # TODO: df_to_search is a pivoted table, include in documentation
    find_start = (df_to_search > threshold).idxmax(axis=0)
    find_end = (df_to_search.loc[::-1] > threshold).idxmax(axis=0)  # reversed search
    duration = find_end - find_start
    max_value = df_to_search.max(axis=0)
    max_day = df_to_search.idxmax(axis=0)
    df_to_add_n = df_to_add.copy()  # make non-destructive changes
    add_columns = {
        f'duration_{threshold}': duration,
        f'start_{threshold}': find_start,
        f'end_{threshold}': find_end,
        'max': max_value,
        'max_day': max_day
    }
    return df_to_add_n.assign(**add_columns) # add new columns

