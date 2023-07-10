from __future__ import annotations
from typing import Sequence, TYPE_CHECKING
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import TwoSlopeNorm
# import matplotlib.ticker as ticker ## special scaling stuff
# import cmasher as cmr
# import colorcet as cet
# from importlib import reload  ## for reloading packages
import pickle
import functools
if TYPE_CHECKING:
    from OFM_model import ISIR_PolicyExperiments


# TODO trim out unnecessary imports later

def read_experiments_from_pickle(exps_path:str):
    with open(exps_path, 'rb') as exp_file:
        exps_obj = pickle.load(exp_file)

    return exps_obj

def add_extra_metrics(exp_obj: ISIR_PolicyExperiments|None = None,
                      cols_to_split = ('infected_new_imports','imported'),
                      results: pd.DataFrame | None = None,
                      ) -> pd.DataFrame:
    # convenience function
    # could either take Model-level output or Experiment_postprocessed output
    if results is None:
        results: pd.DataFrame = exp_obj.results_postprocess
        results = results.reset_index(names=['exp_id', 'day'])
    else:
        results['day'] = results.index  # append since this won't be present at Model-level output

    # split out the columns for imports
    for to_split in cols_to_split:
        imports_split = pd.DataFrame(results[to_split].to_list())
        col_names = [ f'{to_split}_{col}' for col in imports_split.columns]
        results[col_names] = imports_split

    results.drop(columns=list(cols_to_split), inplace=True)

    results['cum_infected'] = results['infected_total'] + results['isolated']
    results['cum_infected_pct'] = results['cum_infected'] / exp_obj.c_pop_total
    # elapsed day from nominal day (not policy start day!)
    results['elapsed_nominal'] = results['day'] - exp_obj.c_nominal_ref_date
    results['elapsed_policy'] = results['day'] - results.p_FlightBans
    results['p_FlightBans_rel'] = results['p_FlightBans'].map(exp_obj.p_flightbans_map)
    results['p_FlightBans_nom'] = results['p_FlightBans'] - exp_obj.c_nominal_ref_date
    return results


def add_threshold_time_search(df_to_search: pd.DataFrame,
                              df_to_add: pd.DataFrame,
                             threshold):
    # TODO: df_to_search is a pivoted table, include in documentation
    find_start = (df_to_search > threshold).idxmax(axis=0)
    find_end = (df_to_search.loc[::-1] > threshold).idxmax(axis=0)  # reversed search
    find_start = df_to_search.apply(np.searchsorted, axis=1, v=threshold, side='left')
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

