from typing import Sequence, Tuple, Callable

import SALib as salib
from tqdm import tqdm
import numpy as np
import pandas as pd

from OFM_sensitivity import ISIR_SensitivityExperiments


def apply_SA_analysis_over_all_outputs(
        sa_exp_obj: ISIR_SensitivityExperiments,
        analysis: Callable,
        analysis_kwargs: dict = {},
):
    """
    Convenience function to apply sensitivity analysis Analyze method over SA runs (thus only works for SA runs instead
    of normal runs)
    :param sa_exp_obj: SA Experiments object, which includes a 'problem attribute'
    :param analysis: SA Analyze method
    :param analysis_kwargs: keyword args to be passed into the analyze method
    :return: Dict with analysis indices
    """
    sa_analysis = {}
    for out_label in tqdm(sa_exp_obj.problem['outputs']):
        out_idx = sa_exp_obj.problem['outputs'].index(out_label)

        sa_analysis[out_label] = analysis(
            problem = sa_exp_obj.problem,
            X = sa_exp_obj.problem.samples,
            Y = sa_exp_obj.problem.results[:, out_idx], # pre-sectioned model outputs
            **analysis_kwargs,
        )

    return sa_analysis

def multicol_match_get_index(ser,
                             ref:pd.DataFrame,
                             col_labels_to_match:Sequence[str],
                             return_key='days_10k_cumulative'):
    """
    Ad-hoc function to find the cumulative days for the no flight ban scenario. Used with the Dataframe.apply() method below.
    :param col_labels_to_match: column names to match to
    :param ser: Dataframe to search, providing the columns for matching
    :param ref: Reference dataframe to search in for match
    :param return_key: the column value(s) to return
    :return: a single numerical value, which would be stacked as a Series with Dataframe.apply()
    """
    # Convert values from columns of interest into index, for easy matching.
    ref2 = ref.assign(idx=ref.index).set_index(col_labels_to_match)['idx']
    ser2 = ser.set_index(col_labels_to_match)

    return np.vectorize(lambda x: ref2.loc[x])(ser2.index)