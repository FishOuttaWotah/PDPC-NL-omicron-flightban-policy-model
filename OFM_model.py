from typing import Union, Sequence, Dict, Any, Callable, Mapping
import collections, itertools, random
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool
import time, warnings
import tqdm  # for multiprocessing progress bar
import functools
import traceback

from OFM_postprocess_scripts import add_extra_metrics  # for sensitivity analysis

"""
This Python file contains the description of the Omicron Flights Importation Model 
"""

"""
v5.1: (selfnote) the indirect importation function (for mode 2) is considered n% of the normal importation function, 
only applying after the flight ban (and not before).  

This can be found at 2 points: the self.get_import_function method of ISIR_PolicyExperiments, 
under 'elif self.indirect_import_mode == 2'. 

"""


# what ISIR was supposed to mean (Imported, Susceptible, Infected, Recovered)
class ISIR_PolicyExperiments:  # currently v5
    """
    Base Experiment level of the Omicron Flights model.
    Handles the batch experiment input, creates scenarios and runs the epidemiological model with the scenarios.
    Also handles some minor post-simulation data handling for (separate) post-processing operations
    """

    class INPUTS:
        UNCERTAINTY = 'u_'
        POLICY = 'p_'
        CONSTANT = 'c_'

    # define params to be renamed from Experiment level to Model input level

    PARAMS_NAME_MAP = {
        'u_Rzero': 'R_zero',
        'u_TIncub': "t_incubation",
        'u_ImportsFunc': 'imports_flights',
        'p_FlightBans': 'flightban_on',
        'u_Func_Infectious': 'func_infectious',
        'u_ImportsOther': 'imports_other',
        'c_model_engine': 'engine',
        'c_SimTime': 'sim_time',
        'c_PopTotal': 'pop_total',
        's_starting_S_size': 'starting_S',
        # 'u_ImportsFlights': None, # handled externally (in run_experiments)
        # 'u_ImportsIndirect': None, # (ditto)
    }

    # (depreciated) define params to be excluded from single model run
    # PARAMS_EXCLUDE_FROM_MODEL = (
    #     'self',
    #     'c_nominal_ref_date',
    #     'u_ImportsFunc',
    #     'c_import_scaling_mode',
    #     'debug_mode'
    #
    # )

    # define params to be included in model run
    PARAMS_INCLUDE_IN_SCENARIO_GEN = (
        'u_TIncub',
        'u_Rzero',
        # 'u_ImportsFunc'  # this is a resource to be called by importation scaling
        'u_ImportsFlights',
        'u_ImportsIndirect',
        'u_ImportsOther',
        'c_SimTime',
        'c_PopTotal',
        'c_model_engine',
        'p_FlightBans',
        'u_Func_Infectious',
        's_starting_S_size',
    )

    # not used
    PARAMS_PASS_THROUGH_VARIABLE = (
        'u_ImportsOther',
    )

    # (depreciated)
    PARAMS_PASS_THROUGH_CONSTANT = (

    )

    PARAMS_EXCLUDE_FROM_RESULTS = (
        "u_Func_Infectious",  # a numpy array, would make the output too large
    )

    def __init__(self,
                 u_ImportsFunc: pd.Series,
                 u_TIncub: Sequence[int] = (3,),
                 u_Rzero: Sequence[float] = (1.3,),  # was 3.22 baseline
                 u_ImportsFlights: Sequence[float] | None = (10.,),
                 u_Func_Infectious: Sequence[Sequence[float]] | None = None,
                 u_ImportsIndirect: Sequence[float] | None = None,  # v4 (indirect entry)
                 u_ImportsOther: Sequence[pd.Series] | None = None,  # v4 (miscellaneous manual inputs)
                 p_FlightBans: Sequence[int | None] | None = None,
                 c_SimTime: int = 120,  # days
                 c_PopTotal: int = 17480000,  # 17.48 mil for NL
                 c_nominal_ref_date=17,  # 17 corresponds with 26th Nov in the importation function
                 c_import_scaling_mode=2,  # v3
                 c_model_engine='step_v2_variable_beta',  # v3
                 debug_mode=False,  # v4: only runs one scenario and saves as state
                 save_constants=True,  # v4: decide whether constants are saved in the results_metadate ref
                 s_indirect_import_mode: int = 2,  # v4.1  import mode 2 is a ratio of direct import_func
                 s_starting_S_size: Sequence[float] | None = None,
                 # v4.1  reduced S compartment size (absolute numbers or fraction)
                 sim_mode=True,  # TODO: might be depreciated
                 # v5: True to process as normal (elaborate), False eg. if passing into sensitivity analysis mode
                 ):

        self.params = locals().copy()  # get the model inputs, this WILL be modified for passing into the single model run.
        self.PARAMS_RAW = locals().copy()  # make a non-modified version of the model inputs
        self.debug_unmapped_args = []
        self.debug_mode = debug_mode
        self.debug_experiment = None  # TODO: decide if one experiment or several
        self.save_consts = save_constants
        self.experiments = None  # for parallel processing (for now)

        # Save experiment inputs
        self.u_imports_func = u_ImportsFunc
        self.u_TIncub = u_TIncub  # incubation time
        self.u_Rzero = u_Rzero
        self.p_flightbans = self.get_flightban_days(p_FlightBans,
                                                    c_nominal_ref_date)  # TODO think about more intuitive naming
        self.p_flightbans_map = dict([(d, p_FlightBans[i]) for i, d in enumerate(self.p_flightbans)])
        self.params['p_FlightBans'] = self.p_flightbans
        self.c_sim_time = c_SimTime
        self.c_pop_total = c_PopTotal
        self.c_nominal_ref_date = c_nominal_ref_date
        self.indirect_import_mode = s_indirect_import_mode  # v4.1
        self.s_starting_S = s_starting_S_size  # v4.1

        ## Handle import scaling mode
        # decides how the import function should be (up)scaled
        # 1 = scale the cumulative sum up to nominal ref date to ImportsFlights
        # 2 = (new) scale only the import_func on the nominal ref date to Importflights
        self.import_scaling_mode = c_import_scaling_mode
        if c_import_scaling_mode == 1:
            self.c_import_ref_dates = slice(None, self.c_nominal_ref_date - 1)
        elif c_import_scaling_mode == 2:
            self.c_import_ref_dates = self.c_nominal_ref_date
        else:
            raise NotImplementedError(
                f"OFM_experiments:(c_import_scaling_mode of {c_import_scaling_mode} is invalid for model.")

        ## Experiment attributes for import functions
        self.u_imports_flights = u_ImportsFlights
        self.u_imports_indirect = u_ImportsIndirect  # v4
        self.c_imports_other = u_ImportsOther  # v4
        # if import_func func not provided, these states are all None
        self.i_flights_scaled = self.i_flights_cumul = self.i_flights_infl_factor = self.i_nonflights_scaled = self.i_nonflights_cumul = self.i_nonflights_infl_factor = None

        if sim_mode:
            self.handler_flights_scaling(u_ImportsFunc, u_ImportsFlights, u_ImportsIndirect)
            # ^ modifies the 'i_' attributes defined above.

            # identify which params are varied or constant
            self.params_varied, self.params_const, self.params_null = self.sort_variables(self.params)
            self.scenarios = self.generate_scenarios(
                variables=self.params_varied,
                constants=self.params_const,
                arg_name_map=self.PARAMS_NAME_MAP
            )

            # items for saving results_s
            self.results_metadata = {}  # todo: think should be 'scenarios?'
            self.results_verbose = {}  # instantiate empty dict for saving
            self.results_postprocess = None

    def sort_variables(self, input_params):
        """
        Sorts the input params (meant for the model) into variables and constants.
        The variables set would be used to generate scenarios later.
        Raises an error if a param
        :param input_params: Dict of input params passed into Experiment,
        cleaned for only items that should be used for the Bodel
        :return: tuple with dict of variables and dict of constants
        """
        # exclude params with defined criteria
        input_params = self.filter_included_params(input_params.copy(),
                                                   compare=self.PARAMS_INCLUDE_IN_SCENARIO_GEN)
        # ^ use copy() for non-destructive operations
        p_variable = {}
        p_constant = {}
        p_error = {}
        p_none = []
        for k, v in input_params.items():
            # if there is attribute len, and not present in exclude
            try:  # for catching non-valid len attributes
                # catch criteria:
                # 1. if has valid length attribute (with a non-null value)
                # 2. is not a string (so list, tuple, etc.)
                # 3. if the param is designated in a pass-through set
                if v is None:  # for None-inserted variables
                    p_none.append(k)
                    # continue  # we skip
                # elif k in self.PARAMS_PASS_THROUGH_VARIABLE:  (might be useful at a later stage)
                #     p_variable[k] = v
                elif hasattr(v, "__len__") and isinstance(len(v), int) and not isinstance(v, str):
                    if len(v) > 1:  # if more than 1, thus a variable
                        p_variable[k] = v
                    else:  # if only size 1, thus a constant
                        p_constant[k] = v[0]
                else:  # no len, also a constant
                    p_constant[k] = v
            except Exception as exc:
                print(f'Error: {exc=}, {type(exc)=}')
                p_error[k] = v

        if p_error:
            raise ValueError(f'OFM_experiments: \'{__name__}\' function found invalid key/value(s): {p_error}. '
                             f'This usually means you had a typo, wrong input format, or did not update the '
                             f'PARAMS_INCLUDE_IN_MODEL set')

        return p_variable, p_constant, p_none

    # @staticmethod
    # def filter_excluded_params(params: Dict, compare: Sequence[str] = PARAMS_EXCLUDE_FROM_MODEL):
    #     # keys_to_excl = set(params.keys()) - set(compare)
    #     return dict((k, v) for k, v in params.items() if k not in compare)

    @staticmethod
    def filter_included_params(params: Dict, compare: Sequence[str] = PARAMS_INCLUDE_IN_SCENARIO_GEN):
        keys_to_incl = set(params.keys()) & set(compare)
        return dict((k, params[k]) for k in keys_to_incl)

    @staticmethod
    def generate_scenarios(variables: Dict, constants: Dict,
                           arg_name_map: Dict):

        # code does the following (read from back):
        # 1. creates all possible combinations of (cartesian product) of variables
        # 2. creates a dict with one combo of variables and constants. This serves as the input into the model
        # 3. creates a list with all dicts in 2).
        # TODO: expansion point: if a dict should be given as the output or a list
        # scenarios = [dict(zip(variables.keys(), inputs), **constants) for inputs in itertools.product(*variables.values())]
        scenarios = {}  # was list
        counter = 0
        for vars_set in itertools.product(*variables.values()):
            scenarios[counter] = dict(zip(variables.keys(), vars_set))
            # input_set = dict(zip(variables.keys(), vars_set))  # , **constants)
            # scenarios.append(input_set)
            counter += 1

        return scenarios

    def run_experiments(self):
        # create a bunch of experiments and run sequentially.
        # iterate through all scenarios
        print(f"OFM_PE: Running {len(self.scenarios)} experiments")
        start_time = time.time()
        for exp_idx, exp_vars in self.scenarios.items():
            ## Prepare experiment params for model input
            exp_params = exp_vars.copy()  # make a copy for labelling
            exp_params.update(self.params_const)  # add constants (inefficient, but makes further handling convenient)
            mod_inputs = self.map_input_names(exp_params,
                                              params_name_map=self.PARAMS_NAME_MAP)  # convert from experiment-level param names to model param names

            imports_flights, imports_indirect, mod_inputs = self.get_import_function(model_inputs=mod_inputs)

            # Initiate model
            experiment = ISIRmodel_SingleRun(
                imports_flights=imports_flights,
                imports_indirect=imports_indirect,
                postprocess_to_df=True,
                ref_day=self.c_nominal_ref_date,
                **mod_inputs)

            # Run simulation if not in debug mode
            if not self.debug_mode:
                experiment.run_model()
                ## Saving results_s

                # save either all params or just variables
                if self.save_consts:
                    save_params = exp_params
                else:
                    save_params = exp_vars
                for param_excl in self.PARAMS_EXCLUDE_FROM_RESULTS:  # exclude certain bulky params if present
                    save_params.pop(param_excl, None)

                # save experiment output and meta_s
                self.results_verbose[exp_idx] = experiment.output.assign(**save_params)
                self.results_metadata[exp_idx] = save_params  # save meta_s
            elif self.debug_mode and self.debug_experiment is None:  # preserve 1st experiment
                self.debug_experiment = experiment
                break
            # else: # we only preserve one experiment
            #     break

        if not self.debug_mode:
            self.results_postprocess = self.postprocess_to_df(self.results_verbose,
                                                              map_to_invert=self.PARAMS_NAME_MAP)  # create unified dataframe
            print(f"OFM_PE: all experiments completed in {round((time.time() - start_time) / 60, 1)} mins")
        else:
            print(f'OFM_PE: Debug mode on, 1 experiment initiated, but not run')

    def map_input_names(self, input_set: Dict, params_name_map: Dict):
        """
        Converts parameter names in input_set Dict keys to the target name in params_name_map
        :param input_set: dict whose keys are to be converted
        :param params_name_map: dict containing mapping from original names to target names
        :return: clone of input_set, with converted keys
        """
        # identify which args don't have a mapping
        self.debug_unmapped_args = set(input_set.keys()) - set(params_name_map)
        # if self.debug_unmapped_args:
        #     print(f'Note: experiment-level args {self.debug_unmapped_args} do not have a mapping.')
        mapped = dict((params_name_map.get(k, k), v) for k, v in input_set.items())
        # 'get()' function returns 'k' if not available in input_set
        return mapped

    @staticmethod
    def postprocess_to_df(results_verbose, map_to_invert: Dict | None = None):
        """
        Converts Experiment simulation runs from list of dataframes to one single concatenated pandas DataFrame
        :param results_verbose: list of DataFrames, from Model-level output
        :param map_to_invert: (optional) dict containing value -> key entries, to change the name of DataFrame
        output columns. Note that the mapping is inverse!
        :return: pandas DataFrame of all simulation output
        """
        # concatenate results_s into dataframe and set some added indices
        results_df = pd.concat(results_verbose)
        if map_to_invert is not None:  # Invert column names from model-level to experiment-level
            reverse_map = {v: k for k, v in map_to_invert.items()}
            results_df = results_df.rename(columns=reverse_map)
        return results_df

    # inflate the importation shape for the nominal values provided
    @staticmethod
    def inflate_importation_function(import_func: pd.Series,
                                     nominal_target: Sequence[int | float],
                                     date_slice: Any):
        """
        (low-level method) Scale the importation function multiplicatively, based on a specific importation rate
        within a time window.
        E.G., we scale the entire function such that we have 15 imports (nominal_target) on Nov 26th (date_slice)
        :param import_func: pandas Series object, describing the importation function that governs the 'shape'
        of importation. Row indices describe the days
        :param nominal_target: Array-like containing scaling targets to match
        :param date_slice: pandas-accepted slice object, that represents the time window for scaling
        :return: pd.DataFrame of scaled import functions (number of columns matching the size of nominal_target, and rows
                matching the size of the import_func's index),
                the sum of base imports (unscaled) within time window, and
                list of inflation factors that were multiplied to the base import function.
        """
        # get sum of base imports within time window
        nominal_cumulative = import_func.loc[date_slice].sum()

        # calculate how much the import function should be multiplied to achieve the nominal targets
        inflation_factor = np.array(nominal_target) / nominal_cumulative
        imports_inflated: pd.DataFrame = pd.concat([import_func * factor for factor in inflation_factor], axis=1,
                                                   keys=nominal_target)
        return imports_inflated, nominal_cumulative, list(inflation_factor)

    def handler_flights_scaling(self,
                                u_ImportsFunc: pd.Series,
                                u_ImportsFlights: Sequence[int | float],
                                u_ImportsIndirect):
        """
        Takes the Experiment level input for imports (base import function, import scaling for both direct flights and
        indirect flights) and scales the import function via 'inflate_importation_function' per import scaling params.
        :param u_ImportsFunc: pd.Series, describing the shape of the imported cases over days
        :param u_ImportsFlights: array of numbers describing the target importation scale for direct flights.
        :param u_ImportsIndirect:  array of numbers describing the target importation scale for indirect flights.
        may be handled differently depending on model-level importation mode (indirect_import_mode)
        :return:
        """
        if u_ImportsFunc is not None:
            if u_ImportsFlights is not None:
                self.i_flights_scaled, self.i_flights_cumul, self.i_flights_infl_factor = self.inflate_importation_function(
                    import_func=u_ImportsFunc,
                    nominal_target=self.u_imports_flights,
                    date_slice=self.c_import_ref_dates)

            if u_ImportsIndirect is not None:
                if self.indirect_import_mode == 1:
                    self.i_nonflights_scaled, self.i_nonflights_cumul, self.i_nonflights_infl_factor = self.inflate_importation_function(
                        import_func=u_ImportsFunc,
                        nominal_target=self.u_imports_indirect,
                        date_slice=self.c_import_ref_dates
                    )
                elif self.indirect_import_mode == 2:  # newer mode
                    self.i_nonflights_scaled = True  # duck type, not good practice
                else:
                    raise Exception(f'Argument \'indirect import mode\'{self.indirect_import_mode} undefined,')
        else:
            # if import_func func not provided, these states are all None
            # already handled previously, see __init__()
            # self.i_flights_scaled = self.i_flights_cumul = self.i_flights_infl_factor = self.i_nonflights_scaled = self.i_nonflights_cumul = self.i_nonflights_infl_factor = None
            print('OFM_PE warning: No import function (u_ImportFunc) provided')

    @staticmethod
    def get_flightban_days(flightban_days, ref_date):
        """
        Convert flightban days from relative to non-negative days in the importation function's time
        axis. This is because the flightban days are described relative to the reference date
        (which is more human-intuitive), but the importation function can only describe time in non-negative days.
        :param flightban_days: array of flight ban days relative to the reference day
        :param ref_date: index of reference day in the importation function
        :return: tuple of corrected flight ban days in the importation function time axis
        """
        # convert the (relative) flightban days
        non_int_check = [i for i in flightban_days if type(i) != int]  # update for v4 to handle non-numeric (eg None)
        if non_int_check:
            print(
                f"OFM_PE Warning: {__name__} has non-integer flightban value(s) {non_int_check}. If this is intentional, do proceed.")
        return tuple(ref_date + delta if isinstance(delta, int) else delta for delta in flightban_days)

    def get_import_function(self,
                            model_inputs: Dict):  # TODO: an opaque method, consider reworking
        """
        Retrieves the scaled importation function from the Experiment-level's pre-calculated records.
        Also deletes certain model entries which shouldn't be passed into the Model level.
        :param model_inputs: Dict of model inputs, of which containing the importation scaling for direct and indirect
        flights/entry.
        :return: Direct importation function, Indirect importation function, and cleaned model inputs dict.
        """
        model_inputs = model_inputs.copy()  # prevent unintended casting
        # print(model_inputs)
        # Get scaled import function for model and drop the scaling factor value
        if 'u_ImportsFlights' in model_inputs and self.i_flights_scaled is not None:  # if it is a variable
            imports_flights = self.i_flights_scaled[
                model_inputs.pop('u_ImportsFlights')].copy()  # get which import function to take, and
            # ^ eliminate the entry "u_ImportsFlights", because it shouldn't be given as an input.
        else:
            model_inputs.pop('u_ImportsFlights', None)
            imports_flights = None
            # raise Exception('cannot generate \'import_func\' variable in run_experiments()')

        # check if indirect flights (optional) is available
        if 'u_ImportsIndirect' in model_inputs and self.i_nonflights_scaled is not None:

            # todo: might have an edge condition for none imports but with indirect imports
            if self.indirect_import_mode == 1:
                imports_indirect = self.i_nonflights_scaled[
                    model_inputs.pop('u_ImportsIndirect')].copy()
            elif self.indirect_import_mode == 2:
                indirect_ratio = model_inputs.pop('u_ImportsIndirect')
                imports_indirect = imports_flights * indirect_ratio
                imports_indirect = imports_indirect.loc[model_inputs['flightban_on']:]
                # assumes that
                # imports_indirect = imports_flights / (1 - indirect_ratio) * indirect_ratio
                # updated: such that ratio of indirect + direct = 1
            else:
                raise NotImplementedError()
        else:
            model_inputs.pop('u_ImportsIndirect', None)  # changed from 'u_ImportsFlights'
            imports_indirect = None

        return imports_flights, imports_indirect, model_inputs

    def run_experiments_multiprocess(self,
                                     n_workers=5,
                                     postprocess_ops: Sequence | None = None
                                     ) -> bool:
        print(f'OFM_PE: Running MULTIPROCESS mode with {len(self.scenarios)} experiments')

        # def r_callback(result):
        #     print(result, flush=True)

        def r_error(e):
            traceback.print_exception(type(e), e, e.__traceback__)  # some more details from child process error
            raise Exception(e)

        # dumb sanity check
        # if self.experiments is not None:
        #     raise Exception(f"OFM_PE: Experiments were already run")

        # TODO: check for dict casting issues

        start_time = time.time()
        jobs = []

        # Create multiprocessing job pool and assign simulation tasks
        with multiprocessing.Pool(n_workers) as pool, tqdm.tqdm(total=len(self.scenarios)) as pbar:
            try:
                for idx, inputs in self.scenarios.items():
                    if postprocess_ops is not None:  # turn on postprocessing operations. Outputs will be different
                        args = (idx, inputs, self.params_const, postprocess_ops)
                    else:  # only return model iterations and model outputs
                        args = (idx, inputs, self.params_const)

                    # create single async job with a callback that updates the progress bar UI
                    j = pool.apply_async(
                        func=self.handle_single_run,
                        # handling function that will take the inputs and pass them into the
                        # single model run
                        args=args,  # list of arguments, note cannot handle kwargs
                        callback=lambda _: pbar.update(1),  # update script for progress bar
                        error_callback=r_error
                    )  # returns an AsyncResult object
                    jobs.append(j)

                pool.close()  # close submission of jobs (not strictly necessary)
                self.experiments = [j.get() for j in jobs]  # check and block until all experiment jobs are done
                # ^ a bit of black magic, apparently get() is blocking
            except Exception as e:
                pool.terminate()
                pool.join()
                print('Exception encountered')
                raise e

        print(f'OFM_PE: all experiments completed at {round((time.time() - start_time) / 60, 2)} mins')

        ## If no postprocessing operations are done (ie. sensitivity analysis), reshape the output (list of pd.DataFrame) as a big dataframe
        if postprocess_ops is None:
            self.reshape_MP_output(scenarios=self.scenarios,
                                   results=self.experiments)

        return True  # TODO: why was this here again? as blocking?

    def handle_single_run(self,
                          idx,
                          variables,
                          constants,
                          post_ops: tuple[Callable, Dict] | None = None,
                          ):

        # note/document that model inputs is a tuple with (label,inputs)
        # TODO: consider optional output for meta-data, especially opaque with post-processed output.
        model_inputs = variables.copy()
        model_inputs.update(constants)  # add constants
        model_inputs = self.map_input_names(
            model_inputs,
            params_name_map=self.PARAMS_NAME_MAP)

        # retrieves the associated scaled importation function for the scenario, plus some input cleaning
        imports_flights, imports_indirect, model_inputs = self.get_import_function(model_inputs=model_inputs)

        experiment = ISIRmodel_SingleRun(
            imports_flights=imports_flights,
            imports_indirect=imports_indirect,
            postprocess_to_df=True,  # thus a Pandas ref result should be available as an attribute after simulation
            ref_day=self.c_nominal_ref_date,
            **model_inputs)
        experiment.run_model()  # run model with given scenario
        for param_excl in self.PARAMS_EXCLUDE_FROM_RESULTS:
            variables.pop(param_excl, None)
            constants.pop(param_excl, None)
        experiment.output = experiment.output.assign(**variables, **constants)

        if post_ops is not None:  # if postprocessing operations are provided
            # add useful metrics such as cumulative cases, elapsed time
            experiment.output = add_extra_metrics(
                exp_obj=self,
                results=experiment.output)  # from OFM_postprocess_scripts

            # conduct further postprocessing operations
            post_outputs = {}
            # for loop for easier debugging, at likely a performance cost
            for func, kwargs in post_ops:  # items in post_ops are a tuple of callable and dict (kwargs)
                job = func(self=experiment, **kwargs)
                post_outputs.update(job)
            return idx, post_outputs
        else:  # no post-processing, we return all outputs
            return idx, experiment.output

    def reshape_MP_output(self,
                          scenarios: Dict[int, Dict],
                          results: list[tuple[str, pd.DataFrame]]) -> None:
        """
        Modifies the Multiprocessing output to the more conventional form used for post-simulation data processing
        :param scenarios: dict containing the scenarios (in keyword dict), keyed with the scenario number
        :param results: list of tuples containing the scenario number and output as pd.Dataframe
        :return: None, modifies the Experiment-level attribute in place.
        """
        self.results_verbose = dict(results)  # input variables are already assigned at single-run method
        self.results_metadata = dict(scenarios)
        self.results_postprocess = self.postprocess_to_df(self.results_verbose,
                                                          map_to_invert=self.PARAMS_NAME_MAP)

    def setup_postprocessing(self):  # just a convenience function
        # from OFM_model import ISIRmodel_SingleRun as single
        # highlight, ctrl-shift-num minus/plus to expand fragments
        days_10k_cumulative = (ISIRmodel_SingleRun.post_find_metric,
                               {'return_name': 'days_10k_cumulative',
                                'col_name': 'cum_infected',
                                'metric': 10000,
                                'return_arg': True,
                                'return_corr': -self.c_nominal_ref_date})  # note minus

        days_5k_daily = (ISIRmodel_SingleRun.post_find_metric,
                         {'return_name': 'days_5k_daily',
                          'col_name': 'infected_new',
                          'metric': 5000,
                          'return_arg': True,
                          'return_corr': -self.c_nominal_ref_date})  #

        resultant_infected = (ISIRmodel_SingleRun.post_get_last,
                              {'return_name': 'resultant_infected',
                               'col_name': 'cum_infected_pct'})

        outbreak_shape = (ISIRmodel_SingleRun.post_get_infection_trajectory_shape,
                          {'return_name': ('duration', 'peak_day', 'peak_value'),
                           'col_name': 'infected_new',
                           'metric': 1000,
                           'return_corr': -self.c_nominal_ref_date
                           })

        return days_10k_cumulative, days_5k_daily, resultant_infected, outbreak_shape

    # def __getstate__(self):  might be useful when pickling is required
    #     self_dict = self.__dict__.copy()
    #     del self_dict['experiments']
    #     return self_dict
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)


####################  SINGLE MODEL RUN  ###########################
class ISIRmodel_SingleRun:
    # Describing items that need to be created at the start, for future use
    def __init__(self,
                 *,  # ensures function is called with parameter names
                 R_zero,
                 t_incubation: int,
                 pop_total: int,
                 sim_time: int,
                 imports_flights: pd.Series | None = None,
                 # func_infectious: Sequence[float] | None = None,  ## v3
                 func_infectious: Dict[str, Sequence[float]] | Sequence[float] | None = None,  # v6 change
                 flightban_on: int | None = None,
                 postprocess_to_df: bool = False,
                 engine: str = 'step_v2b_variable_beta_constant_R',
                 imports_indirect: pd.Series | None = None,  # added v4
                 imports_other: pd.Series | None = None,  # added v4
                 starting_S=None,  # added v4.1
                 ref_day=None,  # added v4.1
                 stop_cumulative = 10000, # added v6, for stopping simulation when a cumulative value is applied
                 ):

        # make copy of model inputs
        self.debug_inputs = locals().copy()
        if 'kwargs' in self.debug_inputs:
            self.debug_inputs.update(self.debug_inputs.pop('kwargs'))

        # check if engine choice is valid
        if hasattr(self, engine) and callable(getattr(self, engine)):
            self.engine_name = engine
            self.engine = getattr(self, engine)
        else:
            raise NotImplementedError(f"Cannot find engine '{engine}' in class {self.__class__.__name__}")

        # Handling input params
        self.R_zero = R_zero
        self.t_inc = t_incubation
        self.POP_TOTAL = pop_total  # constant
        self.starting_S = starting_S
        self.stop_cumulative = stop_cumulative

        self.sim_end = sim_time + ref_day  # nominal simulation end if termination condition is not met
        self.flight_ban: int | None | bool = flightban_on
        self.postprocess_to_df: bool = postprocess_to_df
        # describes the weights of variable beta in I compartment

        self.FUNC_INFECTIOUS_NAME, self.FUNC_INFECTIOUS_BASE = func_infectious  # original un-corrected weights
        # self.FUNC_INFECTIOUS = np.array(func_infectious)
        self.FUNC_INFECTIOUS = self.FUNC_INFECTIOUS_BASE / sum(self.FUNC_INFECTIOUS_BASE)  # normalisation  TODO: sensitive!
        self.beta_func_rt = None  # model state

        # v2: early terminate conditions  (think: what is a good determinant?)
        self.TERMINATE_BELOW_R = .5  # if R-eff-rt is below this value, start checking for next termination condition
        self.TERMINATE_BELOW_I_TOTAL = 1.  # if total infectious compartment is less than this, terminate
        # self.TERMINATE_AFTER_CONSECUTIVE_DS =   (10, 0)
        self.terminate = False  # for early termination

        ## Handling for case importation (updated v4)
        if imports_flights is not None:
            if flightban_on is not None:
                imports_flights = imports_flights.loc[
                                  :flightban_on - 1]  # we assume no more imports on the flightban day itself
                # Note: modification of input!

        self.indirect = imports_indirect  # TODO: ad hoc: delete!
        isets_dict = {
            'direct': imports_flights,
            'indirect': imports_indirect,
            'others': imports_other,
        }
        import_sets = dict((label, iset) for label, iset in isets_dict.items()
                           if iset is not None)

        # conduct superimposed import function
        if len(import_sets) >= 1:
            self.imports_components = pd.concat(import_sets, axis=1).fillna(value=0.)  # note fillna (v4.1)
            self.imports_func = self.imports_components.sum(axis=1)
        else:
            self.imports_func = {}  # empty set
            raise Warning('OFM_model warning: Omicron imports function is empty')

        # Instantiate model states
        self.sim_current: int = 0
        if self.starting_S is not None:
            if 0 < self.starting_S <= 1.:  # if fraction of population given
                self.pop_susceptible = self.POP_TOTAL * self.starting_S
            else:  # assume that this gives S size in persons
                self.pop_susceptible = self.starting_S
        else:
            self.pop_susceptible = pop_total
        self.s = self.pop_susceptible / self.POP_TOTAL
        self.pop_isolated = 0
        self.r_rt = 0  # rt = real-time (ie. for that day)
        self.beta_eff_rt = 0
        self.beta_null = self.get_beta_null()  # uses u__R_eff and t_inc
        if engine == 'step_v2_variable_beta' or engine == 'step_v2b_variable_beta_constant_R':
            if func_infectious is None:
                raise Exception(f"OFM_model: \'step_v2_variable_beta\' model engine needs a defined func_infectious")
            # generate a deque (double-sided queue) that maintains a constant length, any new append operation would remove an element from the opposite end.
            self.pop_infectious = collections.deque([0] * len(self.FUNC_INFECTIOUS_BASE),
                                                    maxlen=len(self.FUNC_INFECTIOUS_BASE))  # representation of
            # generate deques for each importation substream (direct, indirect, etc).
            self.pop_imports_comps = dict(
                (col, collections.deque([0] * len(self.FUNC_INFECTIOUS_BASE), maxlen=len(self.FUNC_INFECTIOUS_BASE)))
                for col in self.imports_components.columns)
            self.dS_import = None
            self.dS_native = None


        elif engine == 'step_v1_constant_beta':  ## TODO: depreciated
            self.pop_infectious = collections.deque([0] * t_incubation,
                                                    maxlen=t_incubation)  # representation of infectious population as a python Deque
            self.pop_imports = collections.deque([0] * t_incubation,
                                                 maxlen=t_incubation)
        # ^ another deque to keep population of imports separate

        # Instantiate model recording
        self.record_dict = {}
        self.output: pd.DataFrame | None = None  # clean up model outputs (in dict form) for later processing

    def run_model(self):
        """
        Runs model with the defined simulation engine for the defined number of steps. Handles importation input per step, and conducts a post-processed function at the end.
        """
        # check if simulation end criteria is met
        while self.sim_current <= self.sim_end:
            # retrieve imported cases from external dataset, for current timestep
            if 'step_v2' in self.engine_name:
                if self.sim_current in self.imports_func:
                    imported = self.imports_components.loc[self.sim_current, :]
                else:
                    # provide zero dummy
                    imported = pd.Series(0., index=self.imports_components.columns)

            else:
                if self.sim_current in self.imports_func:
                    imported = self.imports_func[self.sim_current]

                else:
                    imported = 0
            self.engine(imported)
            self.sim_current += 1

            if self.terminate:  # when simulation engine reached terminal state
                break  # terminate governed by step function

        if self.postprocess_to_df:
            self.output = self.post_process()

    def step_v2_variable_beta(self, imported: pd.Series):  # this includes the E compartment
        # if isinstance(imported, pd.Series):
        #     import_comps = imported
        #     imported = imported.sum()
        # else:
        # TODO: possibility to add cumulative summation here instead of post-processing. This might allow simulation termination by the time a certain cumulative value is reached.
        for col, dq in self.pop_imports_comps.items():
            dq.appendleft(imported[col])  # should cast back to the original dict by reference
        # self.pop_imports_comps.appendleft(imported)
        self.s = self.pop_susceptible / self.POP_TOTAL

        self.r_rt = self.R_zero * self.s
        self.beta_func_rt = self.get_I_func_rt(R_eff_rt=self.r_rt)

        # note we have multiple streams now
        self.dS_import: Dict[str, float] = dict((col, np.array(dq) @ self.beta_func_rt)
                                                for col, dq in self.pop_imports_comps.items())  # @ is dot product
        self.dS_native: float = np.array(self.pop_infectious) @ self.beta_func_rt
        i_components = (*self.dS_import.values(), self.dS_native)  # * for unpacking
        sum_i_comps = sum(i_components)
        if sum_i_comps < self.pop_susceptible:
            dS = sum_i_comps
            dS_comps = [round(n, 2) for n in self.dS_import.values()]
        else:
            dS = self.pop_susceptible
            # get ratio of contribution instead
            dS_comps = [round((n / sum_i_comps * self.pop_susceptible), 2) for n in self.dS_import.values()]
        # dS = min(self.dS_import + self.dS_native, self.pop_susceptible)

        dR = self.pop_infectious[-1]

        self.record_metrics({
            'infected_new': round(dS, 2),
            'infected_new_imports': dict(zip(self.pop_imports_comps.keys(), dS_comps)),
            'infected_new_native': round(self.dS_native, 2),
            'infected_total': round(sum(self.pop_infectious), 2),
            'infected_list': [round(n, 2) for n in self.pop_infectious],
            'susceptible': self.pop_susceptible,
            'susceptible_r': self.s,
            'isolated': self.pop_isolated,
            'imported': imported.to_dict(),
            # 'imported_in': [round(n, 2) for n in sum(self.pop_imports_comps.values())],
            'R_eff_rt': self.r_rt})

        # update population values
        self.pop_infectious.appendleft(dS)
        self.pop_isolated += dR
        self.pop_susceptible -= dS

        # check for termination condition
        # rate-based?
        if self.r_rt < self.TERMINATE_BELOW_R:
            if sum(self.pop_infectious) < self.TERMINATE_BELOW_I_TOTAL:
                self.terminate = True

    # simplified version without reduction from proportion of remaining susceptible population (s)
    # NOTE: this is only valid for early phase simulation, ie. for simulation up to 10k local cumulative cases.
    # TODO: check what new cum_infected calculation here has an impact on the output
    def step_v2b_variable_beta_constant_R(self, imported: pd.Series):  # this includes the E compartment
        # insert new imports from function
        for col, dq in self.pop_imports_comps.items():
            dq.appendleft(imported[col])  # should cast back to the original dict by reference
        # we only use constant beta_func for this simulation, not updated compared to step_v2
        if self.beta_func_rt is None:
            self.beta_func_rt = self.get_I_func_rt(R_eff_rt=self.R_zero)

        # calculate expected 2ndary infections from importation subqueue and local infectious population
        self.dS_import: Dict[str, float] = dict((col, np.array(dq) @ self.beta_func_rt)
                                                for col, dq in self.pop_imports_comps.items())  # @ is dot product
        self.dS_native: float = np.array(self.pop_infectious) @ self.beta_func_rt

        # sum all new infection components to get new infectious population
        dS = sum((*self.dS_import.values(), self.dS_native))  # '*' for unpacking
        # dS = sum(i_components)  # sum for total new infectious persons
        dR = self.pop_infectious[-1]

        # calculate current local cumulative infected population (excl. new infectious (dS) because unrealistic)
        cum_infected = self.pop_isolated + sum(self.pop_infectious)
        # round current importation queue for readable output and debug
        dS_comps = [round(n, 2) for n in self.dS_import.values()]
        # record timestep for model output
        # TODO: note new additions to recording!
        self.record_metrics({
            'infected_new': round(dS, 2),
            'infected_new_imports': dict(zip(self.pop_imports_comps.keys(), dS_comps)),
            'infected_new_native': round(self.dS_native, 2),
            'infected_total': round(sum(self.pop_infectious), 2),
            'infected_list': [round(n, 2) for n in self.pop_infectious],
            'cum_infected': cum_infected,
            'cum_infected_pct': cum_infected / self.POP_TOTAL,
            'susceptible': self.pop_susceptible,
            'susceptible_r': self.s,
            'isolated': self.pop_isolated,
            'imported': imported.to_dict(),
            'R_eff_rt': self.r_rt})
        # update population values for next timestep
        self.pop_infectious.appendleft(dS)
        self.pop_isolated += dR
        self.pop_susceptible -= dS

        # check for termination condition
        if cum_infected > self.stop_cumulative:
            self.terminate = True

    # Calculate the effective infection function for that day, using the R_eff
    def get_I_func_rt(self, R_eff_rt):
        # total_weights = sum(self.FUNC_INFECTIOUS)  # TODO: depreciated because calculation done at initialisation
        return np.array(self.FUNC_INFECTIOUS) * R_eff_rt  # / total_weights

    # Record relevant statistics per day
    def record_metrics(self, in_dict: Dict[str, Any]):
        self.record_dict[self.sim_current] = in_dict

    # Process output into nicer format for later data processing
    def post_process(self):
        return pd.DataFrame.from_dict(self.record_dict, orient='index')

    def step_v1_constant_beta(self, imported):
        # use standard SIR calculation
        # self.pop_infectious[0] = self.pop_infectious[0]
        self.pop_imports.appendleft(imported)  # NOTE
        self.s = self.pop_susceptible / self.POP_TOTAL  # fraction of susceptible population
        # adjust beta/R with susceptible fraction
        self.beta_eff_rt = self.beta_null * self.s
        self.r_rt = self.R_zero * self.s
        self.record_metrics({'imported': imported,
                             'infected_list': [round(n, 2) for n in self.pop_infectious],
                             'infected_total': round(sum(self.pop_infectious), 2),
                             'infected_new': round(self.pop_infectious[0], 2),
                             'susceptible': self.pop_susceptible,
                             'isolated': self.pop_isolated,
                             'susceptible_r': self.s * 100,
                             'beta_eff_rt': self.beta_eff_rt,
                             'R_eff_rt': self.r_rt})

        # determine change in population for today
        dS = min(
            self.beta_null * self.s *
            (sum(self.pop_infectious) + sum(self.pop_imports)),
            self.pop_susceptible)
        dR = self.pop_infectious[-1]

        # update population values
        self.pop_infectious.appendleft(dS)
        self.pop_isolated += dR
        self.pop_susceptible -= dS

    def get_beta_null(self):
        return self.R_zero / self.t_inc

    ############## POST-PROCESSING OPERATIONS (FOR SENSITIVITY ANALYSIS) ############
    @staticmethod
    def post_read_handler(func):  # a decorator for file handling
        @functools.wraps(func)  # copies additional attributes (eg. original name, module, annotations)
        def wrapper(self,  # only use with class methods? a bit of black magic here...
                    col_name: str | None | slice | int,
                    # column to search, could be None if df_to_search is already sliced
                    return_name: str | Sequence[str],
                    df_to_search: pd.DataFrame | pd.Series | None = None,
                    # this normally would be the model.output attribute
                    reverse: bool = False,  # should the df_to_search be reversed
                    # return_arg: bool = False, # should the function return the value (default) or arg
                    # return_corr: int | None = None, # if arg is returned, should it be corrected (ie. for nominal date). Only works for numeric values
                    *args, **kwargs):
            """

            :param self: object which the wrapped func is attributed to (by default, the Model instance)
            :param col_name:
            :param return_name:
            :param df_to_search: optional input for pandas dataframe. By default (None) it defaults to
            :param reverse:  If the input dataframe/series should be reversed in order
            :param args: iterable inputs that will be passed into the inner function
            :param kwargs: keyword inputs that will be passed into the inner function
            :return:
            """
            if df_to_search is None:  # default: uses model.output
                if hasattr(self, 'output'):  # check if present
                    df_to_search = self.output
                else:
                    raise AttributeError(f"Attribute 'output' not found in class {self}")

            if col_name:  # if col_name is provided
                df_to_search: pd.Series = df_to_search[col_name]  # normally a pd.Series
            # might have an edge case where array_to_match is a numpy array and a col_name str index was attempted
            if reverse:
                df_to_search = df_to_search.loc[::-1]  # reverse order

            output = func(to_search=df_to_search, *args, **kwargs)

            if hasattr(output, '__len__') and isinstance(return_name, Sequence):
                if len(output) == len(return_name):
                    return dict(zip(return_name, output))
                else:
                    raise Exception(f"return_name arg not same length as func output: {return_name} vs {output}")
            else:
                return {return_name: output}

        return wrapper

    @staticmethod
    @post_read_handler
    def post_find_metric(  # self,
            to_search: pd.Series,  # only operates on Series, a limitation for now
            metric: float | int,  # what metric to measure on
            return_arg: bool = False,
            return_corr: int | None = None, ) -> int:

        # find first ordinal argument where the threshold should be inserted to maintain order
        arg = to_search.searchsorted(value=metric, side='left')
        # return either arg (might correct for date) or value
        if not return_arg:
            arg = to_search.iloc[arg]  # overwrite arg as output

        if return_corr is not None:  # if correction is required
            arg += return_corr
        return arg

    @staticmethod
    @post_read_handler
    def post_get_last(to_search: pd.Series) -> int:
        return to_search.iloc[-1]  # might need to add specifically .value?

    @staticmethod
    @post_read_handler
    def post_get_infection_trajectory_shape(to_search: pd.Series,
                                            metric: float | int,
                                            return_corr: int | None = None):
        duration = sum(to_search >= metric)
        peak_day = to_search.idxmax()
        if return_corr is not None:
            peak_day += return_corr
        peak_value = to_search.max()
        return duration, peak_day, peak_value

## COLD STORAGE

# old v1 code, depreciated
# Calculate the effective R number for that day, factoring in the number of susceptible people.
# def get_R_eff_rt(self):
#     return self.R_zero * (self.pop_susceptible / self.POP_TOTAL)
# Calculate effective beta (number of infected people per day), related to R.
# def get_beta_eff_rt(self):
#     s = self.pop_susceptible / self.POP_TOTAL
#     return self.R_zero * s * s * self.t_inc


# a failed multiprocessing experiment

# Multiprocess build that doesn't work as well
# def run_experiments_multiprocess(self, n_workers=5):
#     print(f'OFM_PE: Running MULTIPROCESS mode with {len(self.scenarios)} experiments')
#     raise NotImplementedError()
#
#     def r_callback(result):
#         print(result)
#
#     def r_error(error):
#         raise Exception(error)
#
#     # dumb sanity check
#     if self.experiments is not None:
#         raise Exception(f"OFM_PE: Experiments were run")
#
#     enum_inputs = [(e, inputs) for e, inputs in enumerate(self.scenarios)]
#
#     # create Pool instance
#     start_time = time.time()
#     pool = Pool(processes=n_workers)
#     self.experiments = pool.starmap_async(
#         func=self.handle_single_run,
#         iterable=enum_inputs,
#         callback=r_callback,
#         error_callback=r_error
#
#     )
#     pool.close() # closes entry to pool. Worker processes will exit when all tasks completed
#     pool.join() # wait for all tasks to complete before proceeding
#
#     print(f'OFM_PE: all experiments completed at {round((time.time() - start_time) / 60, 2)} mins')
#
# def handle_single_run(self, idx, model_inputs):
#     # note/document that model inputs is a tuple with (label,inputs)
#     variables = model_inputs.copy()
#     variables = self.map_input_names(
#         variables,
#         params_name_map=self.PARAMS_NAME_MAP)
#
#     if 'u_ImportsFlights' in variables:
#         import_func = self.i_flights_scaled[variables.pop('u_ImportsFlights')]
#     else: raise Exception('cannot generate \'import_func\' variable (run_experiments_multiprocess/handle_single_run')
#
#     experiment = ISIRmodel_SingleRun(
#         imports_func=import_func,
#         postprocess_to_df=True,
#         **variables
#     )
#     print(f'running experiment {idx}')
#     experiment.run_model()
#     variables.pop('u_Func_Infectious') # todo: hack!
#
#     self.results_verbose[idx] = experiment.output.assign(**variables)

## COLD STORAGE PATHOS CODE, JUNKED DUE TO UNCLEAR DOCUMENTATION AND LESS PERFORMANT THAN STOCK MULTIPROCESSING
# pool = pathos.multiprocessing.ProcessPool(ncpus=n_processes)
#
# # run simulations, print progress bar and get outputs
# self.experiments = list(
#     tqdm.tqdm(
#         pool.uimap(
#         self.handle_single_run,
#         # iterables from here
#         enums,
#         inputs,
#         [self.params_const.copy()]*len(inputs),  # constants, a bit hacky
#         # callback=r_callback,
#         # error_callback=r_error
#         chunksize=n_processes,),
#     total=len(inputs))
# )
## ^ UNDERSTANDING THE CODE, FROM INSIDE OUT (note might be incomplete):
# handle_single_run() takes scenario ID (number), scenario variables and constants, runs one simulation, and returns the ID and results per simulation
# we pass scenarios as arguments into the 'uimap()' (unordered/unblocking map)  function (different from multiprocessing's), which would pass the Nth element of the iterables (ie. enums, inputs) as one scenario into handle_single_run()
# we wrap the simulation jobs into tqdm's progress bar visual interface
# as uimap() returns an iterator, we use list() to extract the simulation results.

###### EXPERIMENT WITH VARIABLE BETA, FOR V3 BUILD

# def step_v2_variable_beta_OLD(self, imported: pd.Series | float):  # this includes the E compartment
#
#     self.pop_imports_comps.appendleft(imported)
#     self.s = self.pop_susceptible / self.POP_TOTAL
#
#     self.r_rt = self.R_zero * self.s
#     self.beta_func_rt = self.get_I_func_rt(R_eff_rt=self.r_rt)
#
#     # note we have two streams now
#     self.dS_import = np.array(self.pop_imports_comps) @ self.beta_func_rt  # @ is dot product
#     self.dS_native = np.array(self.pop_infectious) @ self.beta_func_rt
#     i_components = (self.dS_import, self.dS_native)
#     sum_i_comps = sum(i_components)
#     if sum_i_comps < self.pop_susceptible:
#         dS = sum_i_comps
#         dS_comps = (round(n, 2) for n in i_components)
#     else:
#         dS = self.pop_susceptible
#         # get ratio of contribution instead
#         dS_comps = (round((n / sum_i_comps * self.pop_susceptible), 2) for n in i_components)
#     # dS = min(self.dS_import + self.dS_native, self.pop_susceptible)
#
#     dR = self.pop_infectious[-1]
#
#     self.record_metrics({
#         'infected_new': round(dS, 2),
#         'infected_new_comps': dS_comps,
#         'infected_total': round(sum(self.pop_infectious), 2),
#         'infected_list': [round(n, 2) for n in self.pop_infectious],
#         'susceptible': self.pop_susceptible,
#         'susceptible_r': self.s * 100,
#         'isolated': self.pop_isolated,
#         'imported': imported,
#         'imported_in': [round(n, 2) for n in self.pop_imports_comps],
#         'R_eff_rt': self.r_rt})
#
#     # update population values
#     self.pop_infectious.appendleft(dS)
#     self.pop_isolated += dR
#     self.pop_susceptible -= dS
#
#     # check for termination condition
#     # rate-based?
#     if self.r_rt < self.TERMINATE_BELOW_R:
#         if sum(self.pop_infectious) < self.TERMINATE_BELOW_I_TOTAL:
#             self.terminate = True
