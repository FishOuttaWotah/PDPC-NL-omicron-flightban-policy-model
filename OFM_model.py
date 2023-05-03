from typing import Union, Sequence, Dict, Any, Callable
import collections, itertools, random
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time

"""
This Python file contains the description of the Omicron Flights Importation Model 
"""


#
class ISIR_PolicyExperiments:
    # handles the importation parts and conducts bulk runs with the propagator

    # code for inferred imports by certain date
    # the apply float method?
    # instantiate and run propagator
    # get outputs?
    # should this already be the experiment handling level already?
    # yeah have a very simple experiment handling (policy for flight ban)

    # just a serial run, nothing too fancy for now
    # then focus on plotting and illustration

    class INPUTS:
        UNCERTAINTY = 'u_'
        POLICY = 'p_'
        CONSTANT = 'c_'

    # define params to be renamed from Experiment level to Model input level
    ARG_NAME_MAP = {
        'u_Rzero': 'R_zero',
        'u_TIncub': "t_incubation",
        'u_ImportsFunc': 'imports_flights',
        'p_FlightBans': 'flightban_on',
        'u_Func_Infectious': 'func_infectious',
        'u_ImportsIndirect': 'imports_indirect',
        'u_ImportsOther': 'imports_other',
        'c_model_engine': 'engine',
        'c_SimTime': 'sim_time',
        'c_PopTotal': 'pop_total',
    }

    # define params to be excluded from single model run
    PARAMS_EXCLUDE_FROM_MODEL = (
        'self',
        'c_nominal_ref_date',
        'u_ImportsFunc',
        'c_import_scaling_mode'
    )

    def __init__(self,
                 u_ImportsFunc,
                 u_TIncub: Sequence[int] = (3,),
                 u_Rzero: Sequence[float] = (1.3,),  # was 3.22 baseline
                 u_ImportsFlights: Sequence[float] = (10.,),
                 u_Func_Infectious: Sequence[Sequence[float]] | None = None,
                 u_ImportsIndirect: Sequence[float] = (10.,),  # v4 (indirect entry)
                 u_ImportsOther: Sequence[float] = (10.,),  # v4 (miscellaneous addition)
                 p_FlightBans: Sequence[int] | None = None,
                 c_SimTime: int = 120,  # days
                 c_PopTotal: int = 17480000,  # 17.48 mil
                 c_nominal_ref_date=16,  # 16 corresponds with 25th Nov
                 c_import_scaling_mode=2,  # v3
                 c_model_engine='step_v2',  # v3
                 ):
        self.params = locals().copy()  # get the model inputs, this WILL be modified for passing into the single model run.
        self.PARAMS_RAW = locals().copy()  # make a non-modified version of the model inputs
        self.debug_unmapped_args = []

        # Save experiment inputs
        self.u_imports_func = u_ImportsFunc
        self.u_TIncub = u_TIncub  # incubation time
        self.u_Rzero = u_Rzero
        self.p_flightbans = self.get_flightban_days(p_FlightBans, c_nominal_ref_date)  # TODO think about more intuitive naming
        self.p_flightbans_map = dict([(d, p_FlightBans[i]) for i, d in enumerate(self.p_flightbans)])
        self.params['p_FlightBans'] = self.p_flightbans
        self.c_sim_time = c_SimTime
        self.c_pop_total = c_PopTotal
        self.c_nominal_ref_date = c_nominal_ref_date
        self.u_nominal_imports = u_ImportsFlights

        ## Handle import scaling mode
        # decides how the import function should be (up)scaled
        # 1 = scale the cumulative sum up to nominal ref date to ImportsFlights
        # 2 = (new) scale only the imports on the nominal ref date to Importflights
        self.import_scaling_mode = c_import_scaling_mode
        if c_import_scaling_mode == 1:
            self.c_import_ref_dates = slice(None, self.c_nominal_ref_date - 1)
        elif c_import_scaling_mode == 2:
            self.c_import_ref_dates = self.c_nominal_ref_date
        else:
            raise NotImplementedError(
                f"OFM_experiments:(c_import_scaling_mode of {c_import_scaling_mode} is invalid for model.")

        # Conduct scaling for import functions
        self.imports_inflated, self.imports_nominal_cumul, self.inflation_factor = self.inflate_importation_function(
            imports=u_ImportsFunc,
            nominal_target=self.u_nominal_imports,
            date_slice=self.c_import_ref_dates)

        # self.imports_inflated = self.imports_inflated.to_dict()  # hack from Pd Series to dict

        # identify which params are varied or constant
        self.params_varied, self.params_const = self.sort_variables(self.params)
        self.scenarios = self.generate_scenarios(
            variables=self.params_varied,
            constants=self.params_const,
            arg_name_map=self.ARG_NAME_MAP
        )

        # items for saving results
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
        input_params = self.filter_excluded_params(input_params.copy(), ## copy() for non-destructive operations
                                                   compare=self.PARAMS_EXCLUDE_FROM_MODEL)

        p_varied = {}
        p_constant = {}
        p_error = {}
        for k, v in input_params.items():
            # if there is attribute len, and not present in exclude
            try:  # for catching non-valid len attributes
                # catch criteria:
                # 1. if has valid length attribute (with a non-null value)
                # 2. is not a string (so list, tuple, etc.)
                if hasattr(v, "__len__") and isinstance(len(v), int) and not isinstance(v, str):
                    if len(v) > 1:  # if more than 1, thus a variable
                        p_varied[k] = v
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
                             f'PARAMS_EXCLUDE_FROM_MODEL set')

        return p_varied, p_constant

    @staticmethod
    def filter_excluded_params(params: Dict, compare: Sequence[str] = PARAMS_EXCLUDE_FROM_MODEL):
        return dict((k, v) for k, v in params.items() if k not in compare)

    def generate_scenarios(self, variables: Dict, constants: Dict,
                           arg_name_map: Dict):

        # code does the following (read from back):
        # 1. creates all possible combinations of (cartesian product) of variables
        # 2. creates a dict with one combo of variables and constants. This serves as the input into the model
        # 3. creates a list with all dicts in 2).
        # TODO: expansion point: if a dict should be given as the output or a list
        # scenarios = [dict(zip(variables.keys(), inputs), **constants) for inputs in itertools.product(*variables.values())]
        scenarios = []
        for vars_set in itertools.product(*variables.values()):
            input_set = dict(zip(variables.keys(), vars_set), **constants)
            scenarios.append(input_set)

        return scenarios

    def run_experiments(self):
        # create a bunch of experiments and run, no sophisticated things
        # iterate through all scenarios
        print(f"OFM_PE: Running {len(self.scenarios)} experiments")
        start_time = time.time()
        for scen_idx, scen_args in enumerate(self.scenarios):
            variables = scen_args.copy()  # make a copy for labelling
            variables = self.map_input_names(variables, arg_name_map=self.ARG_NAME_MAP)

            if 'u_ImportsFlights' in scen_args:  # if it is a variable
                imports = self.imports_inflated[
                    variables.pop('u_ImportsFlights')]  # get which import function to take, and
                # ^ eliminate the entry "u_ImportsFlights", because it shouldn't be given as an input.
            else:
                raise Exception('cannot generate \'imports\' variable in run_experiments()')
            # m_inputs = dict((self.ARG_NAME_MAP[k], v) for k, v in scen_inputs.items())  # rename to
            experiment = ISIRmodel_SingleRun(
                imports_flights=imports,
                postprocess_to_df=True,
                **variables
                # **m_inputs,
                # **m_constants,
            )
            experiment.run_model()
            #  add experiment entries to output
            scen_args.pop('u_Func_Infectious')  # TODO hack!
            self.results_verbose[scen_idx] = experiment.output.assign(**scen_args)
            self.results_metadata[scen_idx] = scen_args
        self.results_postprocess = self.postprocess_to_df(self.results_verbose,
                                                          map_to_invert=self.ARG_NAME_MAP)  # create unified dataframe
        print(f"OFM_PE: all experiments completed in {round((time.time() - start_time) / 60, 1)} mins")

    def map_input_names(self, input_set: Dict, arg_name_map: Dict):
        # identify which args don't have a mapping
        self.debug_unmapped_args = set(input_set.keys()) - set(arg_name_map)
        # if self.debug_unmapped_args:
        #     print(f'Note: experiment-level args {self.debug_unmapped_args} do not have a mapping.')
        mapped = dict((arg_name_map.get(k, k), v) for k, v in input_set.items())
        return mapped

    @staticmethod
    def postprocess_to_df(results_verbose, map_to_invert: Dict | None = None):
        # concat results into dataframe and set some added indices
        results_df = pd.concat(results_verbose)
        if map_to_invert is not None:
            reverse_map = {v: k for k, v in map_to_invert.items()}
            results_df = results_df.rename(columns=reverse_map)
        return results_df

    # inflate the importation shape for the nominal values provided
    @staticmethod
    def inflate_importation_function(imports, nominal_target, date_slice):
        # get cumulative imports from start of omicron detection till date of reference
        nominal_cumulative = imports.loc[date_slice].sum()
        # else:
        #     nominal_cumulative = imports.loc[:ref_date-1].sum() # note pandas end on ref_date
        inflation_factor = np.array(nominal_target) / nominal_cumulative
        imports_inflated: pd.DataFrame = pd.concat([imports * factor for factor in inflation_factor], axis=1, keys=nominal_target)
        return imports_inflated, nominal_cumulative, list(inflation_factor)


    @staticmethod
    def get_flightban_days(flightban_days, ref_date):
        # convert the (relative) flightban days
        # TODO: note that if flightban_days has None, will raise an Error
        return tuple(ref_date + delta for delta in flightban_days)


class ISIRmodel_SingleRun:
    # Describing items that need to be created at the start, for future use
    def __init__(self,
                 *,  # ensures function is called with parameter names
                 R_zero,
                 t_incubation: int,
                 func_infectious: Sequence[float],  ## NEW ITEM
                 pop_total: int,
                 sim_time: int,
                 imports_flights: pd.Series | None = None,
                 flightban_on: int | None = None,
                 postprocess_to_df: bool = False,  # TODO note
                 engine: str = 'step_standard',
                 imports_indirect: pd.Series | None = None,  # added v4
                 imports_other: pd.Series | None = None  # added v4
                 ):
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
        self.pop_susceptible = pop_total
        self.s = self.pop_susceptible / self.POP_TOTAL
        self.sim_end = sim_time
        self.flight_ban: int | None | bool = flightban_on
        self.postprocess_to_df: bool = postprocess_to_df
        self.FUNC_INFECTIOUS = func_infectious  # TODO: NEW
        self.beta_func_rt = None  # This is the model state!  TODO: New

        # v2: early terminate conditions
        self.TERMINATE_BELOW_R = 1.  #
        self.TERMINATE_BELOW_I_TOTAL = 1.  # total infectious
        self.terminate = False  # TODO: NEW, for early termination

        ## Handling for case importation (v4)
        if imports_flights is not None:
            if flightban_on is not None:
                imports_flights = imports_flights.loc[:flightban_on-1] # we assume no more imports on the flightban day itself
                # Note: modification of input!

        isets_dict = {
            'flights': imports_flights,
            'nonflights': imports_indirect,
            'others': imports_other,
        }
        import_sets = dict((label, iset) for label, iset in isets_dict.items()
                       if iset is not None)

        # conduct superimposed import function
        if len(import_sets) >= 1:
            self.imports_all = pd.concat(import_sets, axis=1)
            self.imports_func = self.imports_all.sum(axis=1)
        else:
            raise Exception("ISIR_Single: no importation function was provided")


        # # Handle case importation
        # if imports_func is None:  # if no imports applied
        #     self.imports_func = {}  # make empty dict
        #     self.flight_ban = False
        # else:  # TODO note potential edge cases, incomplete
        #     self.imports_func = imports_func  # dict
        #     if imports_func:  # if not empty
        #         if isinstance(flightban_on, int):
        #             # note catch case for negative values
        #             days = self.imports_func.keys()  # get keys
        #             days_allowed = [d for d in days if d < flightban_on]  #
        #             self.imports_func = dict([(d, imports_func[d]) for d in days_allowed])
        #         elif flightban_on is None:
        #             self.flight_ban = False  #
        #             pass
        #         else:
        #             raise ValueError(f"Parameter {flightban_on} must be a positive int or None")
        #             pass
        #         pass
        #     else:  # if empty
        #
        #         pass

        # Instantiate model states
        self.sim_current = 0
        if engine == 'step_v2':
            self.pop_infectious = collections.deque([0] * len(func_infectious),
                                                    maxlen=len(func_infectious))  # representation of
            self.pop_imports = collections.deque([0] * len(func_infectious),
                                                 maxlen=len(func_infectious))
            self.dS_import = None  # TODO: debug item
            self.dS_native = None


        elif engine == 'step_standard':  ## TODO: old representation
            self.pop_infectious = collections.deque([0] * t_incubation,
                                                    maxlen=t_incubation)  # representation of infectious population as a python Deque
            self.pop_imports = collections.deque([0] * t_incubation,
                                                 maxlen=t_incubation)
        # ^ another deque to keep population of imports separate
        self.pop_isolated = 0
        self.r_rt = 0  # rt = real-time (ie. for that day)
        self.beta_eff_rt = 0
        self.beta_null = self.get_beta_null()  # uses u__R_eff and t_inc

        # Instantiate model recording
        self.record_dict = {}
        self.output: pd.DataFrame | None = None  # clean up model outputs (in dict form) for later processing

    def run_model(self):
        """
        Runs model with the defined simulation engine for the defined number of steps. Handles importation input per step, and conducts a post-processed function at the end.
        """
        while self.sim_current <= self.sim_end:
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

    def step_standard(self, imported):
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

    def step_v2(self, imported):  # this includes the E compartment
        self.pop_imports.appendleft(imported)
        self.s = self.pop_susceptible / self.POP_TOTAL

        self.r_rt = self.R_zero * self.s
        self.beta_func_rt = self.get_I_func_rt(R_eff_rt=self.r_rt)
        self.record_metrics({'imported': imported,
                             'imported_in': [round(n, 2) for n in self.pop_imports],
                             'infected_list': [round(n, 2) for n in self.pop_infectious],
                             'infected_total': round(sum(self.pop_infectious), 2),
                             'infected_new': round(self.pop_infectious[0], 2),
                             'susceptible': self.pop_susceptible,
                             'isolated': self.pop_isolated,
                             'susceptible_r': self.s * 100,
                             'R_eff_rt': self.r_rt})

        # we have two streams now
        self.dS_import = np.array(self.pop_imports) @ self.beta_func_rt
        self.dS_native = np.array(self.pop_infectious) @ self.beta_func_rt
        dS = min(self.dS_import + self.dS_native, self.pop_susceptible)
        dR = self.pop_infectious[-1]

        # update population values
        self.pop_infectious.appendleft(dS)
        self.pop_isolated += dR
        self.pop_susceptible -= dS

        # check for termination condition
        # rate-based?
        if self.r_rt < self.TERMINATE_BELOW_R:
            if sum(self.pop_infectious) < self.TERMINATE_BELOW_I_TOTAL:
                self.terminate = True

    # Calculate the effective R number for that day, factoring in the number of susceptible people.
    def get_R_eff_rt(self):
        return self.R_zero * (self.pop_susceptible / self.POP_TOTAL)

    # Calculate the effective infection function for that day, using the R_eff
    def get_I_func_rt(self, R_eff_rt):
        total_weights = sum(self.FUNC_INFECTIOUS)
        return np.array(self.FUNC_INFECTIOUS) * R_eff_rt / total_weights

    # Calculate effective beta (number of infected people per day), related to R.
    def get_beta_eff_rt(self):
        s = self.pop_susceptible / self.POP_TOTAL
        return self.R_zero * s * s * self.t_inc

    def get_beta_null(self):
        return self.R_zero / self.t_inc

    # Record relevant statistics per day
    def record_metrics(self, in_dict: Dict[str, Any]):
        self.record_dict[self.sim_current] = in_dict

    # Process output into nicer format for later data processing
    def post_process(self):
        return pd.DataFrame.from_dict(self.record_dict, orient='index')

## COLD STORAGE
# a failed multiprocessing experiment

# Multiprocess build that doesn't work as well
# def run_experiments_multiprocess(self, n_processes=5):
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
#     pool = Pool(processes=n_processes)
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
#         arg_name_map=self.ARG_NAME_MAP)
#
#     if 'u_ImportsFlights' in variables:
#         imports = self.imports_inflated[variables.pop('u_ImportsFlights')]
#     else: raise Exception('cannot generate \'imports\' variable (run_experiments_multiprocess/handle_single_run')
#
#     experiment = ISIRmodel_SingleRun(
#         imports_func=imports,
#         postprocess_to_df=True,
#         **variables
#     )
#     print(f'running experiment {idx}')
#     experiment.run_model()
#     variables.pop('u_Func_Infectious') # todo: hack!
#
#     self.results_verbose[idx] = experiment.output.assign(**variables)
