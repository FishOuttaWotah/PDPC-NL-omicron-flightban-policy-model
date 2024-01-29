from typing import Sequence, Callable, Dict, Any, Mapping, Iterable

import SALib

from OFM_model import ISIR_PolicyExperiments, ISIRmodel_SingleRun
# from SALib import ProblemSpec
import SALib as salib
from SALib.sample import sobol, fast_sampler
from SALib.analyze import sobol, fast
from dataclasses import dataclass
import pandas as pd
import numpy as np
import itertools

import math
import functools


class ISIR_SensitivityExperiments(ISIR_PolicyExperiments):  # copied from the v4

    PARAMS_DISCRETISE = {
        'u_Rzero': Sequence,
        'u_ImportsFlights': Sequence,
        'u_ImportsIndirect': Sequence,
        'p_FlightBans': Sequence,
        'u_Func_Infectious': Sequence,
        # otherwise 'bool' for binary (#TODO think about implementation)
    }

    DEFAULT_SAMPLERS = {
        'sobol': {'sampler': salib.sample.sobol.sample,
                  'analyser': salib.analyze.sobol.analyze}
        ,
        'fast': {'sampler': salib.sample.fast_sampler.sample,
                 'analyser': salib.analyze.fast.analyze}
    }

    def __init__(self,
                 sa_method: str | None = None,
                 sampler_args: Dict[str, Any] | None = None,
                 n_samples=None,
                 # problem_outputs: Sequence[str] | None = None,
                 run_sa: bool = True,
                 **kwargs  # for experiment inputs
                 # TODO: consider sensitivity may need their own kwargs too?
                 ):

        # TODO: mention that the inputs should be provided as even number for Sobol (not very sure)
        # TODO: include test for convergence with increasing metrics?? Read up convergence test methodology?

        # the sensitivity analysis instantiates the Experiment code in a different order, but should give the same input
        # some notable changes:
        # - import scaling is done later (after sampling)
        # - limitations of SALib means some transformation for int/bool inputs
        # - generation of scenarios is done by the sensitivity sampler, and not the normal generate_scenarios method.
        # - variables are only in tuples of length 2, this defines the bounds for the SALib sampler
        # - the model outputs are already processed within this object, instead of conducted separately in a jupyter notebook.
        # TODO: make sure that the effects of discretising is documented somewhere here
        # Discretising input factors should not affect Sobol and FAST, but will affect Morris. Source: https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/

        # For Sobol' method, a N_samples value is recommended to be a power of 2
        # TODO: describe the convoluted reasoning why discretisation is necessary, and why the SA.evaluate method was not used and essentially bypassed

        super().__init__(**kwargs)  # instantiate base model object

        ## sort variables and constants
        self.params_varied, self.params_const, self.params_null = self.sort_variables(self.params)
        self._SA_SCENARIOS_GENERATED = False  # check to see if the problem sampling is already parsed and
        # converted into scenario form
        self._SA_SAMPLED = False # check to see if the problem sampling was conducted before running? currently a dummy

        # New SA code
        self._params_v_sample_range = self.discretise_generate_sample_range(params_to_check=self.params_varied)
        ## Create SA Problem object, a SALib object that contains all relevant information for SA run.
        self.problem = self.generate_SALib_ProblemSpec(params=self._params_v_sample_range)


        # Dumb check if none of the main arguments were defined
        if run_sa:
            self._sampler_args = sampler_args
            # sa_method = sa_method.lower()  # convert to lower case
            self._n_samples = n_samples
            self._sa_method = sa_method
            self._sampler = None
            self._analyser = None
            self._samples_discretised = None

            # get the names of outputs that are used in the pre-defined post-processing ops (find in Experiment level)9
            # self._sa_outputs =  [post_op[1]['return_name'] for post_op in self.setup_postprocessing()]

            # retrieve SA method
            # self._sa_method = sa_method
            # if sa_method in self.DEFAULT_SAMPLERS:
            #     self._sampler = self.DEFAULT_SAMPLERS[sa_method]['sampler']
            #     self._analyser = self.DEFAULT_SAMPLERS[sa_method]['analyser']
            # else:
            #     raise NotImplementedError("Haven\'t implemented different SA method handling yet")

            ## generate set of variables that need to be discretised for SALib's problem definition
            # self._params_v_sample_range = self.discretise_generate_sample_range(params_to_check=self.params_varied)
            ## Create SA Problem object, a SALib object that contains all relevant information for SA run.
            # self.problem = self.generate_SALib_ProblemSpec(params=self._params_v_sample_range)

            # extract the following?
            # self.problem = self.problem.sample(func=self._sampler, N=self._n_samples,
            #                                    **self._sampler_args)  # create SA samples
            # self._samples_discretised, self.scenarios = self.postsample_rediscretise_repackage_to_scenarios(
            #     sa_problem=self.problem)  # convert SA samples to Experiment-level scenarios

            # not sure what's the output format for SALib would provide if multiple outputs are provided

    def sample_ff(self):
        # conducts a full-factorial sample

        self.problem.samples = self.handler_sampler_ff()
        self._samples_discretised, self.scenarios = self.postsample_rediscretise_repackage_to_scenarios(
            sa_problem=self.problem
        )

    def handler_sampler_ff(self):
        # generate full-factorial samples array
        # self.generate_scenarios(variables = None)  # insert ordinal variables surrogate?
        # think about repacking into labelled dict -> or is already done in a different function
        # need to generate full-factorial x array
        return np.array(list(itertools.product(*list(range(*b) for b in self.problem['bounds']))))

    def sample(self):

        # Separated the Sampling aspect of SA so that given-data approaches are accepted
        # get the names of outputs that are used in the pre-defined post-processing ops (find in Experiment level)9
        # self._sa_outputs =  [post_op[1]['return_name'] for post_op in self.setup_postprocessing()]

        # retrieve SA method
        # self._sa_method = sa_method
        if self._sa_method in self.DEFAULT_SAMPLERS:
            self._sampler = self.DEFAULT_SAMPLERS[self._sa_method]['sampler']
            self._analyser = self.DEFAULT_SAMPLERS[self._sa_method]['analyser']
        else:
            raise NotImplementedError("Haven\'t implemented different SA method handling yet")

        ## generate set of variables that need to be discretised for SALib's problem definition
        self._params_v_sample_range = self.discretise_generate_sample_range(params_to_check=self.params_varied)
        self.problem = self.generate_SALib_ProblemSpec(params=self._params_v_sample_range)

        # append new attributes to Problem
        self.problem = self.problem.sample(func=self._sampler, N=self._n_samples,
                                           **self._sampler_args)  # create SA samples
        self._samples_discretised, self.scenarios = self.postsample_rediscretise_repackage_to_scenarios(
            sa_problem=self.problem)  # convert SA samples to Experiment-level scenarios


    def ISIR_evaluate_mp(self, n_workers: int = 5):
        #
        if not self._SA_SCENARIOS_GENERATED:  # if SA samples not converted into Experiment scenarios form
            self._samples_discretised, self.scenarios = self.postsample_rediscretise_repackage_to_scenarios(
                sa_problem=self.problem)

        # Create and run multiprocessing experiments, with post-processing operations. Outputs will be saved as self.experiments
        self.run_experiments_multiprocess(n_workers=n_workers,
                                          postprocess_ops=self.setup_postprocessing())
        self.convert_experiments_to_SA_outputs()  # quick and dirty operation

    def ISIR_analyse(self, **kwargs):
        self.problem.analyze(self._analyser, **kwargs)
        return self.problem.analyze

    def convert_experiments_to_SA_outputs(self):
        # reshapes the experiments output back to SA output form
        reshape = pd.DataFrame.from_dict(dict(self.experiments), orient='index')
        self.problem['outputs'] = reshape.columns.tolist()  # adds the output labels to sa problem
        reshape_np = reshape.to_numpy()
        self.problem.set_results(reshape_np)

    def postsample_rediscretise_repackage_to_scenarios(self, sa_problem: SALib.ProblemSpec):
        """
        Convenience post-sampling, pre-simulation function that
        takes the SALib Problem's samples, converts the continuous samples back into the
        discrete forms, and packages them into the expected input datatype for the ISIR_experiments
        :param sa_problem: SALib Problem object
        :return: tuple with discretised samples (as dataframe) and in samples in scenarios form (as dict)
        """
        discretised: pd.DataFrame = self.discretise_read_bulk(problem_samples=sa_problem.samples,
                                                              problem_varnames=sa_problem['names'])
        scenarios = self.convert_SAsamples_to_scenarios(problem_samples_df=discretised)
        return discretised, scenarios

    def discretise_generate_sample_range(self,
                                         params_to_check: Dict[str, Sequence]
                                         ) -> Dict[str, Sequence]:
        """
        Generate a proxy sampling space for the continous SA sampling method, such that we can accept discrete inputs
        for specific predefined variables.
        Handles both Sequence-like and boolean discretisation.
        (Sequence)
        Converts a discrete input space (per variable) of size N to a sample space of (0, N). The SA sampling method
        will then provide continuous samples for this variable, which will be discretised back to the original
        input space.
        (Boolean)
        Converts the boolean into a range of (0,1), which will be discretised back to either True/False
        :param params_to_check: Dict containing the original variables and discrete form
        :return: a similar dict to the input dict, but the discrete variables will have been converted to a tuple of
        (0, N) for the SA problem definition and sampling purposes.
        """
        # Depreciated?
        inclusion_check = set(params_to_check.keys()) - set(self.PARAMS_DISCRETISE.keys())
        if inclusion_check:
            raise self.DISCRETISE_FORM_ERROR(inclusion_check)

        params_discretise = set(params_to_check.keys()) & set(self.PARAMS_DISCRETISE.keys())

        params_out = params_to_check.copy()  # to avoid overwriting the original
        for param in params_discretise:
            arg = params_to_check[param]
            # get length/size/shape
            discretise_form = self.PARAMS_DISCRETISE[param]
            if discretise_form == Sequence:
                params_out[param] = (0, len(arg))
            elif discretise_form == bool:
                params_out[param] = (0, 1)
            else:
                self.DISCRETISE_FORM_ERROR(discretise_form)  # call error

        return params_out

    def discretise_read_basic(self,
                              varname,
                              vals,
                              ):
        """
        Basic discretisation operation: converts 1 variable's continuous value(s) and maps it to a discrete space.
        Handles both sequence-like and boolean discretisation.
        (Sequence discretisation)
            - Continuous input values are fitted into 'bins' corresponding to the discrete value's index
            (ie. bin #1 = discrete value #1).
            - The SA sampling operation will provide values from 0 -> len(discrete space), where the values 0 <= x < 1
            form the 1st bin.
            - A correction is necessary to include the last point (len(discrete space)) as part of the last bin,
            otherwise it'll be out of bounds.

        (Boolean discretisation)
            - the SA sampling operation will provide values 0 <= x <= 1
            - we thus discriminate values <= 0.5 to be 1, and 0 otherwise
        :param varname: variable name to be discretised
        :param vals: value(s) to be discretised
        :return: mapped discretised values as defined in 'self.params_varied[varname]'
        """
        # varname for retrieval to discretise form

        # Retrieve target discretised space for variable
        ref = self.params_varied[varname]  # should be a sequence of values
        # Retrieve discretisation form of variable (sequence or bool)
        discretise_form = self.PARAMS_DISCRETISE[varname]
        if discretise_form == Sequence:

            idxs = np.trunc(vals).astype(int)  # not best, but can handle both array and scalar
            corr = idxs == len(ref)  # get intermediate item(s) are on the right edge
            idxs = idxs - corr  # correct if on right edge
            if isinstance(idxs, Iterable) and not isinstance(vals, str):
                out = [ref[idx] for idx in idxs]  # get original values
            elif isinstance(idxs, int):  # numeric value
                out = ref[idxs]
            else:
                raise NotImplementedError(f"Sequence item {varname} doesn't qualify as either an Iterable or float\n"
                                          f"{vals}")
        elif discretise_form == bool:
            # just bool if higher than a certain value
            out = vals >= 0.5
        else:
            raise self.DISCRETISE_FORM_ERROR(discretise_form)  # error handling
        return out

    def discretise_read_bulk(self,
                             problem_varnames: list[str],
                             problem_samples: np.ndarray,
                             ):
        """

        :param problem_varnames: all param variable names from the SALib problem formulation
        :param problem_samples: numpy ndarray of all samples, from the SALib problem formulation
        :return: rediscretised array (likely ndarray) of smples
        """
        # do bulk discretisation per variable, not per sample

        ## Organise all samples into pd Dataframe for easier slicing
        samples_df: pd.DataFrame = pd.DataFrame(data=problem_samples, columns=problem_varnames)
        # get params labels that need to be re-read into discretised form
        vars_to_discr = [var for var in samples_df.columns if var in self.PARAMS_DISCRETISE.keys()]
        for var in vars_to_discr:
            vals = samples_df[var]  # get column containing all values per variable
            samples_df[var] = self.discretise_read_basic(var, vals)  # do discretisation operation and write to DF
        return samples_df  # note that output is in pd.dataframe form, thus different datatype

    def convert_SAsamples_to_scenarios(self, problem_samples_df: pd.DataFrame):
        # repack the (re-discretised) output into normal scenario form (as done in Experiment level)
        self._SA_SCENARIOS_GENERATED = True  # not sure if best practice
        return problem_samples_df.to_dict(orient='index')

    @staticmethod
    def generate_SALib_ProblemSpec(params: Dict,  # includes names and bounds
                                   groups: Sequence[str] = None,
                                   outputs: Sequence[str] = None,
                                   dists: Sequence[str] = None  # distributions (uniform, lognorm, etc.)
                                   ):
        """
        Generate automated SALib ProblemSpec (aka 'problem') object
        :param params: labels of parameters and their associated parameter range. For discretised parameters,
        discretisation operations happen before this method such that the SALib's continuous sampling method works,
        by generating an artificial continuous sampling range and converting the samples to discrete array indices.
        :param groups: Categorical groups that describe mutually-exclusive areas of parameter space
        :param outputs: output metrics for model/function run. Expects singular values instead of eg. large experiment
        objects.
        :param dists: sampling distributions for params, eg. uniform, lognorm. Defaults to uniform
        :return:
        """
        names = list(params.keys())  # list() constructor needed to convert from DictArray form
        num_vars = len(names)
        # create numpy 2d array of size 'num_vars' rows x 2 columns
        bounds = np.array(list(params.values()))  # list() constructor here again

        return salib.ProblemSpec(
            {
                'names': names,
                'num_vars': num_vars,
                'bounds': bounds,
                'groups': groups,
                'outputs': outputs,
                'dists': dists
            }
        )

    @staticmethod
    def DISCRETISE_FORM_ERROR(discretise_form):
        # Shows a pre-emptive error if (custom) discretisation instructions are not provided, because the multiprocessing
        # environment doesn't handle errors very well
        return Exception(
            f"Discretise form {discretise_form} not recognised/present, please check the PARAMS_DISCRETISE "
            f"attribute and add handling to the \'discretise_generate_sample_range()\' and/or "
            f"\'discretise_read()\' method in Sensitivity Experiments class")
