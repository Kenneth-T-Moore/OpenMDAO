"""
Helper function to find all the `cite` attributes throughout a model.
"""
from __future__ import print_function
from collections import OrderedDict
import inspect
import sys

from openmdao.utils.logger_utils import get_logger


def _check_cite(obj, citations):
    """
    Grab the cite attribute, if it exists.

    Parameters
    ----------
    obj : object
        the instance to check for citations on
    citations : dict
        the dictionary to add a citation to, if found
    """
    if inspect.isclass(obj):
        if obj.cite:
            citations[obj] = obj.cite
    if obj.cite:
        klass = obj.__class__
        # return klass, cite
        citations[klass] = obj.cite


def find_citations(prob, out_stream='stdout'):
    """
    Compiles a list of citations from all classes in the problem.

    Parameters
    ----------
    prob : <Problem>
        The Problem instance to be searched
    out_stream : 'stdout', 'stderr' or file-like
            Where to send human readable output. Default is 'stdout'.
            Set to None to suppress.

    Returns
    -------
    dict
        dict of citations keyed by class
    """
    # dict keyed by the class so we don't report multiple citations
    # for the same class showing up in multiple instances
    citations = OrderedDict()
    _check_cite(prob, citations)
    _check_cite(prob.driver, citations)

    _check_cite(prob._vector_class, citations)

    # recurse down the model
    for subsys in prob.model.system_iter(include_self=True, recurse=True):
        _check_cite(subsys, citations)
        if subsys.nonlinear_solver is not None:
            _check_cite(subsys.nonlinear_solver, citations)
        if subsys.linear_solver is not None:
            _check_cite(subsys.linear_solver, citations)

    if out_stream:
        logger = get_logger('list_inputs', out_stream=out_stream)
        for klass, cite in citations.items():
            # print("Class: {}".format(klass), file=out_stream)
            logger.info("Class: {}".format(klass))
            lines = cite.split('\n')
            for line in lines:
                # print("    {}".format(line), file=out_stream)
                logger.info("    {}".format(line))

    return citations
