"""
A Case class.
"""

import re
import itertools


class Case(object):
    """
    Case wraps the data from a single iteration of a recording to make it more easily accessible.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded.
    counter : int
        The global execution counter.
    iteration_coordinate : str
        The string that holds the full unique identifier for this iteration.
    timestamp : float
        Time of execution of the case.
    success : str
        Success flag for the case.
    msg : str
        Message associated with the case.
    prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    inputs : PromotedToAbsoluteMap
        Map of inputs to values recorded.
    outputs : PromotedToAbsoluteMap
        Map of outputs to values recorded.
    residuals : PromotedToAbsoluteMap
        Map of outputs to residuals recorded.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                 prom2abs, abs2prom, meta, inputs, outputs, residuals=None):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the Case was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        inputs : array
            Inputs to read in from the recording file.
        outputs : array
            Outputs to read in from the recording file.
        residuals : array, optional
            Residuals to read in from the recording file.

        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg
        self.inputs = None
        self.outputs = None
        self.residuals = None
        self.meta = meta
        self.prom2abs = prom2abs
        self.abs2prom = abs2prom

        if inputs is not None and inputs.dtype.names:
            self.inputs = PromotedToAbsoluteMap(inputs[0], prom2abs, abs2prom, output=False)
        if outputs is not None and outputs.dtype.names:
            self.outputs = PromotedToAbsoluteMap(outputs[0], prom2abs, abs2prom)
        if residuals is not None and residuals.dtype.names:
            self.residuals = PromotedToAbsoluteMap(residuals[0], prom2abs, abs2prom)

    def get_desvars(self):
        """
        Get the design variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('desvar')

    def get_objectives(self):
        """
        Get the objective variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('objective')

    def get_constraints(self):
        """
        Get the constraint variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('constraint')

    def get_responses(self):
        """
        Get the response variables and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        return self._get_variables_of_type('response')

    def _get_variables_of_type(self, var_type):
        """
        Get the variables of a given type and their values.

        Parameters
        ----------
        var_type : str
            String indicating which value for 'type' should be accepted for a variable
            to be included in the returned map.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of variables to their values.
        """
        if self.outputs is None:
            return PromotedToAbsoluteMap({}, self.prom2abs, self.abs2prom)

        ret_vars = {}
        for var in self.outputs.absolute_names():
            if var_type in self.meta[var]['type']:
                ret_vars[var] = self.outputs[var]

        return PromotedToAbsoluteMap(ret_vars, self.prom2abs, self.abs2prom)


class DriverCase(Case):
    """
    Wrap data from a single iteration of a Driver recording to make it more easily accessible.

    Attributes
    ----------
    _var_settings : dict
        Dictionary mapping absolute variable names to variable settings.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success,
                 msg, inputs, outputs, prom2abs, abs2prom, meta, var_settings):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the DriverCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for the desired iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        inputs : array
            Driver inputs to read in from the recording file.
        outputs : array
            Driver outputs to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        var_settings : dict
            Dictionary mapping absolute variable names to variable settings.
        """
        super(DriverCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg, prom2abs,
                                         abs2prom, meta, inputs, outputs)
        self._var_settings = var_settings

    def scale(self):
        """
        Scale the outputs array using _var_settings.
        """
        for name in self.outputs.absolute_names():
            if name in self._var_settings:
                # physical to scaled
                if self._var_settings[name]['adder'] is not None:
                    self.outputs[name] += self._var_settings[name]['adder']
                if self._var_settings[name]['scaler'] is not None:
                    self.outputs[name] *= self._var_settings[name]['scaler']


class SystemCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, inputs,
                 outputs, residuals, prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for the desired iteration.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        inputs : array
            System inputs to read in from the recording file.
        outputs : array
            System outputs to read in from the recording file.
        residuals : array
            System residuals to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(SystemCase, self).__init__(filename, counter, iteration_coordinate,
                                         timestamp, success, msg, prom2abs,
                                         abs2prom, meta, inputs, outputs,
                                         residuals=residuals)


class SolverCase(Case):
    """
    Wraps data from a single iteration of a System recording to make it more accessible.

    Attributes
    ----------
    abs_err : array
        Solver absolute error that has been read in from the recording file.
    rel_err : array
        Solver relative error that has been read in from the recording file.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg,
                 abs_err, rel_err, inputs, outputs, residuals, prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The iteration coordinate, in a specific format.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        abs_err : array
            Solver absolute error to read in from the recording file.
        rel_err : array
            Solver relative error to read in from the recording file.
        inputs : array
            Solver inputs to read in from the recording file.
        outputs : array
            Solver outputs to read in from the recording file.
        residuals : array
            Solver residuals to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(SolverCase, self).__init__(filename, counter, iteration_coordinate, timestamp,
                                         success, msg, prom2abs, abs2prom, meta,
                                         inputs, outputs, residuals=residuals)

        self.abs_err = abs_err
        self.rel_err = rel_err


class ProblemCase(Case):
    """
    Wraps data from a single case of a Problem recording to make it more accessible.
    """

    def __init__(self, filename, counter, case_name, timestamp, success, msg,
                 outputs, prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the SystemCase was constructed.
        counter : int
            The global execution counter.
        case_name : str
            Name used to identify this Problem case.
        timestamp : float
            Time of execution of the case
        success : str
            Success flag for the case
        msg : str
            Message associated with the case
        outputs : array
            Solver outputs to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        super(ProblemCase, self).__init__(filename, counter, case_name, timestamp,
                                          success, msg, prom2abs, abs2prom, meta,
                                          None, outputs)


class PromotedToAbsoluteMap(dict):
    """
    A dictionary that enables accessing values via absolute or promoted variable names.

    Attributes
    ----------
    _values : array or dict
        Array or dict of values accessible via absolute variable name.
    _keys : array
        Absolute variable names that map to the values in the _values array.
    _prom2abs : {'input': dict, 'output': dict}
        Dictionary mapping promoted names to absolute names.
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    _is_output : bool
        True if this should map using output variable names, False for input variable names.
    """

    def __init__(self, values, prom2abs, abs2prom, output=True):
        """
        Initialize.

        Parameters
        ----------
        values : array or dict
            Numpy structured array or dictionary of values.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        output : bool
            True if this should map using output variable names, False for input variable names.
        """
        super(PromotedToAbsoluteMap, self).__init__()

        self._is_output = output

        self._prom2abs = prom2abs
        self._abs2prom = abs2prom

        if output:
            prom2abs = self._prom2abs['output']
            abs2prom = self._abs2prom['output']
        else:
            prom2abs = self._prom2abs['input']
            abs2prom = self._abs2prom['input']

        if isinstance(values, dict):
            # dict of values, keyed on either absolute or promoted names
            self._values = {}
            for key in values.keys():
                if isinstance(key, tuple) or ',' in key:
                    # derivative keys can be either (of, wrt) or 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    for abs_key in abs_keys:
                        self._values[abs_key] = values[key]
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                else:
                    if key in abs2prom:
                        # key is absolute name
                        self._values[key] = values[key]
                        prom_key = abs2prom[key]
                        super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                    elif key in prom2abs:
                        # key is promoted name
                        for abs_key in prom2abs[key]:
                            self._values[abs_key] = values[key]
                        super(PromotedToAbsoluteMap, self).__setitem__(key, values[key])
            self._keys = self._values.keys()
        else:
            # numpy structured array, which will always use absolute names
            self._values = values
            self._keys = values.dtype.names
            for key in self._keys:
                if key in abs2prom:
                    prom_key = abs2prom[key]
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])
                elif ',' in key:
                    # derivative keys will be a string in the form of 'of,wrt'
                    abs_keys, prom_key = self._deriv_keys(key)
                    super(PromotedToAbsoluteMap, self).__setitem__(prom_key, values[key])

    def _deriv_keys(self, key):
        """
        Get the absolute and promoted name versions of the provided derivative key.

        Parameters
        ----------
        key : tuple or string
            derivative key as either (of, wrt) or 'of,wrt'.

        Returns
        -------
        list of tuples:
            list of (of, wrt) mapping the provided key to absolute names.
        tuple :
            (of, wrt) mapping the provided key to promoted names.
        """
        prom2abs = self._prom2abs['output']
        abs2prom = self._abs2prom['output']

        # derivative could be tuple or string, using absolute or promoted names
        if isinstance(key, tuple):
            of, wrt = key
        else:
            of, wrt = re.sub('[( )]', '', key).split(',')

        # if promoted, will map to all connected absolute names
        abs_of = [of] if of in abs2prom else prom2abs[of]
        abs_wrt = [wrt] if wrt in abs2prom else prom2abs[wrt]
        abs_keys = ['%s,%s' % (o, w) for o, w in itertools.product(abs_of, abs_wrt)]

        prom_of = of if of in prom2abs else abs2prom[of]
        prom_wrt = wrt if wrt in prom2abs else abs2prom[wrt]
        prom_key = (prom_of, prom_wrt)

        return abs_keys, prom_key

    def __getitem__(self, key):
        """
        Use the variable name to get the corresponding value.

        Parameters
        ----------
        key : string
            Absolute or promoted variable name.

        Returns
        -------
        array :
            An array entry value that corresponds to the given variable name.
        """
        if key in self._keys:
            # absolute name
            return self._values[key]

        elif key in self:
            # promoted name
            return super(PromotedToAbsoluteMap, self).__getitem__(key)

        elif isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)
            return super(PromotedToAbsoluteMap, self).__getitem__(prom_key)

        raise KeyError(key)

    def __setitem__(self, key, value):
        """
        Set the value for the given key, which may use absolute or promoted names.

        Parameters
        ----------
        key : string
            Absolute or promoted variable name.
        value : any
            value for variable
        """
        if self._is_output:
            prom2abs = self._prom2abs['output']
            abs2prom = self._abs2prom['output']
        else:
            prom2abs = self._prom2abs['input']
            abs2prom = self._abs2prom['input']

        if isinstance(key, tuple) or ',' in key:
            # derivative keys can be either (of, wrt) or 'of,wrt'
            abs_keys, prom_key = self._deriv_keys(key)

            for abs_key in abs_keys:
                self._values[abs_key] = value

            super(PromotedToAbsoluteMap, self).__setitem__(prom_key, value)

        elif key in self._keys:
            # absolute name
            self._values[key] = value
            super(PromotedToAbsoluteMap, self).__setitem__(abs2prom[key], value)
        else:
            # promoted name, propagate to all connected absolute names
            for abs_key in prom2abs[key]:
                if abs_key in self._keys:
                    self._values[abs_key] = value
            super(PromotedToAbsoluteMap, self).__setitem__(key, value)

    def absolute_names(self):
        """
        Yield absolute names for variables contained in this dictionary.

        Similar to keys() but with absolute variable names instead of promoted names.

        Yields
        ------
        str
            absolute names for variables contained in this dictionary.
        """
        for key in self._keys:
            if ',' in key:
                # return derivative keys as tuples instead of strings
                of, wrt = re.sub('[( )]', '', key).split(',')
                yield (of, wrt)
            else:
                yield key


class DriverDerivativesCase(object):
    """
    Wrap data from a derivative calculation in a Driver recording to make it more accessible.

    Attributes
    ----------
    filename : str
        The file from which the case was loaded.
    counter : int
        The global execution counter.
    iteration_coordinate : str
        The string that holds the full unique identifier for this iteration.
    timestamp : float
        Time of execution of the case.
    success : str
        Success flag for the case.
    msg : str
        Message associated with the case.
    prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
    abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    meta : dict
        Dictionary mapping absolute variable names to variable metadata.
    totals : PromotedToAbsoluteMap
        Map of inputs to values recorded.
    """

    def __init__(self, filename, counter, iteration_coordinate, timestamp, success, msg, totals,
                 prom2abs, abs2prom, meta):
        """
        Initialize.

        Parameters
        ----------
        filename : str
            The filename from which the Case was constructed.
        counter : int
            The global execution counter.
        iteration_coordinate : str
            The string that holds the full unique identifier for this iteration.
        timestamp : float
            Time of execution of the case.
        success : str
            Success flag for the case.
        msg : str
            Message associated with the case.
        totals : array
            Derivatives to read in from the recording file.
        prom2abs : {'input': dict, 'output': dict}
            Dictionary mapping promoted names to absolute names.
        abs2prom : {'input': dict, 'output': dict}
            Dictionary mapping absolute names to promoted names.
        meta : dict
            Dictionary mapping absolute variable names to variable metadata.
        """
        self.filename = filename
        self.counter = counter
        self.iteration_coordinate = iteration_coordinate

        self.timestamp = timestamp
        self.success = success
        self.msg = msg
        self.meta = meta
        self.prom2abs = prom2abs
        self.abs2prom = abs2prom

        if totals is not None and totals.dtype.names:
            self.totals = PromotedToAbsoluteMap(totals[0], prom2abs, abs2prom, output=True)

    def get_derivatives(self):
        """
        Get the derivatives and their values.

        Returns
        -------
        PromotedToAbsoluteMap
            Map of derivatives to their values.
        """
        ret_vars = {}
        for key in self.totals.absolute_names():
            ret_vars[key] = self.totals[key]
        return PromotedToAbsoluteMap(ret_vars, self.prom2abs, self.abs2prom, output=True)
