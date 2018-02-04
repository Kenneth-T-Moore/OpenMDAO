"""
Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).

This driver is based on the EGO-Like Framework (EGOLF) for the simultaneous
design-mission-allocation optimization problem. Handles
mixed-integer/discrete type design variables in a computationally efficient
manner and finds a near-global solution to the above MINLP/MDNLP problem.

Developed by Satadru Roy
Purdue University, West Lafayette, IN
July 2016
Implemented in OpenMDAO, Aug 2016, Kenneth T. Moore
"""
from __future__ import print_function

from collections import OrderedDict
from time import time

from six import iteritems
from six.moves import range

import numpy as np

from openmdao.core.driver import Driver
from openmdao.drivers.amiego_util.branch_and_bound import Branch_and_Bound
from openmdao.drivers.amiego_util.kriging import AMIEGOKrigingSurrogate
from openmdao.drivers.scipy_optimizer import ScipyOptimizeDriver
from openmdao.recorders.recording_iteration_stack import Recording


class AMIEGO_driver(Driver):
    """
    Driver for AMIEGO (A Mixed Integer Efficient Global Optimization).

    This driver is based on the EGO-Like Framework (EGOLF) for the
    simultaneous design-mission-allocation optimization problem. It handles
    mixed-integer/discrete type design variables in a computationally
    efficient manner and finds a near-global solution to the above
    MINLP/MDNLP problem. The continuous optimization is handled by the
    optimizer slotted in self.cont_opt, which is ScipyOptimizer by
    default.

    AMIEGO_driver supports the following:
        integer_design_vars

    Options
    -------
    options['disp'] : bool
        Display flag. Toggle printing of AMIEGO output to stdout.
    options['ei_tol_rel'] :  0.001
        Relative tolerance on the expected improvement.
    options['max_infill_points'] : 10
        Maximum number of additional points per design variable.
        points.
    options['r_penalty'] : 1.0
        Constraint penalty applied to objective for surrogate model.

    Attributes
    ----------
    c_dvs : list
        Cache of continuous design variable names.
    con_sampling : dict(list)
        Optional constraint values from user-supplied pre-optimized initial samples.
    cont_opt : <Driver>
        Slot for continuous optimizer.
    i_idx : dict
        Cache of local sizes for each design variable.
    i_size : int
        Number of integer design variables.
    minlp : <Branch_and_Bound>
        Slot for Branch and Bound subdriver.
    n_train : int
        Number of training points for surrogate.
    obj_sampling : dict(list)
        Optional objective values from user-supplied pre-optimized initial samples.
    sampling : dict(list)
        Initial sampling points.
    sampling_eflag : dict(list)
        Optional success flag from user-supplied pre-optimized initial samples.
    """

    def __init__(self):
        """
        Initialize the AMIEGO driver.
        """
        super(AMIEGO_driver, self).__init__()

        # What we support
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = False
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['linear_constraints'] = False
        self.supports['gradients'] = True
        self.supports['mixed_integer'] = True
        self.supports['two_sided_constraints'] = True

        # Options
        opt = self.options
        opt.declare('disp', True,
                    desc='Set to False to prevent printing of iteration messages.')
        opt.declare('ei_tol_rel', 0.001, lower=0.0,
                    desc='Relative tolerance on the expected improvement.')
        opt.declare('max_infill_points', 10, lower=1,
                    desc='Maximum number of additional points per design variable.')
        opt.declare('r_penalty', 2.0,
                    desc='Constraint penalty applied to objective.')

        # The default continuous optimizer. User can slot a different one
        self.cont_opt = ScipyOptimizeDriver()
        self.cont_opt.options['optimizer'] = 'SLSQP'

        # The default MINLP optimizer
        self.minlp = Branch_and_Bound()

        # Default surrogate. User can slot a modified one, but it essentially
        # has to provide what Kriging provides.
        self.surrogate = AMIEGOKrigingSurrogate

        self.c_dvs = []
        self.i_size = 0
        self.i_idx = OrderedDict()
        self.n_train = 0

        # Initial Sampling of integer design points
        # TODO: Somehow slot an object that generates this (LHC for example)
        self.sampling = {}

        # User can pre-load these to skip initial continuous optimization
        # in favor of pre-optimized points.
        self.obj_sampling = None
        self.con_sampling = None
        self.sampling_eflag = None

    def _setup_driver(self, problem, assemble_var_info=True):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        assemble_var_info : bool
            If True, then gather all the designvars, objectives, and constraints from the model.
        """
        super(AMIEGO_driver, self)._setup_driver(problem, assemble_var_info)

        # Need to clean out anything in the continuous optimizer first.
        cont_opt = self.cont_opt
        cont_opt._cons = OrderedDict()
        cont_opt._objs = OrderedDict()
        cont_opt._designvars = OrderedDict()

        if 'disp' in cont_opt.options:
            cont_opt.options['disp'] = self.options['disp']

        minlp = self.minlp
        minlp._setup_driver(problem)
        minlp.options['disp'] = self.options['disp']

        # Identify and size our design variables.
        prom2abs = problem.model._var_allprocs_prom2abs_list['output']
        sampling_abs_names = {}
        i_dvs = []
        for name, data in iteritems(self.sampling):
            abs_name = prom2abs[name][0]
            sampling_abs_names[abs_name] = data
            i_dvs.append(abs_name)
        self.sampling = sampling_abs_names
        self.n_train = len(self.sampling[abs_name])

        # If we attached a set of pre optimized cases, we need to convert the keys from promoted
        # to absolute names and keep them.
        obj_sampling_abs_names = {}
        if self.obj_sampling is not None:
            for name, data in iteritems(self.obj_sampling):
                abs_name = prom2abs[name][0]
                obj_sampling_abs_names[abs_name] = data
            self.obj_sampling = obj_sampling_abs_names

        con_sampling_abs_names = {}
        if self.con_sampling is not None:
            for name, data in iteritems(self.con_sampling):
                abs_name = prom2abs[name][0]
                con_sampling_abs_names[abs_name] = data
            self.con_sampling = con_sampling_abs_names

        for name, val in iteritems(self.get_design_var_values()):
            if name in i_dvs:
                i_size = len(val)
                self.i_idx[name] = i_size
            else:
                self.c_dvs.append(name)

        j = 0
        for var, idx in iteritems(self.i_idx):
            idx_tuple = (j, j + idx)
            j += idx
            self.i_idx[var] = idx_tuple
        self.i_size = j

        # Lower and Upper bounds for integer desvars
        self.xI_lb = np.empty((self.i_size, ))
        self.xI_ub = np.empty((self.i_size, ))
        dv_dict = self._designvars
        for var, idx in iteritems(self.i_idx):
            i, j = idx
            self.xI_lb[i:j] = dv_dict[var]['lower']
            self.xI_ub[i:j] = dv_dict[var]['upper']

        # Continuous Optimization only gets continuous desvars
        for name in self.c_dvs:
            cont_opt._designvars[name] = self._designvars[name]

        # MINLP Optimization only gets discrete desvars
        for name in self.i_idx:
            minlp._designvars[name] = self._designvars[name]

        # Both MINLP and Continuous see the objective
        cont_opt._objs = self._objs
        minlp._objs = self._objs

        # Continuous optimizer sees all constraints.
        for name, con in iteritems(self._cons):
            cont_opt._cons[name] = con

        # Finish setting up the subdrivers.
        cont_opt._setup_driver(problem, assemble_var_info=False)

    def run(self):
        """
        Execute the AMIEGO driver.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem
        n_i = self.i_size
        ei_tol_rel = self.options['ei_tol_rel']
        disp = self.options['disp']
        r_pen = self.options['r_penalty']
        cont_opt = self.cont_opt
        minlp = self.minlp
        xI_lb = self.xI_lb
        xI_ub = self.xI_ub

        self.iter_count = 0

        # ----------------------------------------------------------------------
        # Step 1: Generate a set of initial integer points
        # TODO: Use Latin Hypercube Sampling to generate the initial points
        # User supplied (in future use LHS). Provide num_xI+2 starting points
        # ----------------------------------------------------------------------

        max_pt_lim = self.options['max_infill_points'] * n_i

        # Since we need to add a new point every iteration, make these lists
        # for speed.
        x_i = []
        obj = []
        cons = {}
        best_int_design = {}
        best_cont_design = {}
        n_train = self.n_train
        for con in self._cons:
            cons[con] = []

        # Start with pre-optimized samples.
        if self.obj_sampling:
            pre_opt = True
            c_start = c_end = n_train

            for i_train in range(n_train):

                xx_i = np.empty((self.i_size, ))
                for var, idx in iteritems(self.i_idx):
                    i, j = idx
                    xx_i[i:j] = self.sampling[var][i_train]

                    # Save the best design too (see below)
                    if i_train == 0:
                        best_int_design[var] = self.sampling[var][i_train]

                x_i.append(xx_i)

            current_objs = self.get_objective_values()
            obj_name = list(current_objs.keys())[0]
            obj = self.obj_sampling[obj_name]
            cons = self.con_sampling

            # Satadru's suggestion is that we start with the first point as
            # the best obj.
            lowest = 0

            # However, if we know which cases were infeasible, then we can find the lowest
            # objective.
            if self.sampling_eflag is not None:
                for j, val in enumerate(obj):
                    if self.sampling_eflag[j] == 1 and (val < obj[lowest]):
                        lowest = j

            best_obj = obj[lowest].copy()

        # Prepare to optimize the initial sampling points
        else:
            best_obj = 1000.0
            pre_opt = False
            c_start = 0
            c_end = n_train

            for i_train in range(n_train):

                xx_i = np.empty((self.i_size, ))
                for name, idx in iteritems(self.i_idx):
                    i, j = idx
                    xx_i[i:j] = self.sampling[name][i_train, :]

                x_i.append(xx_i)

        # Need to cache the continuous desvars so that we start each new
        # optimization back at the original initial condition.
        xc_cache = {}
        desvars = cont_opt.get_design_var_values()
        for var, val in iteritems(desvars):
            xc_cache[var] = val.copy()

        ei_max = 1.0
        term = 0.0
        terminate = False
        tot_newpt_added = 0
        tot_pt_prev = 0
        ec2 = 0
        i_con_opt = 0

        # AMIEGO main loop
        while not terminate:
            self.iter_count += 1

            # ------------------------------------------------------------------
            # Step 2: Perform the optimization w.r.t continuous design
            # variables
            # ------------------------------------------------------------------

            if disp:
                print(22 * "=" + "ContinuousOptimization-Start" + 37 * "=")
                t0 = time()

            # In initial iteration, we only optimize points if we don't have samples of the cons
            # and objs.
            # In subsequent iterations, we just optimize the new candidate point.
            for i_run in range(c_start, c_end):

                if disp:
                    print('Optimizing for the given integer/discrete type design variables.',
                          x_i[i_run])

                # Set Integer design variables
                for var, idx in iteritems(self.i_idx):
                    i, j = idx
                    self.set_design_var(var, x_i[i_run][i:j])

                # Restore initial condition for all continuous vars.
                for var, val in iteritems(xc_cache):
                    cont_opt.set_design_var(var, val)

                # If we are doing any prescreening, we need to attach the
                # list of integer desvars to the cont_opt
                cont_opt.trip = x_i[i_run]

                # Optimize continuous variables
                with Recording('AMIEGO_cont_opt', i_con_opt, self) as rec:
                    self.pre_cont_opt_hook()
                    fail = cont_opt.run()
                    rec.abs = 0.0
                    rec.rel = 0.0

                i_con_opt += 1
                eflag_conopt = not fail
                if disp:
                    print("Exit Flag:", eflag_conopt)

                # Get objectives and constraints.
                current_objs = self.get_objective_values()
                obj_name = list(current_objs.keys())[0]
                current_obj = current_objs[obj_name].copy()
                obj.append(current_obj)
                for name, value in iteritems(self.get_constraint_values()):
                    cons[name].append(value.copy())

                # If best solution, save it
                if eflag_conopt and current_obj < best_obj:
                    best_obj = current_obj
                    # Save integer and continuous DV
                    desvars = self.get_design_var_values()

                    for name in self.i_idx:
                        val = desvars[name]
                        best_int_design[name] = val.copy()

                    for name in self.c_dvs:
                        best_cont_design[name] = desvars[name].copy()

            if disp:
                print('Elapsed Time:', time() - t0)
                print(22 * "=" + "ContinuousOptimization-End" + 39 * "=")
                t0 = time()

            # ------------------------------------------------------------------
            # Step 3: Build the surrogate models
            # ------------------------------------------------------------------
            n = len(x_i)
            P = np.zeros((n, 1))

            # TODO: Scale back the objective to the original Value
            # As Kriging objective is normalized separately
            #scale_fac_conopt = np.array([1.0e3])
            #obj_surr = obj[:] * scale_fac_conopt

            num_vio = np.zeros((n, 1), dtype=np.int)

            # Normalize the objective data
            X_mean = np.mean(x_i, axis=0)
            X_std = np.std(x_i, axis=0)
            X_std[X_std == 0.] = 1.

            Y_mean = np.mean(obj_surr, axis=0)
            Y_std = np.std(obj_surr, axis=0)
            Y_std[Y_std == 0.] = 1.

            X = (x_i - X_mean) / X_std
            Y = (obj_surr - Y_mean) / Y_std

            for name, val in iteritems(cons):
                val = np.array(val)

                # Note, Branch and Bound defines constraints to be violated
                # when positive, so we need to transform from OpenMDAO's
                # freeform.
                meta = self._cons[name]
                val_u = val - meta['upper']
                val_l = meta['lower'] - val

                # Normalize the constraint data
                g_mean = np.mean(val, axis=0)
                g_std = np.std(val, axis=0)
                g_std[g_std == 0.] = 1.0
                g_norm = (val - g_mean) / g_std
                g_vio_ub = val_u / g_std
                g_vio_lb = val_l / g_std

                # Make the problem appear unconstrained to Amiego
                M = val.shape[1]
                for ii in range(n):
                    for mm in range(M):

                        if val_u[ii][mm] > 0:
                            P[ii] += g_vio_ub[ii][mm]**2
                            num_vio[ii] += 1

                        elif val_l[ii][mm] > 0:
                            P[ii] += g_vio_lb[ii][mm]**2
                            num_vio[ii] += 1

            for ii in range(n):
                if num_vio[ii] > 0:
                    Y[ii] += (r_pen * P[ii] / num_vio[ii])

            obj_surrogate = self.surrogate()
            obj_surrogate.use_snopt = True

            obj_surrogate.X, obj_surrogate.X_mean, obj_surrogate.X_std = X, X_mean, X_std
            obj_surrogate.Y, obj_surrogate.Y_mean, obj_surrogate.Y_std = Y, Y_mean, Y_std
            obj_surrogate.train(X, Y, KPLS=True, norm_data=True)

            best_obj_norm = (best_obj - obj_surrogate.Y_mean) / obj_surrogate.Y_std

            if disp:
                print("\nSurrogate building of the objective is complete...")
                print('Elapsed Time:', time() - t0)

            # ------------------------------------------------------------------
            # Step 4: Maximize the expected improvement function to obtain an
            # integer infill point.
            # ------------------------------------------------------------------

            if disp:
                print("AMIEGO-Iter: %d" % self.iter_count)
                print("The best solution so far: yopt = %0.4f" % best_obj)

            tot_newpt_added += c_end - c_start
            if pre_opt or tot_newpt_added != tot_pt_prev:

                minlp.obj_surrogate = obj_surrogate
                minlp.xI_lb = xI_lb
                minlp.xI_ub = xI_ub

                if disp:
                    t0 = time()
                    print(22 * "=" + "MINLPBB-Start" + 37 * "=")
                minlp.run()
                if disp:
                    print('Elapsed Time:', time() - t0)
                    print(22 * "=" + "MINLPBB-End" + 39 * "=")

                eflag_MINLPBB = minlp.eflag_MINLPBB
                x0I = minlp.xopt
                ei_min = minlp.fopt

                if disp:
                    print("Eflag = ", eflag_MINLPBB)

                if eflag_MINLPBB >= 1:

                    ei_max = -ei_min
                    tot_pt_prev = tot_newpt_added

                    if disp:
                        print("New xI = ", x0I)
                        print("EI_min = ", ei_min)

                    # Prevent the correlation matrix being close to singular. No
                    # point allowed within the pescribed hypersphere of any
                    # existing point
                    rad = 0.5
                    for ii in range(len(x_i)):
                        dist = np.sum((x_i[ii] - x0I)**2)**0.5
                        if dist <= rad:
                            if disp:
                                print("Point already exists!")
                            ec2 = 1
                            break

                    x_i.append(x0I)

                else:
                    ec2 = 1
            else:
                ec2 = 1

            # ------------------------------------------------------------------
            # Step 5: Check for termination
            # ------------------------------------------------------------------

            c_start = c_end
            c_end += 1

            term = np.abs(ei_tol_rel * best_obj_norm)
            if (not pre_opt and ei_max <= term) or ec2 == 1 or tot_newpt_added >= max_pt_lim:
                terminate = True
                if disp:
                    if ei_max <= term:
                        print("No Further improvement expected! Terminating algorithm.")
                    elif ec2 == 1:
                        print("No new point found that improves the surrogate. "
                              "Terminating algorithm.")
                    elif tot_newpt_added >= max_pt_lim:
                        print("Maximum allowed sampling limit reached! Terminating algorithm.")

            pre_opt = False

        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state
        for name, val in iteritems(best_int_design):
            self.set_design_var(name, val)
        for name, val in iteritems(best_cont_design):
            self.set_design_var(name, val)

        with Recording('AMIEGO_cont_opt', i_con_opt, self) as rec:
            problem.model._solve_nonlinear()
            rec.abs = 0.0
            rec.rel = 0.0

        if disp:
            print("\n===================Result Summary====================")
            print("The best objective: %0.4f" % best_obj)
            print("Total number of continuous minimization: %d" % len(x_i))
            print("Best Integer designs: ", best_int_design)
            print("Corresponding continuous designs: ", best_cont_design)
            print("=====================================================")

        return True

    def pre_cont_opt_hook(self):
        """
        Perform calculations prior to continuous optimization.

        Override this to perform any pre-continuous-optimization operations.
        """
        pass
