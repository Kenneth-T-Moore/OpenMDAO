from __future__ import print_function

import os
import shutil
import tempfile
import warnings

import unittest
import numpy as np
import math

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from scipy.sparse import load_npz

from openmdao.api import Problem, IndepVarComp, ExecComp, DirectSolver,\
    ExplicitComponent, LinearRunOnce, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.coloring import get_simul_meta, _solves_info
from openmdao.test_suite.tot_jac_builder import TotJacBuilder
import openmdao.test_suite


# check that pyoptsparse is installed
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class RunOnceCounter(LinearRunOnce):
    def __init__(self, *args, **kwargs):
        self._solve_count = 0
        super(RunOnceCounter, self).__init__(*args, **kwargs)

    def _iter_execute(self):
        super(RunOnceCounter, self)._iter_execute()
        self._solve_count += 1

# note: size must be an even number
SIZE = 10

def run_opt(driver_class, mode, color_info=None, sparsity=None, **options):

    p = Problem()

    p.model.linear_solver = RunOnceCounter()
    indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

    # the following were randomly generated using np.random.random(10)*2-1 to randomly
    # disperse them within a unit circle centered at the origin.
    indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                      0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
    indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                     -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
    indeps.add_output('r', .7)

    p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

    p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r',
                                            g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)
    p.model.add_subsystem('theta_con', ExecComp('g=arctan(y/x) - theta',
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE),
                                                theta=thetas))
    p.model.add_subsystem('delta_theta_con', ExecComp('g = arctan(y/x)[::2]-arctan(y/x)[1::2]',
                                                      g=np.ones(SIZE//2), x=np.ones(SIZE),
                                                      y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)

    p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

    p.model.connect('r', ('circle.r', 'r_con.r'))
    p.model.connect('x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

    p.model.connect('x', 'l_conx.x')

    p.model.connect('y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

    p.driver = driver_class()
    p.driver.options.update(options)

    p.model.add_design_var('x')
    p.model.add_design_var('y')
    p.model.add_design_var('r', lower=.5, upper=10)

    # nonlinear constraints
    p.model.add_constraint('r_con.g', equals=0)

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[0::2]  # all odd indices
    p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=ODD_IND)
    p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

    # this constrains x[0] to be 1 (see definition of l_conx)
    p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

    # linear constraint
    p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

    p.model.add_objective('circle.area', ref=-1)

    # # setup coloring
    if color_info is not None:
        p.driver.set_simul_deriv_color(color_info)
    elif sparsity is not None:
        p.driver.set_total_jac_sparsity(sparsity)

    p.setup(mode=mode)
    p.run_driver()

    return p


class SimulColoringPyoptSparseTestCase(unittest.TestCase):

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt_fwd(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)

        color_info = {"fwd": [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ]],
        "sparsity": {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }}
        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info, optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_simul_coloring_snopt_auto(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)

    def test_simul_coloring_pyoptsparse_slsqp_fwd(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = {"fwd": [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ]],
            "sparsity": {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }}

        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info, optimizer='SLSQP', print_results=False)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    def test_dynamic_simul_coloring_pyoptsparse_slsqp_auto(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        p_color = run_opt(pyOptSparseDriver, 'auto', optimizer='SLSQP', print_results=False,
                          dynamic_simul_derivs=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'auto', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)


class SimulColoringPyoptSparseRevTestCase(unittest.TestCase):
    """Reverse coloring tests for pyoptsparse."""

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)

        color_info = {"rev": [[
            [4, 5, 6, 7, 8, 9, 10],   # uncolored rows
            [1, 18, 19, 20, 21],   # color 1
            [0, 17, 13, 14, 15, 16],   # color 2
            [2, 11],   # color 3
            [3, 12]   # color 4
            ],
            [
            [20],   # row 0
            [0, 10, 20],   # row 1
            [1, 11, 20],   # row 2
            [2, 12, 20],   # row 3
            None,   # row 4
            None,   # row 5
            None,   # row 6
            None,   # row 7
            None,   # row 8
            None,   # row 9
            None,   # row 10
            [0],   # row 11
            [0, 10],   # row 12
            [2, 12],   # row 13
            [4, 14],   # row 14
            [6, 16],   # row 15
            [8, 18],   # row 16
            [0, 1, 10, 11],   # row 17
            [2, 3, 12, 13],   # row 18
            [4, 5, 14, 15],   # row 19
            [6, 7, 16, 17],   # row 20
            [8, 9, 18, 19]   # row 21
            ]],
        "sparsity": {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            }
        }}
        p_color = run_opt(pyOptSparseDriver, 'rev', color_info, optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1) / 11)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_rev_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - bidirectional coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1 - 22 * 3) / 11)

    def test_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = {"rev": [[
           [1, 4, 5, 6, 7, 8, 9, 10],
           [3, 17],
           [0, 11, 13, 14, 15, 16],
           [2, 12, 18, 19, 20, 21]
        ],
        [
           [20],
           None,
           [1, 11, 20],
           [2, 12, 20],
           None,
           None,
           None,
           None,
           None,
           None,
           None,
           [0],
           [0, 10],
           [2, 12],
           [4, 14],
           [6, 16],
           [8, 18],
           [0, 1, 10, 11],
           [2, 3, 12, 13],
           [4, 5, 14, 15],
           [6, 7, 16, 17],
           [8, 9, 18, 19]
        ]],
        "sparsity": {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            }
        }}

        p_color = run_opt(pyOptSparseDriver, 'rev', color_info, optimizer='SLSQP', print_results=False)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1) / 11)

    def test_dynamic_rev_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        p_color = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False,
                          dynamic_simul_derivs=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'rev', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1 - 22 * 3) / 11)


class SimulColoringScipyTestCase(unittest.TestCase):

    def setUp(self):
        self.color_info = {"fwd": [[
               [20],   # uncolored columns
               [0, 2, 4, 6, 8],   # color 1
               [1, 3, 5, 7, 9],   # color 2
               [10, 12, 14, 16, 18],   # color 3
               [11, 13, 15, 17, 19]   # color 4
            ],
            [
               [1, 11, 16, 21],   # column 0
               [2, 16],   # column 1
               [3, 12, 17],   # column 2
               [4, 17],   # column 3
               [5, 13, 18],   # column 4
               [6, 18],   # column 5
               [7, 14, 19],   # column 6
               [8, 19],   # column 7
               [9, 15, 20],   # column 8
               [10, 20],   # column 9
               [1, 11, 16],   # column 10
               [2, 16],   # column 11
               [3, 12, 17],   # column 12
               [4, 17],   # column 13
               [5, 13, 18],   # column 14
               [6, 18],   # column 15
               [7, 14, 19],   # column 16
               [8, 19],   # column 17
               [9, 15, 20],   # column 18
               [10, 20],   # column 19
               None   # column 20
            ]],
            "sparsity": None
        }

    def test_simul_coloring_fwd(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'fwd', self.color_info, optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)
        
        # check for proper handling if someone calls compute_totals on Problem with different set or different order 
        # of desvars/responses than were used to define the coloring.  Behavior should be that coloring is turned off 
        # and a warning is issued.
        with warnings.catch_warnings(record=True) as w:
            p_color.compute_totals(of=['delta_theta_con.g', 'circle.area', 'r_con.g', 'theta_con.g', 'l_conx.g'], 
                                   wrt=['x', 'y', 'r'])
        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message), "compute_totals called using a different list of design vars and/or responses than those used to define coloring, so coloring will be turned off.\ncoloring design vars: ['indeps.x', 'indeps.y', 'indeps.r'], current design vars: ['indeps.x', 'indeps.y', 'indeps.r']\ncoloring responses: ['circle.area', 'r_con.g', 'theta_con.g', 'delta_theta_con.g', 'l_conx.g'], current responses: ['delta_theta_con.g', 'circle.area', 'r_con.g', 'theta_con.g', 'l_conx.g'].")

    def test_bad_mode(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(ScipyOptimizeDriver, 'rev', self.color_info, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does forward solves but mode has been set to 'rev'")

    def test_dynamic_simul_coloring_auto(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'auto', optimizer='SLSQP', disp=False, dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)

    def test_simul_coloring_example(self):

        from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizeDriver
        import numpy as np

        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                          -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

        p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r',
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

        thetas = np.linspace(0, np.pi/4, SIZE)
        p.model.add_subsystem('theta_con', ExecComp('g=arctan(y/x) - theta',
                                                    g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE),
                                                    theta=thetas))
        p.model.add_subsystem('delta_theta_con', ExecComp('g = arctan(y/x)[::2]-arctan(y/x)[1::2]',
                                                          g=np.ones(SIZE//2), x=np.ones(SIZE),
                                                          y=np.ones(SIZE)))

        thetas = np.linspace(0, np.pi/4, SIZE)

        p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

        p.model.connect('x', 'l_conx.x')

        p.model.connect('y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

        p.driver = ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        p.model.add_constraint('r_con.g', equals=0)

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[0::2]  # all odd indices
        p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=ODD_IND)
        p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

        # linear constraint
        p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

        p.model.add_objective('circle.area', ref=-1)

        # setup coloring
        color_info = {"fwd": [[
           [20],   # uncolored column list
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19],   # color 4
        ],
        [
           [1, 11, 16, 21],   # column 0
           [2, 16],   # column 1
           [3, 12, 17],   # column 2
           [4, 17],   # column 3
           [5, 13, 18],   # column 4
           [6, 18],   # column 5
           [7, 14, 19],   # column 6
           [8, 19],   # column 7
           [9, 15, 20],   # column 8
           [10, 20],   # column 9
           [1, 11, 16],   # column 10
           [2, 16],   # column 11
           [3, 12, 17],   # column 12
           [4, 17],   # column 13
           [5, 13, 18],   # column 14
           [6, 18],   # column 15
           [7, 14, 19],   # column 16
           [8, 19],   # column 17
           [9, 15, 20],   # column 18
           [10, 20],   # column 19
           None,   # column 20
        ]],
        "sparsity": None
        }

        p.driver.set_simul_deriv_color(color_info)

        p.setup(mode='fwd')
        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)


class SimulColoringRevScipyTestCase(unittest.TestCase):
    """Rev mode coloring tests."""

    def setUp(self):
        self.color_info = {"rev": [[
               [4, 5, 6, 7, 8, 9, 10],   # uncolored rows
               [2, 21],   # color 1
               [3, 16],   # color 2
               [1, 17, 18, 19, 20],   # color 3
               [0, 11, 12, 13, 14, 15]   # color 4
            ],
            [
               [20],   # row 0
               [0, 10, 20],   # row 1
               [1, 11, 20],   # row 2
               [2, 12, 20],   # row 3
               None,   # row 4
               None,   # row 5
               None,   # row 6
               None,   # row 7
               None,   # row 8
               None,   # row 9
               None,   # row 10
               [0, 10],   # row 11
               [2, 12],   # row 12
               [4, 14],   # row 13
               [6, 16],   # row 14
               [8, 18],   # row 15
               [0, 1, 10, 11],   # row 16
               [2, 3, 12, 13],   # row 17
               [4, 5, 14, 15],   # row 18
               [6, 7, 16, 17],   # row 19
               [8, 9, 18, 19],   # row 20
               [0]   # row 21
            ]],
            "sparsity": None}

    def test_simul_coloring(self):

        color_info = self.color_info

        p = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'rev', color_info, optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - (total_solves - 1) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1) / 11)
    
    def test_bad_mode(self):
        with self.assertRaises(Exception) as context:
            p_color = run_opt(ScipyOptimizeDriver, 'fwd', self.color_info, optimizer='SLSQP', disp=False)
        self.assertEqual(str(context.exception),
                         "Simultaneous coloring does reverse solves but mode has been set to 'fwd'")

    def test_dynamic_simul_coloring(self):

        p_color = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False, dynamic_simul_derivs=True)
        p = run_opt(ScipyOptimizeDriver, 'rev', optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - bidirectional coloring saves 11 solves per driver iter  (11 vs 22)
        # - initial solve for linear constraints takes 1 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 22 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 1 for the uncolored case and 22 * 3 + 1 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 1) / 22,
                         (p_color.model.linear_solver._solve_count - 1 - 22 * 3) / 11)


class SparsityTestCase(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='SparsityTestCase-')
        os.chdir(self.tempdir)

        self.sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            }
        }

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_sparsity_snopt(self):
        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SNOPT', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)

    def test_sparsity_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SLSQP', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SLSQP', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)


class BidirectionalTestCase(unittest.TestCase):
    def test_eisenstat(self):
        for n in range(6, 20, 2):
            builder = TotJacBuilder.eisenstat(n)
            builder.color('auto', stream=None)
            tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(builder.coloring)
            if tot_colors == n // 2 + 3:
                raise unittest.SkipTest("Current bicoloring algorithm requires n/2 + 3 solves, so skipping for now.")
            self.assertLessEqual(tot_colors, n // 2 + 2, 
                                 "Eisenstat's example of size %d required %d colors but shouldn't "
                                 "need more than %d." % (n, tot_colors, n // 2 + 2))

            builder_fwd = TotJacBuilder.eisenstat(n)
            builder_fwd.color('fwd', stream=None)
            tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(builder_fwd.coloring)
            # The columns of Eisenstat's example are pairwise nonorthogonal, so fwd coloring
            # should require n colors.
            self.assertEqual(n, tot_colors,
                             "Eisenstat's example of size %d was not constructed properly. "
                             "fwd coloring required only %d colors but should have required "
                             "%d" % (n, tot_colors, n))

    def test_arrowhead(self):
        for n in range(5, 50, 5):
            builder = TotJacBuilder(n, n)
            builder.add_row(0)
            builder.add_col(0)
            builder.add_block_diag([(1,1)] * (n-1), 1, 1)
            builder.color('auto', stream=None)
            tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(builder.coloring)
            self.assertEqual(tot_colors, 3)

    def test_can_715(self):
        # this test is just to show the superiority of bicoloring vs. single coloring in 
        # either direction.  Bicoloring gives only 21 colors in this case vs. 105 for either
        # fwd or rev.
        matdir = os.path.join(os.path.dirname(openmdao.test_suite.__file__), 'matrices')

        # uses matrix can_715 from the sparse matrix collection website
        mat = load_npz(os.path.join(matdir, 'can_715.npz')).toarray()
        mat = np.asarray(mat, dtype=bool)
        coloring = get_simul_meta(None, 'auto', include_sparsity=False, setup=False,
                                  run_model=False, bool_jac=mat,
                                  stream=None)

        tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(coloring)

        self.assertEqual(tot_colors, 21)

        # verify that unidirectional colorings are much worse (105 vs 21 for bidirectional)
        coloring = get_simul_meta(None, 'fwd', include_sparsity=False, setup=False,
                                  run_model=False, bool_jac=mat,
                                  stream=None)

        tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(coloring)

        self.assertEqual(tot_colors, 105)

        coloring = get_simul_meta(None, 'rev', include_sparsity=False, setup=False,
                                  run_model=False, bool_jac=mat,
                                  stream=None)

        tot_size, tot_colors, fwd_solves, rev_solves, pct = _solves_info(coloring)

        self.assertEqual(tot_colors, 105)


if __name__ == '__main__':
    unittest.main()
