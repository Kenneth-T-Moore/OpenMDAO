""" Unit tests for the AMIEGO driver."""

import unittest

import numpy as np

from openmdao.api import IndepVarComp, Group, Problem, ExecComp, pyOptSparseDriver
from openmdao.devtools.testutil import assert_rel_error
from openmdao.drivers.amiego_driver import AMIEGO_driver
from openmdao.test_suite.components.branin import Branin
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss, ThreeBarTrussVector
from openmdao.utils.general_utils import set_pyoptsparse_opt

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class TestAMIEGOdriver(unittest.TestCase):

    def setUp(self):
        if OPT is None:
            raise unittest.SkipTest("pyoptsparse is not installed")

        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest("SNOPT is needed to run this test")

    def test_simple_branin_opt(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        #model.approx_totals(method='fd')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = AMIEGO_driver()
        #prob.driver.options['disp'] = False

        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'
        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        prob.driver.sampling = {'p2.xI' : np.array([[-5.0], [0.0], [5.0]])}

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.49398, 1e-5)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_three_bar_truss(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('xc_a1', IndepVarComp('area1', 5.0), promotes=['*'])
        model.add_subsystem('xc_a2', IndepVarComp('area2', 5.0), promotes=['*'])
        model.add_subsystem('xc_a3', IndepVarComp('area3', 5.0), promotes=['*'])
        model.add_subsystem('xi_m1', IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        #prob.driver.cont_opt.options['tol'] = 1e-12
        #prob.driver.options['disp'] = False
        #model.approx_totals()
        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        model.add_design_var('area1', lower=0.0005, upper=10.0)
        model.add_design_var('area2', lower=0.0005, upper=10.0)
        model.add_design_var('area3', lower=0.0005, upper=10.0)
        model.add_design_var('mat1', lower=1, upper=4)
        model.add_design_var('mat2', lower=1, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('mass')
        model.add_constraint('stress', upper=1.0)

        npt = 5
        samples = np.array([[4, 2, 3],
                            [1, 3, 1],
                            [3, 1, 2],
                            [3, 4, 2],
                            [1, 1, 4]])

        prob.driver.sampling = {'mat1' : samples[:, 0].reshape((npt, 1)),
                                'mat2' : samples[:, 1].reshape((npt, 1)),
                                'mat3' : samples[:, 2].reshape((npt, 1))}

        prob.setup(check=False)

        prob.run_driver()

        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat1'], 3, 1e-5)
        assert_rel_error(self, prob['mat2'], 3, 1e-5)
        #Material 3 can be anything

    def test_three_bar_truss_preopt(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('xc_a', IndepVarComp('area', np.array([5.0, 5.0, 5.0])), promotes=['*'])
        model.add_subsystem('xi_m', IndepVarComp('mat', np.array([1, 1, 1])), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTrussVector(), promotes=['*'])

        prob.driver = AMIEGO_driver()
        #prob.driver.cont_opt.options['tol'] = 1e-12
        #prob.driver.options['disp'] = False
        prob.driver.cont_opt = pyOptSparseDriver()
        prob.driver.cont_opt.options['optimizer'] = 'SNOPT'

        prob.driver.minlp.options['trace_iter'] = 3
        prob.driver.minlp.options['trace_iter_max'] = 5

        model.add_design_var('area', lower=0.0005, upper=10.0)
        model.add_design_var('mat', lower=1, upper=4)
        model.add_objective('mass')
        model.add_constraint('stress', upper=1.0)

        npt = 5
        samples = [np.array([ 4.,  2.,  3.]),
                   np.array([ 1.,  3.,  1.]),
                   np.array([ 3.,  1.,  2.]),
                   np.array([ 3.,  4.,  2.]),
                   np.array([ 1.,  1.,  4.])]

        obj_samples = [np.array([ 20.42278739]),
                       np.array([ 7.87886979]),
                       np.array([ 11.40011902]),
                       np.array([ 13.86862925]),
                       np.array([ 11.65356799])]

        con_samples = [np.array([ 1.        ,  1.00000028,  0.66329027]),
                       np.array([ 1.        ,  1.        ,  0.73587506]),
                       np.array([ 0.49384804,  1.        ,  0.09501274]),
                       np.array([ 1.00000004,  1.        ,  0.91702808]),
                       np.array([ 0.85256063,  1.00000001,  0.4147135])]

        prob.driver.sampling = {'mat' : samples}
        prob.driver.obj_sampling = {'mass' : obj_samples}
        prob.driver.con_sampling = {'stress' : con_samples}
        prob.driver.int_con = ['stress']

        prob.setup(check=False)

        prob.run_driver()

        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat'][0], 3, 1e-5)
        assert_rel_error(self, prob['mat'][1], 3, 1e-5)
        # Material 3 can be anything

    #def test_simple_greiwank_opt(self):

        #unittest.SkipTest('TODO: Make this a bit more robust.')

        #prob = Problem()
        #model = prob.model = Group()

        #model.add('p1', IndepVarComp('xC', np.array([0.0, 0.0, 0.0])), promotes=['*'])
        #model.add('p2', IndepVarComp('xI', np.array([0, 0, 0])), promotes=['*'])
        #model.add('comp', Greiwank(num_cont=3, num_int=3), promotes=['*'])

        #prob.driver = AMIEGO_driver()
        #prob.driver.cont_opt.options['tol'] = 1e-12
        #prob.driver.options['disp'] = True
        #model.deriv_options['type'] = 'fd'

        #prob.driver.add_desvar('xI', lower=-5, upper=5)
        #prob.driver.add_desvar('xC', lower=-5.0, upper=5.0)

        #prob.driver.add_objective('f')
        #samples = np.array([[1.0, 0.25, 0.75],
                            #[0.0, 0.75, 0.0],
                            #[0.75, 0.0, 0.25],
                            #[0.75, 1.0, 0.5],
                            #[0.25, 0.5, 1.0]])
        ## prob.driver.sampling = {'xI' : np.array([[0.0], [.76], [1.0]])}
        #prob.driver.sampling = {'xI' : samples}

        #prob.setup(check=False)
        #prob.run()

        ## Optimal solution
        #assert_rel_error(self, prob['f'], 0.0, 1e-5)
        #assert_rel_error(self, prob['xI'], 0.0, 1e-5)


if __name__ == "__main__":
    unittest.main()
