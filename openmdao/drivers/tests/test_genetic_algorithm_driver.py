""" Unit tests for the SimpleGADriver Driver."""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.drivers.genetic_algorithm_driver import SimpleGADriver
from openmdao.test_suite.components.branin import Branin
from openmdao.test_suite.components.three_bar_truss import ThreeBarTruss
from openmdao.utils.assert_utils import assert_rel_error


class TestSimpleGA(unittest.TestCase):

    def test_simple_test_func(self):

        class MyComp(ExplicitComponent):

            def setup(self):
                self.add_input('x', np.zeros((2, )))

                self.add_output('a', 0.0)
                self.add_output('b', 0.0)
                self.add_output('c', 0.0)
                self.add_output('d', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                outputs['a'] = (2.0*x[0] - 3.0*x[1])**2
                outputs['b'] = 18.0 - 32.0*x[0] + 12.0*x[0]**2 + 48.0*x[1] - 36.0*x[0]*x[1] + 27.0*x[1]**2
                outputs['c'] = (x[0] + x[1] + 1.0)**2
                outputs['d'] = 19.0 - 14.0*x[0] + 3.0*x[0]**2 - 14.0*x[1] + 6.0*x[0]*x[1] + 3.0*x[1]**2


        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp('x', np.array([0.2, -0.2])))
        model.add_subsystem('comp', MyComp())
        model.add_subsystem('obj', ExecComp('f=(30 + a*b)*(1 + c*d)'))

        model.connect('px.x', 'comp.x')
        model.connect('comp.a', 'obj.a')
        model.connect('comp.b', 'obj.b')
        model.connect('comp.c', 'obj.c')
        model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        model.add_design_var('px.x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        model.add_objective('obj.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'px.x' : 16}

        prob.setup(check=False)
        prob.run_driver()

        # TODO: Satadru listed this solution, but I get a way better one.
        # Solution: xopt = [0.2857, -0.8571], fopt = 23.2933
        assert_rel_error(self, prob['obj.f'], 12.37341703, 1e-4)
        assert_rel_error(self, prob['px.x'][0], 0.2, 1e-4)
        assert_rel_error(self, prob['px.x'][1], -0.88705882, 1e-4)

    def test_mixed_integer_branin(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('xC', 7.5))
        model.add_subsystem('p2', IndepVarComp('xI', 0.0))
        model.add_subsystem('comp', Branin())

        model.connect('p2.xI', 'comp.x0')
        model.connect('p1.xC', 'comp.x1')

        model.add_design_var('p2.xI', lower=-5.0, upper=10.0)
        model.add_design_var('p1.xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'p1.xC' : 8}
        prob.driver.options['max_gen'] = 400

        prob.setup(check=False)
        prob.run_driver()

        # Optimal solution
        assert_rel_error(self, prob['comp.f'], 0.49398, 1e-4)
        self.assertTrue(int(prob['p2.xI']) in [3, -3])

    def test_mixed_integer_3bar(self):

        class ObjPenalty(ExplicitComponent):
            """
            Weight objective with penalty on stress constraint.
            """
            def setup(self):
                self.add_input('obj', 0.0)
                self.add_input('stress', val=np.zeros((3, )))

                self.add_output('weighted', 0.0)

            def compute(self, inputs, outputs):
                obj = inputs['obj']
                stress = inputs['stress']

                pen = 0.0
                for j in range(len(stress)):
                    if stress[j] > 1.0:
                        pen += 10.0*(stress[j] - 1.0)**2

                outputs['weighted'] = obj + pen


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('xc_a1', IndepVarComp('area1', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a2', IndepVarComp('area2', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xc_a3', IndepVarComp('area3', 5.0, units='cm**2'), promotes=['*'])
        model.add_subsystem('xi_m1', IndepVarComp('mat1', 1), promotes=['*'])
        model.add_subsystem('xi_m2', IndepVarComp('mat2', 1), promotes=['*'])
        model.add_subsystem('xi_m3', IndepVarComp('mat3', 1), promotes=['*'])
        model.add_subsystem('comp', ThreeBarTruss(), promotes=['*'])
        model.add_subsystem('obj_with_penalty', ObjPenalty(), promotes=['*'])

        model.add_design_var('area1', lower=0.0005, upper=10.0)
        model.add_design_var('area2', lower=0.0005, upper=10.0)
        model.add_design_var('area3', lower=0.0005, upper=10.0)
        model.add_design_var('mat1', lower=1, upper=4)
        model.add_design_var('mat2', lower=1, upper=4)
        model.add_design_var('mat3', lower=1, upper=4)
        model.add_objective('weighted')

        prob.driver = SimpleGADriver()
        prob.driver.options['bits'] = {'area1' : 16,
                                       'area2' : 16,
                                       'area3' : 16}
        prob.driver.options['max_gen'] = 3000
        prob.driver.options['pop_size'] = 25

        prob.setup(check=False)

        prob.run_driver()

        print(prob['mass'], prob['mat1'], prob['mat2'], prob['mat3'], prob['stress'])
        assert_rel_error(self, prob['mass'], 5.287, 1e-3)
        assert_rel_error(self, prob['mat1'], 3, 1e-5)
        assert_rel_error(self, prob['mat2'], 3, 1e-5)
        #Material 3 can be anything

if __name__ == "__main__":
    unittest.main()
