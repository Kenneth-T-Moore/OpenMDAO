""" Test for the relaxation Line Search"""

import unittest

import numpy as np

try:
    import matplotlib
except ImportError:
    matplotlib = None

from openmdao.api import Problem, IndepVarComp, ScipyKrylov, NewtonSolver
from openmdao.solvers.linesearch.relaxation import RelaxationLS
from openmdao.test_suite.components.implicit_newton_linesearch import ImplCompTwoStates
from openmdao.test_suite.test_examples.test_circuit_analysis import Circuit
from openmdao.utils.assert_utils import assert_rel_error


class TestRelaxationLS(unittest.TestCase):

    def test_bad_settings(self):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        newton = p.model.circuit.nonlinear_solver
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['relax_far'] = 1.0
        newton.linesearch.options['relax_near'] = 2.0

        with self.assertRaises(Exception) as raises_cm:
            p.final_setup()

        exception = raises_cm.exception
        msg = "In options, relax_far must be greater than or equal to relax_near."

        self.assertEqual(exception.args[0], msg)

    def test_circuit_advanced_newton(self):
        p = Problem()
        model = p.model

        model.add_subsystem('ground', IndepVarComp('V', 0., units='V'))
        model.add_subsystem('source', IndepVarComp('I', 0.1, units='A'))
        model.add_subsystem('circuit', Circuit())

        model.connect('source.I', 'circuit.I_in')
        model.connect('ground.V', 'circuit.Vg')

        p.setup()

        # you can change the NewtonSolver settings in circuit after setup is called
        newton = p.model.circuit.nonlinear_solver
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 15
        newton.options['solve_subsystems'] = True
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['initial_relaxation'] = 0.019
        newton.linesearch.options['relax_far'] = 1.15e-3
        newton.linesearch.options['relax_near'] = 1.13e-3

        # set some initial guesses
        p['circuit.n1.V'] = 10.
        p['circuit.n2.V'] = 1e-3

        p.run_model()

    def test_linesearch_bounds_vector(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='vector')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_rel_error(self, top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_wall(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.nonlinear_solver.options['maxiter'] = 10
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='wall')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should go to the lower bound and stall
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        assert_rel_error(self, top['comp.z'], 1.5, 1e-8)

        # Test upper bound: should go to the upper bound and stall
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        assert_rel_error(self, top['comp.z'], 2.5, 1e-8)

    def test_linesearch_bounds_scalar(self):
        top = Problem()
        top.model.add_subsystem('px', IndepVarComp('x', 1.0))
        top.model.add_subsystem('comp', ImplCompTwoStates())
        top.model.connect('px.x', 'comp.x')

        top.model.nonlinear_solver = NewtonSolver()
        top.model.linear_solver = ScipyKrylov()

        top.model.nonlinear_solver.linesearch = RelaxationLS(bound_enforcement='scalar')

        # Setup again because we assigned a new linesearch
        top.setup(check=False)

        # Test lower bound: should stop just short of the lower bound
        top['px.x'] = 2.0
        top['comp.y'] = 0.0
        top['comp.z'] = 1.6
        top.run_model()
        self.assertTrue(1.5 <= top['comp.z'] <= 1.6)

        # Test lower bound: should stop just short of the upper bound
        top['px.x'] = 0.5
        top['comp.y'] = 0.0
        top['comp.z'] = 2.4
        top.run_model()
        self.assertTrue(2.4 <= top['comp.z'] <= 2.5)


class TestFeatureRelaxationLS(unittest.TestCase):

    def test_atan(self):
        from math import atan

        from openmdao.api import Problem, IndepVarComp, ImplicitComponent, NewtonSolver
        from openmdao.api import DirectSolver
        from openmdao.solvers.linesearch.relaxation import RelaxationLS

        class MyComp(ImplicitComponent):

            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('y', 1.0)

                self.declare_partials(of='y', wrt='x')
                self.declare_partials(of='y', wrt='y')

            def apply_nonlinear(self, inputs, outputs, residuals):
                x = inputs['x']
                y = outputs['y']

                residuals['y'] = (33.0 * atan(y-20.0))**2 + x

            def linearize(self, inputs, outputs, jacobian):
                x = inputs['x']
                y = outputs['y']

                jacobian['y', 'y'] = 2178.0*atan(y-20.0) / (y**2 - 40.0*y + 401.0)
                jacobian['y', 'x'] = 1.0


        p = Problem()
        model = p.model

        model.add_subsystem('px', IndepVarComp('x', -100.0))
        model.add_subsystem('comp', MyComp())

        model.connect('px.x', 'comp.x')

        p.setup()

        p['comp.y'] = 12.0

        # You can change the NewtonSolver settings after setup is called
        newton = p.model.nonlinear_solver = NewtonSolver()
        p.model.linear_solver = DirectSolver()
        newton.options['iprint'] = 2
        newton.options['rtol'] = 1e-8
        newton.options['maxiter'] = 75
        newton.options['solve_subsystems'] = True

        # Tailored the relaxation settings to give good convergence.
        newton.linesearch = RelaxationLS()
        newton.linesearch.options['initial_relaxation'] = 0.3
        newton.linesearch.options['relax_far'] = 1.0e-1
        newton.linesearch.options['relax_near'] = 1.0e-4
        newton.linesearch.options['abs_or_rel_norm'] = 'rel'

        p.run_model()

        assert_rel_error(self, p['comp.y'], 19.68734033, 1e-6)

    @unittest.skipUnless(matplotlib, "Matplotlib is required.")
    def test_plot_atan(self):
        from numpy import arctan

        xx = np.linspace(12, 55.0, 100)
        yy = (33.0 * arctan(xx-20.0))**2 - 100.0

        grad = 2178.0*arctan(12-20.0) / (12**2 - 40.0*12 + 401.0)

        dxx = [xx[0], xx[0] - yy[0]/grad]
        dyy = [yy[0], 0.0]

        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(xx, yy)
        plt.plot([19.68734033], [0.], 'r*')
        plt.plot([-19.68734033 + 40.0], [0.], 'r*')
        plt.plot(dxx, dyy, 'r--')
        plt.xlabel("y")
        plt.ylabel('F(y)')
        plt.title("Newton without relaxation")
        plt.text(40, 1150, "First Newton step.")
        plt.text(15, 0, "Roots.")
        plt.grid()
        plt.show()
        print('')

if __name__ == "__main__":
    unittest.main()
