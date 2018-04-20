import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, CSCJacobian, NewtonSolver, \
    ScipyKrylov, LinearRunOnce, DenseJacobian

from openmdao.test_suite.components.double_sellar import DoubleSellar


def _baseline(mode):
    p = Problem()

    dv = p.model.add_subsystem('dv', IndepVarComp(), promotes=['*'])
    dv.add_output('z', [1.,1.])

    p.model.add_subsystem('double_sellar', DoubleSellar())
    p.model.connect('z', ['double_sellar.g1.z', 'double_sellar.g2.z'])

    p.model.add_design_var('z', lower=-10, upper=10)
    p.model.add_objective('double_sellar.g1.y1')

    p.setup(mode=mode)

    p.model.nonlinear_solver = NewtonSolver()
    p.model.nonlinear_solver.options['solve_subsystems'] = True

    p.run_model()

    objective = p['double_sellar.g1.y1']
    jac = p.compute_totals()

    return objective, jac


def _masking_case(mode):
    p = Problem()

    dv = p.model.add_subsystem('dv', IndepVarComp(), promotes=['*'])
    dv.add_output('z', [1.,1.])

    p.model.add_subsystem('double_sellar', DoubleSellar())
    p.model.connect('z', ['double_sellar.g1.z', 'double_sellar.g2.z'])

    p.model.add_design_var('z', lower=-10, upper=10)
    p.model.add_objective('double_sellar.g1.y1')

    p.setup(mode=mode)

    p.model.double_sellar.g1.jacobian = CSCJacobian()
    p.model.double_sellar.g1.linear_solver = DirectSolver()
    p.model.double_sellar.g1.nonlinear_solver = NewtonSolver()

    p.model.double_sellar.g2.jacobian = CSCJacobian()
    p.model.double_sellar.g2.linear_solver = DirectSolver()
    p.model.double_sellar.g2.nonlinear_solver = NewtonSolver()

    p.model.nonlinear_solver = NewtonSolver()
    p.model.nonlinear_solver.options['solve_subsystems'] = True
    p.model.linear_solver = ScipyKrylov()
    p.model.linear_solver.precon = LinearRunOnce()

    p.run_model()

    objective = p['double_sellar.g1.y1']
    jac = p.compute_totals()

    return objective, jac

class CSCMaskingTestCase(unittest.TestCase):
    def test_csc_masking_fwd(self):
        base_objective, base_jac = _baseline('fwd')
        obj, jac = _masking_case('fwd')

        assert_almost_equal(base_objective, obj, decimal=6)

        for key in jac:
            assert_almost_equal(base_jac[key], jac[key], decimal=6)

    def test_csc_masking_rev(self):
        base_objective, base_jac = _baseline('rev')
        obj, jac = _masking_case('rev')

        assert_almost_equal(base_objective, obj, decimal=6)

        for key in jac:
            assert_almost_equal(base_jac[key], jac[key], decimal=6)
