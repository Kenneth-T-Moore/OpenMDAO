import os
from time import time

import numpy as np

from jax import vmap, jit, jacfwd, jvp, config, jacrev
import jax.numpy as jnp

import openmdao.api as om
from openmdao.components.interp_util.interp import InterpND
from openmdao.core.driver import Driver


# Need this on my Ubuntu VM.
os.environ['JAX_SKIP_CUDA_CONSTRAINTS_CHECK'] = 'True'

# Need 64bit.
config.update("jax_enable_x64", True)

# Config
VEC_SIZE = 10
NUM_ITER = 2000
NUM_COMP = 5
DERIV_FRAC = 1

print('')
print(f"Num Comps = {NUM_COMP}, Num Nodes = {VEC_SIZE}, Num Opt Iters = {NUM_ITER}, Deriv Iter Percentage = {100.0 / DERIV_FRAC}")
print('')

# Problem Setup: Interpolate over a 2d space.

p1 = np.linspace(0, 100, 25)
p2 = np.linspace(-10, 10, 15)
p3 = np.linspace(0, 1, 12)
P1, P2, P3 = np.meshgrid(p1, p2, p3, indexing='ij')
values = np.sqrt(P1) - P2 * P3

x_np = np.array([[8.5, 2.6, 0.34]])

p1_jax = jnp.linspace(0, 100, 25)
p2_jax = jnp.linspace(-10, 10, 15)
p3_jax = jnp.linspace(0, 1, 12)
p_jax = (p1_jax, p2_jax, p3_jax)
P1_jax, P2_jax, P3_jax = jnp.meshgrid(p1_jax, p2_jax, p3_jax, indexing='ij')
values_jax = jnp.sqrt(P1_jax) - P2_jax * P3_jax

x_jax = jnp.array([[8.5, 2.6, 0.34]])


# Interpolators

interp_om = InterpND(points=(p1, p2, p3), values=values, method='lagrange3', extrapolate=True)
interp_om_fastest = InterpND(points=(p1, p2, p3), values=values, method='3D-lagrange3', extrapolate=True)

@jit
def sub_interp(x, x_data, y_data, idx=None):

    if idx is None:
        idx = jnp.searchsorted(x_data[0], x[0]) - 1
        idx = jnp.clip(idx, 0, len(x_data[0]) - 3)

    p1 = x_data[0][idx - 1]
    p2 = x_data[0][idx]
    p3 = x_data[0][idx + 1]
    p4 = x_data[0][idx + 2]

    xx1 = x[0] - p1
    xx2 = x[0] - p2
    xx3 = x[0] - p3
    xx4 = x[0] - p4

    c12 = 1.0 / (p1 - p2)
    c13 = 1.0 / (p1 - p3)
    c14 = 1.0 / (p1 - p4)
    c23 = 1.0 / (p2 - p3)
    c24 = 1.0 / (p2 - p4)
    c34 = 1.0 / (p3 - p4)

    if len(x) > 1:

        sub_idx = jnp.searchsorted(x_data[1], x[1]) - 1
        idx = jnp.clip(idx, 0, len(x_data[1]) - 3)

        q1 = sub_interp(x[1:], x_data[1:], y_data[idx - 1, :], idx=sub_idx) * (c12 * c13 * c14)
        q2 = sub_interp(x[1:], x_data[1:], y_data[idx, :], idx=sub_idx) * (c12 * c23 * c24)
        q3 = sub_interp(x[1:], x_data[1:], y_data[idx + 1, :], idx=sub_idx) * (c13 * c23 * c34)
        q4 = sub_interp(x[1:], x_data[1:], y_data[idx + 2, :], idx=sub_idx) * (c14 * c24 * c34)

    else:
        q1 = y_data[idx - 1] * (c12 * c13 * c14)
        q2 = y_data[idx] * (c12 * c23 * c24)
        q3 = y_data[idx + 1] * (c13 * c23 * c34)
        q4 = y_data[idx + 2] * (c14 * c24 * c34)

    return xx4 * (xx3 * (q1 * xx2 - q2 * xx1) + q3 * xx1 * xx2) - q4 * xx1 * xx2 * xx3


@jit
def slinear_interpolate(x_data_jax, y_data_jax, x_query):

    def interp(xq):
        return sub_interp(xq, x_data_jax, y_data_jax)

    return vmap(interp, in_axes=0)(x_query)


# Components

class OMInterpSlow(om.ExplicitComponent):

    def setup(self):

        self.add_input('x', shape=(VEC_SIZE, 3))
        self.add_output('y', shape=(VEC_SIZE, ))

    def setup_partials(self):

        rows = np.repeat(np.arange(VEC_SIZE), 3)
        cols = np.arange(VEC_SIZE * 3)
        self.declare_partials('y', 'x', rows=rows, cols=cols)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x = inputs['x']
        y, self.dydx = interp_om.interpolate(x, compute_derivative=True)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials['y', 'x'] = self.dydx.ravel()


class OMInterpFast(OMInterpSlow):

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        x = inputs['x']
        y, self.dydx = interp_om_fastest.interpolate(x, compute_derivative=True)


class JAXInterp(om.JaxExplicitComponent):


    def __init__(self, fallback_derivs_method='fd', **kwargs):  # noqa
        super().__init__(**kwargs)

        # faster with this off, but individual decorates active
        self.options['use_jit'] = False

    def setup(self):

        self.add_input('x', shape=(VEC_SIZE, 3))
        self.add_output('y', shape=(VEC_SIZE, ))

    def setup_partials(self):

        self.declare_partials('y', 'x')

    def compute_primal(self, x):

        y = slinear_interpolate(p_jax, values_jax, x)
        return y


class FakeOpt(Driver):

    def run(self):

        for j in range(NUM_ITER):

            self._run_solve_nonlinear()

            if not j % DERIV_FRAC:
                self._compute_totals(of=list(self._cons.keys()),
                                     wrt=list(self._designvars.keys()),
                                     return_format='dict')


def run_driver(comp_class):

    prob = om.Problem()
    model = prob.model

    prob.driver = FakeOpt()

    model.add_design_var('x')

    for j in range(NUM_COMP):
        comp = comp_class()
        name = f'comp{j}'
        model.add_subsystem(name, comp, promotes_inputs=['*'])
        model.add_constraint(f'{name}.y', lower=0)

    prob.setup()

    prob.set_val('x', np.repeat(x_np, VEC_SIZE, 0))

    t0 = time()
    prob.run_driver()
    t1 = time() - t0

    return t1


t1 = run_driver(OMInterpSlow)
print(f"OM lagrange3:    {t1}")

t1 = run_driver(OMInterpFast)
print(f"OM 3D-lagrange3: {t1}")

t1 = run_driver(JAXInterp)
print(f"JAX lagrange3:   {t1}")


print('done')