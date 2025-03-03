import os
from time import time

import numpy as np

from jax import vmap, jit, jacfwd, jvp, config
import jax.numpy as jnp

from openmdao.components.interp_util.interp import InterpND

# Need this on my Ubuntu VM.
os.environ['JAX_SKIP_CUDA_CONSTRAINTS_CHECK'] = 'True'

# Need 64bit.
config.update("jax_enable_x64", True)

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

# Answers are a little different if you convert them from numpy.
#p_jax = (jnp.asarray(p1), jnp.asarray(p2), jnp.asarray(p3))
#values_jax = jnp.asarray(values)
#x_jax = jnp.asarray(x_np)

# Interpolators

interp_om = InterpND(points=(p1, p2, p3), values=values, method='slinear', extrapolate=True)
interp_om_fastest = InterpND(points=(p1, p2, p3), values=values, method='3D-slinear', extrapolate=True)

@jit
def sub_interp(x, x_data, y_data, idx=None):

    if idx is None:
        idx = jnp.searchsorted(x_data[0], x[0]) - 1
        idx = jnp.clip(idx, 0, len(x_data[0]) - 2)

    dx_local = x[0] - x_data[0][idx]

    x0 = x_data[0][idx]
    x1 = x_data[0][idx + 1]

    if len(x) > 1:

        sub_idx = jnp.searchsorted(x_data[1], x[1]) - 1
        sub_idx = jnp.clip(idx, 0, len(x_data[1]) - 2)

        y0 = sub_interp(x[1:], x_data[1:], y_data[idx, :], idx=sub_idx)
        y1 = sub_interp(x[1:], x_data[1:], y_data[idx + 1, :], idx=sub_idx)

    else:
        y0 = y_data[idx]
        y1 = y_data[idx + 1]

    slope = (y1 - y0) / (x1 - x0)

    return y0 + slope * dx_local


@jit
def slinear_interpolate(x_data_jax, y_data_jax, x_query):

    def interp(xq):
        return sub_interp(xq, x_data_jax, y_data_jax)

    return vmap(interp, in_axes=0)(x_query)

deriv = jacfwd(slinear_interpolate, argnums=2)
#deriv_matfree = jvp(slinear_interpolate)

@jit
def diagonal_jacobian(x_data, y_data, x_query):
    # Compute only the diagonal entries of the Jacobian
    return vmap(lambda i: jacfwd(slinear_interpolate, argnums=2)(x_data, y_data, x_query)[i, i])(jnp.arange(x_query.size))

def deriv_jvp(x_data_jax, y_data_jax, x_query):

    def deriv_jvp_sub(x):
        tangent = jnp.zeros((x.shape), dtype=jnp.float64)
        for j in jnp.arange(x.size):
            dxj = jvp(slinear_interpolate(x, x_data_jax, y_data_jax), (x, ), (tangent, ))

        return
    return vmap(deriv_jvp_sub, in_axes=0)(x_query)

# diagonal_jacobian = deriv_jvp

# Speed Test Setup

vec_size = 50
count = 1000

x_vec_np = np.repeat(x_np, vec_size, 0)
x_vec_jax = jnp.repeat(x_np, vec_size, 0)

print('')
print('SPEED TEST')
print(f"vec_size = {vec_size}, loop count = {count}")
print('')

# Speed Test

print('Interpolate and derivatives')

t0 = time()
for j in range(count):
    f_om, _ = interp_om.interpolate(x_np, compute_derivative=True)

print('Openmdao slinear:   ', time() - t0)

t0 = time()
for j in range(count):
    f_om, _ = interp_om_fastest.interpolate(x_np, compute_derivative=True)

print('Openmdao slinear 3d:', time() - t0)

t0 = time()
for j in range(count):
    f_jax = slinear_interpolate(p_jax, values_jax, x_jax)
    df_jax = diagonal_jacobian(p_jax, values_jax, x_jax)

print('Jax + Jit compile:  ', time() - t0)

t0 = time()
for j in range(count):
    f_jax = slinear_interpolate(p_jax, values_jax, x_jax)
    df_jax = diagonal_jacobian(p_jax, values_jax, x_jax)

print('Jax:                ', time() - t0)

print('')

print('Interpolate')

t0 = time()
for j in range(count):
    f_om = interp_om.interpolate(x_np, compute_derivative=False)

print('Openmdao slinear:   ', time() - t0)

t0 = time()
for j in range(count):
    f_om = interp_om_fastest.interpolate(x_np, compute_derivative=False)

print('Openmdao slinear 3d:', time() - t0)

t0 = time()
for j in range(count):
    f_jax = slinear_interpolate(p_jax, values_jax, x_jax)

print('Jax:                ', time() - t0)

# Test Accuracy with single point

f_om, df_om = interp_om.interpolate(x_np, compute_derivative=True)
f_om_fastest, df_om_fastest = interp_om_fastest.interpolate(x_np, compute_derivative=True)

f_jax = slinear_interpolate(p_jax, values_jax, x_jax)
df_jax = diagonal_jacobian(p_jax, values_jax, x_jax)

print('')
print('')
print('ACCURACY TEST')
print('')
print('Interpolation')
print('Openmdao slinear', f_om)
print('Openmdao slinear 2d', f_om_fastest)
print('Jax', f_jax)

print('')

print('Derivatives')
print('Openmdao slinear', df_om)
print('Openmdao slinear 2d', df_om_fastest)
print('Jax', df_jax)

print('done')

