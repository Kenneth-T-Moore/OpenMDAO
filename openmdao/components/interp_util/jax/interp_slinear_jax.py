import os
from time import time

import numpy as np

from jax import vmap, jit, jacfwd, jvp
import jax.numpy as jnp

os.environ['JAX_SKIP_CUDA_CONSTRAINTS_CHECK'] = 'True'

#@jit
def slinear_interpolate(x_data, y_data, x_query):

    #x_data = jnp.array(x_data_np)
    #y_data = jnp.array(y_data_np)
    #x_query = jnp.array(x_query_np)

    def interp(xq):
        dx = jnp.diff(x_data)
        dy = jnp.diff(y_data)
        slopes = dy / dx

        idx = jnp.searchsorted(x_data, xq) - 1
        idx = jnp.clip(idx, 0, len(slopes) - 1)
        dx_local = xq - x_data[idx]
        return y_data[idx] + slopes[idx] * dx_local

    return vmap(interp)(x_query)

deriv = jacfwd(slinear_interpolate, argnums=2)
#deriv_matfree = jvp(slinear_interpolate)

@jit
def diagonal_jacobian(x_data, y_data, x_query):
    # Compute only the diagonal entries of the Jacobian
    return vmap(lambda i: jacfwd(slinear_interpolate, argnums=2)(x_data, y_data, x_query)[i, i])(jnp.arange(x_query.size))


# Example usage
x_data = jnp.arange(1, 1000, 1)
y_data = 3.0 * jnp.sin(x_data * np.pi / 1000.)
x_query = jnp.linspace(5.0, 50.0, 10000)


t0 = time()
y_interp = slinear_interpolate(x_data, y_data, x_query)
#y_interp = slinear_interpolate(x_data, y_data, np.array([1.0, 500.0]))
#y_deriv = deriv(x_data, y_data, np.array([1.0, 500.0]))
print("Compile Time", time() - t0)

t0 = time()
y_interp = slinear_interpolate(x_data, y_data, x_query)
print("Elapsed Time", time() - t0)

t0 = time()
y_deriv = diagonal_jacobian(x_data, y_data, x_query)
print("Deriv Compile Time", time() - t0)

t0 = time()
y_deriv = diagonal_jacobian(x_data, y_data, x_query)
print("Deriv Elapsed Time", time() - t0)

t0 = time()
y_deriv = deriv(x_data, y_data, x_query)
print("Dense Deriv Compile Time", time() - t0)

t0 = time()
y_deriv = deriv(x_data, y_data, x_query)
print("Dense Deriv Elapsed Time", time() - t0)