from time import time

import numpy as np

from openmdao.components.interp_util.interp import InterpND


# Example usage
x_data = np.arange(1, 1000, 1)
y_data = 3.0 * np.sin(x_data * np.pi / 1000.)
x_query = np.linspace(5.0, 50.0, 100000)

#interp = InterpND(method='scipy_slinear', points=x_data, values=y_data)
interp = InterpND(method='slinear', points=x_data, values=y_data)


t0 = time()
interp.interpolate(x_query, compute_derivative=True)
print("Elapsed Time", time() - t0)