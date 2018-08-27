.. _feature_relaxation_linesearch:

************
RelaxationLS
************

RelaxationLS is a linesearch that implements a relaxation method wherein the step requested by a solver is
multiplied by a number that is usually less than 1.0 to slow down early convergence and prevent it from
shooting off too far to the next point, particularly where there is a large area with a shallow gradient.
The RelaxationLS allows you to specify an initial value for a relaxation parameter for when you are far
from the solution as measured by the absolute norm of the residual. The value near to the solution is
always 1.0. The RelaxationLS also allows you to specify the value of the residual norm that defines the far
and near regions. Between these two points, the relaxation parameter is scaled logarithmically from the
far value to 1.0.

RelaxationLS Options
--------------------

.. embed-options::
    openmdao.solvers.linesearch.relaxation
    RelaxationLS
    options

RelaxationLS Option Examples
----------------------------

The following example shows a difficult problem that Newton cannot solve on its own from the given starting point.
An implicit component is created
to evaluate the function of the square of the inverse tangent of the state. This problem is difficult because it
has a shallow slope far from the solution, and a deep slope near it. The shallow slope causes Newton to overshoot
the actual root on the first iteration. Subsequent iterations overshoot further and the solution diverges.

.. embed-code::
    openmdao.solvers.linesearch.tests.test_relaxation.TestFeatureRelaxationLS.test_plot_atan
    :layout: plot

When we add a relaxation linesearch, we limit the Newton step so that it doesn't overshoot. As we get closer to
the solution, and we dive into the "well" in the plot above, the relaxation gradually returns to 1.0, which
corresponds to a full Newton step, so that we can converge more quickly.

.. embed-code::
    openmdao.solvers.linesearch.tests.test_relaxation.TestFeatureRelaxationLS.test_atan
    :layout: code, output