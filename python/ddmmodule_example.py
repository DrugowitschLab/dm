
import numpy as np

import ddm

mu = np.array([1.0])
mu2 = np.array([1.0])
sig2 = np.array([1.0])
bound = np.array([1.0])
bound_deriv = np.zeros(1)
dt = 0.01
tmax = 4.0


# compare fpt() and fpt_full()
g1, g2 = ddm.fpt(mu, bound, dt, tmax)
g1f, g2f = ddm.fpt_full(mu, sig2, -bound, bound, bound_deriv, bound_deriv, dt, tmax);
print("Largest difference between fpt and fpt_full")
print("g1: %f" % np.max(np.abs(g1 - g1f)))
print("g2: %f" % np.max(np.abs(g2 - g2f)))

# compare fpt() with vector drift and fpt_full()
g1, g2 = ddm.fpt(mu2, bound, dt, tmax)
g1f, g2f = ddm.fpt_full(mu2, sig2, -bound, bound, bound_deriv, bound_deriv, dt, tmax);
print()
print("Largest difference between fpt and fpt_full (vector drift)")
print("g1: %f" % np.max(np.abs(g1 - g1f)))
print("g2: %f" % np.max(np.abs(g2 - g2f)))

