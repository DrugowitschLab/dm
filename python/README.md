Building
--------

The Python module can be built by calling

```Shell
python setup.py build
```

and installed by

```Shell
python setup.py install
```

at the command line. The Python module should be compatible with both Python 2 and 3.

Usage
-----

The module provides the following functions:

```Python
 (g1, g2) = fpt(mu, bound, dt, tmax[, mnorm])
 ```
 
 computes the first-passage time distribution of a diffusion model given drift `mu` and bounds at `-bound` and `bound`. `dt` is the time steps in which this distribution is computed, until time `tmax`.
 
 `mu` and `bound` need to be NumPy arrays of type Float. If they are shorter than the returned distribution arrays, then their last element is repeated.
 
 `dt` and `tmax` are Float-type scalar values.
 
 If `mnorm` (boolean, default false) is given and `true`, the returned distributions are normalised before being returned.
 
 ```Python
 (g1, g2) = fpt_w(a, k, bound, dt, tmax[, mnorm])
 ```

 computes the first-passage time distribution assuming weighted accumulation with weights given by vector `a`. `k` is a scalar that determines the proportionality constant.
 
 `a` and `bound` need to be NumPy arrays of type Float. If they are shorter than the returned distribution arrays, then their last element is repeated.
 
 `k`, `dt`, and `tmax` are Float-type scalar values.
 
 `mnorm` has the same effect as for the function `fpt(.)`.
 
 ```Python
 (g1, g2) = fpt_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv, dt, tmax,
                     [inv_leak, mnorm])
```
 
 computed the first-passage time distribution for drift vector `mu`, variance vector `sig2`, lower and upper bound vectors `b_lo` and `b_up`, their time derivatives in vectors `b_lo_deriv` and `b_up_deriv`, in steps of `dt` until time `tmax`. If given, the inverse integrator time-constant is `inv_leak`.
 
 All vectors are NumPy arrays of type Float, which will be extended, if necessary. All scalars are of type Float.
 
 `mnorm` has the same effect as for the function `fpt(.)`.

 ```Python
 (t, b) = rand_sym(mu, bound, dt, n[, seed])
 ```

draws n first-passage time and bound samples from a diffusion model with drift `mu` and bounds at `-bound` and `bound`. `dt` is the time steps in which the diffusion model is simulated (if the Euler–Maruyama method) is used, and in which `mu` and `bound` are specificed.

`mu` and `bound` need to be NumPy arrays of type Float. If they are shorter than the returned distribution arrays, then their last element is repeated. If they are both of size 1, fast sampling without explicitly drawing whole trajectories is performed.

`dt` is a Float-type scalar value, and `n` needs to be an integer.

The optional `seed` (defaults to 0) needs to be an integer and sets the random number generator seed. If not given or 0, the system random number source is used to initialise the random number generator.

The returned `t` and `b` are NumPy arrays of size `n` and of type Float and Boolean, respectively. `t` contains the sampled first-passage times, and `b` is `true` if the upper bound was reached.

```Python
(t, b) = rand_asym(mu, b_lo, b_up, dt, n[, seed])
```

performs the same sampling as `rand_sym(.)`, only for asymmetric bounds at `b_lo` and `b_up`. These vectors need to be NumPy arrays of type Float. If 'mu', 'b_lo' and 'b_up' are all of size 1, fast sampling without explicitly drawing whole trajectories is performed.
