dm
==

Diffusion model first passage time distribution and sampling library in C++11, with Python and MATLAB interface.

*No guarantee is provided for the correctness of the implementation.*

The code is licensed under the New BSD License.

Content
-------

The library provides classes and methods, written in C++11, that on one hand compute the first-passage time densities of diffusion models with two absorbing boundaries, and on the other hand draws first-passage time and boundary samples. Each specialised class inherits the abstract `DMBase` and provides various optimization to compute this density and to sample. The library supports leaky/weighted integration, time-varying drift rates, and time-varying symmetric or asymmetric boundaries.

In addition to the C++11 implementation, a Python and a MATLAB interface are provided. In both cases, the interface chooses between the various different library classes depending on the provided parameters.

The diffusion models assume a drifting and diffusing particle *x(t)* that starts at *x(0) = 0*and whose time-course follows

*dx = mu(t) dt + sig(t) dW* ,

where *mu(t)* is the current drift, *sig(t)* is the current diffusion standard deviation, and *dW* is a Wiener process. Diffusion is terminated as soon as the particle reaches either the upper boundary *theta_u(t)* or lower boundary *theta_t(t)*. The library requires *theta_u(0) > 0* and *theta_l(0) < 0*. The time at which either boundary is reached is the first-passage time. The associated densities, *g_u(t)* and *g_l(t)*, are the joint densities over bounds and first-passage times, such that

*integral_0^infinity (g_u(t) + g_l(t)) dt = 1* .

In addition to perfect integration, the library provides leaky integration

*dx =  (- x(t)/tau(t) + mu(t)) dt + sig(t) dW* ,

with time constant *tau*, as well as weighted integration

*dx = mu(t) (k mu(t) dt + dW)* .

The library provides specialised classes for time-invariant drifts, *mu(t) = mu_0* for all *t*, time-invariant bounds, *theta(t) = theta_0* for all *t*, symmetric bounds, *theta_l(t) = - theta_u(t)*, and leaks and weights (see below). If not otherwise mentioned, a constant unit diffusion variance is assumed, that is *sig^2(t) = 1* for all *t*.

Usage
-----

The library headers and source are in [src/](src/). They require a C++ compiler that supports the C++11 (or later) standard.

## ExtArray

All diffusion model classes in the library use `ExtArray` to represent vectors of real numbers. This `ExtArray` class is used to provide efficient interfacing to external libraries. Its particular feature is that it can be indexed beyond its actual length, in which case it returns a default `last` value. This is particularly useful for diffusion models that have a drift or bounds that vary until a certain time, and then remain constant. Similarly, time-invariant (i.e. constant) drifts or bounds are just special cases of `ExtArray`.

Internally, `ExtArray` encapsultes a C-style array of type `double[]`, wrapped with a C++11 `std::shared_ptr`. It can operate in two ownership modes:

- Shared ownership: all copies of a particular `ExtArray` share ownership of the encapusalted array through `std::shared_ptr`. As soon as the last instance associated with this array is destroyed, the array itself is deleted. Shared ownership results from calling the constructor with `ExtArray(ExtArray::shared_owner(x), ...)` where `x` is of type `double[]`.

- No ownership: neither copy of a particular `ExtArray` has ownership of the array. Here, it is important that the array is not deleted before all `ExtArray`'s that are associated with it. The no-ownership state results from calling the constructor with `ExtArray(ExtArray::shared_noowner(x), ...)` where `x` is again of type `double[]`. This type of ownership is particularly useful to avoid copying data when providing interfaces to other languages. Both the MATLAB and the Python interface use this type of ownership extensively.

If a `last` argument is provided to the constructor of `ExtArray`, this element is returned for indices beyond the size of the stored array. Otherwise, the array's last element at construction is returned. `ExtArray::size()` returns the actual size of the stored array, such that `x[n]` for all `n >= x.size()` returns the same, last element. This last element is not included in `size()`, such that an `ExtArray` of size 0  returns the last element for all indicies `n`.


## DMBase and inherited classes

All diffusion model classes are based on the abstract `DMBase` class. This class defines the interface to compute first-passage time densities and to draw samples, and provides various factory function to create diffusion models. For time-varying drifts/bounds, the corresponding vectors need to be specified in steps of *dt*, where *dt* is provided upon construction.

The currently provided factory functions are:

Create diffusion model with constant or time-varying drift, and constant or time-varying symmetric boundaries
```C++
DMBase* dm = DMBase::create(const ExtArray& drift, const ExtArray& bound, value_t dt);
```
In the above, the drift *mu(t)* is specified by `drift`, and the bounds *theta_u(t) = -theta_l(t)* by `bound`. `dt` specifies the time-step of the associated vectors.

Create a diffusion model with constant or time-varying drift, constant or time-varying diffusion variance, and constant or time-varying asymmetric boundaries, with or without leaky integration
```C++
DMBase* dm = DMBase::create(const ExtArray& drift, const ExtArray& sig2,
                            const ExtArray& b_lo, const ExtArray& b_up,
                            const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                            double dt[, double invleak]);
```
In the above, `drift` is the vector of drifts, `sig2` the vector of diffusion standard deviations, `b_lo` the vector specifying the lower boundary *theta_l(t)*, `b_up(t)` the vector specifying the upper boundary *theta_u(t)*, `b_lo_deriv` the vector of time-derivatives of the lower boundary *theta_l'(t)*, and `b_up_deriv` the vector of time-derivatives of the upper boundary *theta_u'(t)*. `dt` again specifies the time-step of the associated vectors. The optional `invleak` specified the inverse leak time constant *1/tau*.

For weighted integration, use
```C++
DMBase* dm = DMBase::createw(const ExtArray& drift, const ExtArray& bound,
                             double k, double dt);
```

`DMBase` specifies the following methods to compute the first-passage time densities:

```C++
void pdfseq(int n, ExtArray& g1, ExtArray& g2);
double pdfu(double t);
double pdfl(double t);
```

`pdfseq(n, g1, g2)` computes the first-passage time densities *g_u(t)* and *g_l(t)* in *n* steps of *dt* (provided to the constructor). Both `g1` and `g2` need to have a capacity of at least `n`. After calling `pwdseq(.)`, they contain *g_u(dt), g_u(2 dt), ..., g_u(n dt)* and *g_l(dt), g_l(2 dt), ..., g_l(n dt)*, respectively.

`pdfu(t)` and `pdfl(t)` return *g_u(t)* and *g_l(t)*, respectively. **WARNING**: by default, they call `pdfseq(n,..)` with `n > t / dt` and interpolate between the returned values. If the first-passage time densities need to be computed for both boundaries or multiple times, it is always more efficient to call `pdfseq(.)` directly. Only the `DMConstDriftConstBound` and `DMConstDriftConstABount` classes contain specialised implementations of `pdfu(t)` and `pdfl(t)` that are faster than a single call to `pdfseq(.)`.

`DMBase` provides the following method to draw first-passage time and boundary samples:

```C++
DMSample rand(rngeng_t rngeng);
```
This method returns a single diffusion model sample, where `rngeng` is and C++11 random number generator engine of type `DMBase::rngeng_t`. The returned sample is of type DMSample, which provides two methods. The method `t()` returns the first-passage time of the sample, and the method `upper_bound()` returns `true` if the upper boundary was reached, and `false` otherwise.

In addition to the above, `DMBase` provides the following convenience method:

```C++
void DMBase::mnorm(ExtArray& g1, ExtArray& g2);
```
This method normalises the mass such that `dt (sum(g1) + sum(g2)) = 1`, and additionally makes sure that `g1(t) >= 0` and `g2(t) >= 0` for all `t`. It does so by adding mass to the last elements of `g1` and `g2` such that the ratio `sum(g1) / (sum(g1) + sum(g2))` remains unchanged.


The factory functions creates an instance of one of the following classes:

`DMConstDriftConstBound`: constant drift, constant symmetric bounds.

`DMConstDriftConstABound`: constant drift, constant asymmetric bounds.

`DMConstDriftVarBound`: constant drift, time-varying symmetric bounds.

`DMVarDriftVarBound`: time-varying drift, time-varying symmtric bounds.

`DMWVarDriftVarBound`: time-varying drift, time-varying symmetric bounds, weighted integration.

`DMGeneralDeriv`: time-varying drift, time-varying variance, time-varying asymmetric bounds.

`DMGeneralLeakDeriv`: time-varying drift, time-varying variance, time-varying asymmetric bounds, leaky integration.


Interfaces
----------

See [matlab/README.md](matlab/README.md) and [python/README.md](python/README.md) for a description of the corresponding interface.


References
----------

In general, the library computes the first-passage time densities by finding the solution to an integral equation, as described in

Smith PL (2000). [Stochastic Dynamic Models of Response Time and Accuracy: A Foundational Primer](http://dx.doi.org/10.1006/jmps.1999.1260). *Journal of Mathematical Psychology*, 44 (3). 408-463.

For constant drift and bounds, it instead uses a much faster method, based on an infinite series expansion of these densities, as described in.

Cox DR and Miller HD (1965). *The Theory of Stochastic Processes*. John Wiley & Sons, Inc.

and

Navarro DJ and Fuss IG (2009). [Fast and accurate calculations for first-passage times in Wiener diffusion models](http://dx.doi.org/10.1016/j.jmp.2009.02.003). *Journal of Mathematical Psychology*, 53, 222-230.

Samples are in the most general case drawn by simulating trajectories by the Euler–Maruyama method. For diffusion models with constant drift and (symmetric or asymmetric) boundaries, the following significantly faster method based on rejection sampling is used:

Drugowitsch J (2016). [Fast and accurate Monte Carlo sampling of first-passage times from Wiener diffusion models](http://dx.doi.org/10.1038/srep20490). *Scientific Reports* 6, 20490; doi: 10.1038/srep20490.