[t, b] = ddm_rand_full(mu, sig2, b_lo, b_up, delta_t, n, inv_leak, seed)
%% draw first-passage time samples and boundaries from a diffusion model.
%
% Unless drift, variance and bounds are constant, the method uses the
% Euler-Maruyama method to simulate the diffusion model.
%
% [t, b] = ddm_rand_full(mu, sig2, b_lo, b_up, delta_t, n[, inv_leak[, seed]])
%
% mu, ..., b_up are all vectors in steps of delta_t. mu and sig2 are the
% drift rate and variance, respectively. b_lo and b_up are the location of the
% lower and upper bound. delta_t is the simulation step size. n is the number
% of samples to draw. If the given vectors are shorter than the simulation
% times, then their last element is replicated.
%
% If inv_leak is given, a leaky integator rather than a non-leaky one is
% assumed. In this case, inv_leak is 1 / leak time constant. The non-leaky
% case is the same as inv_leak = 0.
%
% If seed is given at not 0, it is used as the seed for the random number
% generator.
%
% The assumed model is
%
% dx / dt = - inv_leak * x(t) + mu(t) + sqrt(sig2(t)) eta(t)
%
% where eta is zero-mean unit variance white noise. The bound is on x.
%
% Copyright (c) 2016 Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

error('Not implemented as M-file. Make sure that mex file is complied');
