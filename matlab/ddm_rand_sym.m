function [t, b] = ddm_rand_sym(mu, bound, delta_t, n, seed)
%% [t, b] = ddm_rand_sym(mu, bound, delta_t, n[, seed])
%
% draw first-passage time samples and boundaries from diffusion model with
% symmetric boundaries.
%
% [t, b] = ddm_rand_sym(mu, bound, delta_t, n[, seed])
%
% mu and bound are vectors of drift rates and bound height over time, in steps
% of delta_t. n is the number of samples to draw. If seed is given at not 0,
% it is used as the seed for the random number generator.
%
% The assumed model is
%
% dx / dt = mu(t) + eta(t)
%
% where eta is zero-mean unit variance white noise. The bound is on x and -x.
%
% The returned t and b are a vector of first-passage times, and booleans about
% which bound (true = upper) was hit. Both vectors are of size n.
%
% The method uses more efficient methods of computing the first-passage time
% density if either mu is constant (i.e. given as a scalar) or both mu and
% the bound are constant.
%
% Copyright (c) 2013, 2014 Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

error('Not implemented as M-file. Make sure that mex file is complied');
