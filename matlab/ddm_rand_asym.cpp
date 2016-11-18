/**
 * Copyright (c) 2013, 2014 Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *
 * ddm_rand_asym.cpp - draw first-passage time samples and boundaries from
 *                     adiffusion model with asymmetric boundaries.
 *
 * [t, b] = ddm_rand_sym(mu, b_lo, b_up, delta_t, n[, seed])
 *
 * mu, b_lo, and b_up are vectors of drift rates and lower and upper bound
 * heights over time, in steps of delta_t. n is the number of samples to draw.
 * If seed is given at not 0, it is used as the seed for the random number
 * generator.
 *
 * The assumed model is
 *
 * dx / dt = mu(t) + eta(t)
 *
 * where eta is zero-mean unit variance white noise. The bounds are at b_lo
 * and b_up.
 *
 * The returned t and b are a vector of first-passage times, and booleans about
 * which bound (true = upper) was hit. Both vectors are of size n.
 *
 * The method uses more efficient methods of computing the first-passage time
 * density if either mu is constant (i.e. given as a scalar) or both mu and
 * the bound are constant.
 */

#include "mex.h"
#include "matrix.h"

#include "mex_helper.h"
#include "../src/ddm_fpt_lib.h"

#include <string>
#include <cassert>
#include <algorithm>


/** the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* [t, b] = ddm_rand_asym(mu, b_lo, b_up, delta_t, n[, seed]) */

    /* Check argument number */
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("ddm_fpt:WrongOutputs", 
                          "Wrong number of output arguments");
    }
    if (nrhs != 5 && nrhs != 6) {
        mexErrMsgIdAndTxt("ddm_fpt:WrongInputs",
                          "Too few input arguments");
    }

    /* Process first 4 arguments */
    if (!MEX_ARGIN_IS_REAL_VECTOR(0))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "First input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(1))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "Second input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(2))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "Third input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(3))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "Fourth input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(4))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "Fifth input argument expected to be a double");
    if (nrhs == 6 && !MEX_ARGIN_IS_REAL_DOUBLE(5))
        mexErrMsgIdAndTxt("ddm_rand_sym:WrongInput",
                          "Sixth input argument expected to be a double");
    int mu_size = std::max(mxGetN(prhs[0]), mxGetM(prhs[0]));
    int b_lo_size = std::max(mxGetN(prhs[1]), mxGetM(prhs[1]));
    int b_up_size = std::max(mxGetN(prhs[2]), mxGetM(prhs[2]));
    ExtArray mu(ExtArray::shared_noowner(mxGetPr(prhs[0])), mu_size);
    ExtArray b_lo(ExtArray::shared_noowner(mxGetPr(prhs[1])), b_lo_size);
    ExtArray b_up(ExtArray::shared_noowner(mxGetPr(prhs[2])), b_up_size);
    double delta_t = mxGetScalar(prhs[3]);
    int n = (int) mxGetScalar(prhs[4]);
    int rngseed = 0;
    if (nrhs == 5)
        rngseed = (int) mxGetScalar(prhs[5]);
    if (delta_t <= 0.0)
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "delta_t needs to be larger than 0.0");
    if (n <= 0)
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "n needs to be larger than 1");
    
    /* reserve space for output */
    plhs[0] = mxCreateDoubleMatrix(1, n, mxREAL);
    plhs[1] = mxCreateLogicalMatrix(1, n);
    double* t = mxGetPr(plhs[0]);
    mxLogical* b = mxGetLogicals(plhs[1]);

    /* perform sampling */
    DMBase* dm = DMBase::create(mu, ExtArray::const_array(1.0), b_lo, b_up,
                                ExtArray::const_array(0.0), ExtArray::const_array(0.0),
                                delta_t);
    DMBase::rngeng_t rngeng;
    if (rngseed == 0) rngeng.seed(std::random_device()());
    else rngeng.seed(rngseed);
    for (int i = 0; i < n; ++i) {
        DMSample s = dm->rand(rngeng);
        t[i] = s.t();
        b[i] = s.upper_bound() ? 1 : 0;
    }
    delete dm;
}
