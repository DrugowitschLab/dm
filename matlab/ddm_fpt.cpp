/**
 * Copyright (c) 2013, 2014 Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *
 * ddm_fpt.cpp - comuting the DDM first-passage time distribution as described in
 *               Smith (2000) "Stochastic Dynamic Models of Response Time and 
 *               Accurary: A Foundational Primer" and other sources. Both the
 *               drift rate and the (symmetric) bound can vary over time.
 *               A variant for weighted accumulation is also provided.
 *
 * [g1, g2] = ddm_fpt(mu, bound, delta_t, t_max, ...)
 *
 * mu and bound are vectors of drift rates and bound height over time, in steps
 * of delta_t. t_max is the maximum time up until which the first-passage time
 * distributions are evaluated. g1 and g2 hold the probability densities of
 * hitting the upper bound and lower bound, respectively, in steps of delta_t
 * up to and including t_max. If the vectors mu and bound are shorter than
 * t_max, their last elements replicated.
 *
 * The assumed model is
 *
 * dx / dt = mu(t) + eta(t)
 *
 * where eta is zero-mean unit variance white noise. The bound is on x and -x.
 *
 * The method uses more efficient methods of computing the first-passage time
 * density if either mu is constant (i.e. given as a scalar) or both mu and
 * the bound are constant.
 *
 *
 * [g1, g2] = ddm_fpt(a, bound, delta_t, t_max, k, ...)
 *
 * Performs weighted accumulation with weights given by vector a. k is a scalar
 * that determines the proportionality constant. The assumed model is
 *
 * dz / dt = k a(t) + eta(t)
 * dx / dt = a(t) dz / dt
 *
 * The bound is on x and -x.
 *
 * [g1, g2] = ddm_fpt(..., 'mnorm', 'yes')
 *
 * Causes both g1 and g2 to be normalised such that the densities integrate to
 * 1. The normalisation is performed by adding all missing mass to the last
 * element of g1 / g2, such that the proportion of the mass in g1 and g2
 * remains unchanged. This is useful if there is some significant portion of
 * the mass expected to occur after t_max. By default, 'mnorm' is set to 'no'. 
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
    /* [g1, g2] = ddm_fpt(mu, bound, delta_t, t_max) or
       [g1, g2] = ddm_fpt(a, k, bound, delta_t, t_max) */

    /* Check argument number */
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("ddm_fpt:WrongOutputs", 
                          "Wrong number of output arguments");
    }
    if (nrhs < 4) {
        mexErrMsgIdAndTxt("ddm_fpt:WrongInputs",
                          "Too few input arguments");
    }

    /* Process first 4 arguments */
    if (!MEX_ARGIN_IS_REAL_VECTOR(0))
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "First input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_VECTOR(1))
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "Second input argument expected to be a vector");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(2))
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "Third input argument expected to be a double");
    if (!MEX_ARGIN_IS_REAL_DOUBLE(3))
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "Forth input argument expected to be a double");
    int mu_size = std::max(mxGetN(prhs[0]), mxGetM(prhs[0]));
    int bound_size = std::max(mxGetN(prhs[1]), mxGetM(prhs[1]));
    ExtArray mu(ExtArray::shared_noowner(mxGetPr(prhs[0])), mu_size);
    ExtArray bound(ExtArray::shared_noowner(mxGetPr(prhs[1])), bound_size);
    double delta_t = mxGetScalar(prhs[2]);
    double t_max = mxGetScalar(prhs[3]);
    if (delta_t <= 0.0)
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "delta_t needs to be larger than 0.0");
    if (t_max <= delta_t)
        mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                          "t_max needs to be at least as large as delta_t");
    
    /* Process possible 5th non-string argument */
    bool weighted_ddm = false;
    int cur_argin = 4;
    double k = 0.0;
    if (nrhs > 4 && !mxIsChar(prhs[4])) {
        if (!MEX_ARGIN_IS_REAL_DOUBLE(4))
            mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                              "Fifth input argument expected to be a double");
        k = mxGetScalar(prhs[4]);
        weighted_ddm = true;
        ++cur_argin;
    }
    
    /* Process string arguments */
    bool normalise_mass = false;
    if (nrhs > cur_argin) {
        char str_arg[6];
        /* current only accept 'mnorm' string argument */
        if (!mxIsChar(prhs[cur_argin]))
            mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                              "String argument expected but not found");
        if (mxGetString(prhs[cur_argin], str_arg, sizeof(str_arg)) == 1 ||
            strcmp(str_arg, "mnorm") != 0)
            mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                              "\"mnorm\" string argument expected");
        /* this needs to be followed by "yes" or "no" */
        if (nrhs <= cur_argin + 1 || !mxIsChar(prhs[cur_argin + 1]))
            mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                              "String expected after \"mnorm\"");
        if (mxGetString(prhs[cur_argin + 1], str_arg, sizeof(str_arg)) == 1 ||
            (strcmp(str_arg, "yes") != 0 && strcmp(str_arg, "no") != 0))
            mexErrMsgIdAndTxt("ddm_fpt:WrongInput",
                              "\"yes\" or \"no\" expected after \"mnorm\"");
        normalise_mass = (strcmp(str_arg, "yes") == 0);
        
        /* no arguments allowed after that */
        if (nrhs > cur_argin + 2)
            mexErrMsgIdAndTxt("ddm_fpt:WrongInputs",
                              "Too many input arguments");
    }

    /* reserve space for output */
    int k_max = (int) ceil(t_max / delta_t);
    plhs[0] = mxCreateDoubleMatrix(1, k_max, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, k_max, mxREAL);
    ExtArray g1(ExtArray::shared_noowner(mxGetPr(plhs[0])), k_max);
    ExtArray g2(ExtArray::shared_noowner(mxGetPr(plhs[1])), k_max);

    /* compute fpt pdf */
    DMBase* dm = nullptr;
    if (weighted_ddm)
        dm = DMBase::createw(mu, bound, k, delta_t);
    else
        dm = DMBase::create(mu, bound, delta_t);
    dm->pdfseq(k_max, g1, g2);
    if (normalise_mass) dm->mnorm(g1, g2);
    delete dm;
}
