/**
 * Copyright (c) 2016 Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *   
 * mex_helper.h
 **/

#ifndef _MEX_HELPER_H_
#define _MEX_HELPER_H_

#define MEX_ARGIN_IS_REAL_DOUBLE(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) == 1)
#define MEX_ARGIN_IS_REAL_VECTOR(arg_idx) (mxIsDouble(prhs[arg_idx]) && !mxIsComplex(prhs[arg_idx]) && ((mxGetN(prhs[arg_idx]) == 1 && mxGetM(prhs[arg_idx]) >= 1) || (mxGetN(prhs[arg_idx]) >= 1 && mxGetM(prhs[arg_idx]) == 1)))

#endif