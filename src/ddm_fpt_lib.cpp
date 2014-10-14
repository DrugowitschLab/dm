/**
 * Copyright (c) 2013, 2014 Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *  
 * ddm_fpt_lib.cpp - Functions that compute the first-passage time distributions
 *                   of drift-diffusion models.
 **/

#include "ddm_fpt_lib.h"

#include <cassert>
#include <vector>

DMBase* DMBase::create(const ExtArray& drift, const ExtArray& bound,
                       value_t dt)
{
    if (drift.isconst()) {
        if (bound.isconst())
            return new DMConstDriftConstBound(drift[0], bound[0], dt);
        else
            return new DMConstDriftVarBound(drift[0], bound, dt);
    } else
        return new DMVarDriftVarBound(drift, bound, dt);
}


DMBase* DMBase::createw(const ExtArray& drift, const ExtArray& bound,
                        value_t k, value_t dt)
{
    return new DMWVarDriftVarBound(drift, bound, k, dt);
}


DMBase* DMBase::create(const ExtArray& drift, const ExtArray& sig2,
                       const ExtArray& b_lo, const ExtArray& b_up,
                       const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                       value_t dt, value_t invleak)
{
    const bool unit_sig2 = (sig2.isconst() <= 1 && sig2[0] == 1.0);
    const bool infinvleak = isinf(invleak);
    // TODO: add more specialised cases
    if (infinvleak)
        return new DMGeneralDeriv(drift, sig2, b_lo, b_up,
                                  b_lo_deriv, b_up_deriv, dt);
    else
        return new DMGeneralLeakDeriv(drift, sig2, b_lo, b_up,
                                      b_lo_deriv, b_up_deriv, invleak, dt);
}


void DMConstDriftConstBound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    const value_t c1 = 4 * (bound * bound);
    const value_t c2 = (drift * drift) / 2;
    const value_t c3 = drift * bound;
    const value_t c4 = exp(-2 * c3);

    value_t t = dt;
    for (int i = 0; i < n; ++i) {
        const value_t g = fpt_symup(t, c1, c2, c3);
        g1[i] = std::max(g, 0.0);
        g2[i] = std::max(c4 * g, 0.0);
        t += dt;
    }
}


// fpt_symseries - series expansion for fpt lower density, symmetric bounds
DMBase::value_t DMConstDriftConstBound::fpt_symseries(value_t t, value_t a,
                                                      value_t b, value_t tol)
{
    tol *= b;
    double f = exp(-a);
    int twok = 3;
    while (1) {
        double incr = twok * exp(- (twok * twok) * a);
        f -= incr;
        if (incr < tol)
            return f * b;
        twok += 2;
        incr = twok * exp(- (twok * twok) * a);
        f += incr;
        if (incr < tol)
            return f * b;
        twok += 2;
    }
}


void DMConstDriftConstABound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    const value_t bdiff = b_up - b_lo;
    const value_t c1 = bdiff * bdiff;
    const value_t c2 = drift * drift / 2;
    const value_t c3 = drift * b_up;
    const value_t c4 = drift * b_lo;
    const value_t w = - b_lo / bdiff;

    value_t t = dt;
    for (int i = 0; i < n; ++i) {
        g1[i] = std::max(fpt_asymup(t, c1, c2, c3, w), 0.0);
        g2[i] = std::max(fpt_asymlo(t, c1, c2, c4, w), 0.0);
        t += dt;
    }
}


// series expansion for fpt for short t, Navarro & Fuss (2009), Eq. (6) 
DMBase::value_t DMConstDriftConstABound::fpt_asymshortt(value_t t, value_t w, value_t tol)
{
    const value_t b = pow(t, -1.5) / sqrt(TWOPI);
    tol *= b;
    t *= 2;
    size_t k = 1;
    value_t f = w * exp(-w * w / t);
    while (1) {
        value_t c = w + 2 * k;
        value_t incr = c * exp(-c * c / t);
        f += incr;
        if (fabs(incr) < tol)
            return f * b;
        c = w - 2 * k;
        incr = c * exp(-c * c / t);
        f += incr;
        if (fabs(incr) < tol)
            return f * b;
        k += 1;
    }
}


// series expansion for fpt for long t, Navarro & Fuss (2009), Eq. (5)
DMBase::value_t DMConstDriftConstABound::fpt_asymlongt(value_t t, value_t w, value_t tol)
{
    tol *= PI;
    value_t f = 0.0;
    size_t k = 1;
    while (1) {
        const value_t kpi = k * PI;
        value_t incr = k * exp(- (kpi * kpi) * t / 2) * sin(kpi * w);
        f += incr;
        if (fabs(incr) < tol)
            return f * PI;
        k += 1;
    }
}


void DMConstDriftVarBound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    /* precompute some constants */
    const double dt_2 = 2.0 * dt;
    const double pi_dt_2 = PI * dt_2;
    const double drift_dt = dt * drift;
    const double drift_2 = -2 * drift;

    /* derivative of bound */
    std::vector<double> bound_deriv(n);
    for (int j = 1; j < n; ++j) {
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / dt;
    }
    bound_deriv[n - 1] = bound_deriv[n - 2];

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * dt * (i + 1)) 
       norm_t[i] = 1 / (dt * (i + 1)) */
    std::vector<double> norm_sqrt_t(n);
    std::vector<double> norm_t(n);
    for (int j = 0; j < n; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_dt_2 * (j + 1.0));
        norm_t[j] = 1.0 / (dt * (j + 1.0));
    }

    /* fill g1 recursively, g2 is based on g1 */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double bound_k = bound[k];
        const double bound_deriv_k1 = bound_deriv[k] - drift;
        const double bound_deriv_k2 = -bound_deriv[k] - drift;
        const double cum_drift_k = (k + 1) * drift_dt;
        const double norm_t_j = norm_t[k];
        const double norm_sqrt_t_j = norm_sqrt_t[k];
        /* initial values */
        double g1_k = -norm_sqrt_t_j
                      * exp(- 0.5 * (bound_k - cum_drift_k) * (bound_k - cum_drift_k) * norm_t_j)
                      * (bound_deriv_k1 - (bound_k - cum_drift_k) * norm_t_j);
        /* relation to previous values */
        for (int j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            const double bound_j = bound[j];
            const double cum_drift_k_j = (k - j) * drift_dt;
            const double diff1 = bound_k - bound_j - cum_drift_k_j;
            const double diff2 = bound_k + bound_j - cum_drift_k_j;
            const double norm_t_j = norm_t[k - j - 1];
            const double norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += dt * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff1 * diff1 * norm_t_j)
                       * (bound_deriv_k1 - diff1 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff2 * diff2 * norm_t_j)
                       * (bound_deriv_k2 - diff2 * norm_t_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = std::max(g1_k, 0.0);
        g2[k] = std::max(g1_k * exp(drift_2 * bound_k), 0.0);
    }
}


void DMVarDriftVarBound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    /* precompute some constants */
    const double dt_2 = 2.0 * dt;
    const double pi_dt_2 = PI * dt_2;
    
    /* cumulative drift, and derivative of bound */
    std::vector<double> cum_drift(n);
    std::vector<double> bound_deriv(n);
    double curr_cum_drift = dt * drift[0];
    cum_drift[0] = curr_cum_drift;
    for (int j = 1; j < n; ++j) {
        curr_cum_drift += dt * drift[j];
        cum_drift[j] = curr_cum_drift;
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / dt;
    }
    bound_deriv[n - 1] = bound_deriv[n - 2];

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * dt * (i + 1)) 
       norm_t[i] = 1 / (dt * (i + 1)) */
    std::vector<double> norm_sqrt_t(n);
    std::vector<double> norm_t(n);
    for (int j = 0; j < n; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_dt_2 * (j + 1.0));
        norm_t[j] = 1.0 / (dt * (j + 1.0));
    }

    /* fill up g1 and g2 recursively */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double bound_k = bound[k];
        const double bound_deriv_k1 = bound_deriv[k] - drift[k];
        const double bound_deriv_k2 = -bound_deriv[k] - drift[k];
        const double cum_drift_k = cum_drift[k];
        const double norm_t_j = norm_t[k];
        const double norm_sqrt_t_j = norm_sqrt_t[k];
        /* initial values */
        double g1_k = -norm_sqrt_t_j
                      * exp(- 0.5 * (bound_k - cum_drift_k) * (bound_k - cum_drift_k) * norm_t_j)
                      * (bound_deriv_k1 - (bound_k - cum_drift_k) * norm_t_j);
        double g2_k = norm_sqrt_t_j
                      * exp(- 0.5 * (-bound_k - cum_drift_k) * (-bound_k - cum_drift_k) * norm_t_j)
                      * (bound_deriv_k2 - (-bound_k - cum_drift_k) * norm_t_j);
        /* relation to previous values */
        for (int j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            const double bound_j = bound[j];
            const double cum_drift_k_j = cum_drift_k - cum_drift[j];
            const double diff11 = bound_k - bound_j - cum_drift_k_j;
            const double diff12 = bound_k + bound_j - cum_drift_k_j;
            const double norm_t_j = norm_t[k - j - 1];
            const double norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += dt * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff11 * diff11 * norm_t_j)
                       * (bound_deriv_k1 - diff11 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff12 * diff12 * norm_t_j)
                       * (bound_deriv_k1 - diff12 * norm_t_j));
            const double diff21 = -bound_k - bound_j - cum_drift_k_j;
            const double diff22 = -bound_k + bound_j - cum_drift_k_j;
            g2_k -= dt * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff21 * diff21 * norm_t_j)
                       * (bound_deriv_k2 - diff21 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff22 * diff22 * norm_t_j)
                       * (bound_deriv_k2 - diff22 * norm_t_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = std::max(g1_k, 0.0);
        g2[k] = std::max(g2_k, 0.0);
    }
}


void DMWVarDriftVarBound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    /* pre-compute value */
    const double k_2 = -2 * k;

    /* a2(t) = drift(t)^2, A_t(t) = \int^t a2(s) ds, and derivative of bound */
    std::vector<double> a2(n);
    std::vector<double> A(n);
    std::vector<double> bound_deriv(n);
    double cum_a2 = dt * (a2[0] = drift[0] * drift[0]);
    A[0] = cum_a2;
    for (int j = 1; j < n; ++j) {
        cum_a2 += dt * (a2[j] = drift[j] * drift[j]);
        A[j] = cum_a2;
        bound_deriv[j - 1] = (bound[j] - bound[j - 1]) / dt;
    }
    bound_deriv[n - 1] = bound_deriv[n - 2];

    /* fill up g1 and g2 recursively */
    for (int i = 0; i < n; ++i) {
        /* reduce array access */
        const double bound_i = bound[i];
        const double a2_i = a2[i];
        const double A_i = A[i];
        const double bound_deriv_i = bound_deriv[i];

        /* initial values */
        const double diff1 = bound_i - k * A_i;
        const double sqrt_A_i = sqrt(TWOPI * A_i);
        const double tmp = bound_deriv_i - bound_i / A_i * a2_i;
        double g1_i = - exp(-0.5 * diff1 * diff1 / A_i) / sqrt_A_i * tmp;

        /* relation to previous values */
        for (int j = 0; j < i; ++j) {
            /* reduce array access and pre-compute values */
            const double bound_j = bound[j];
            const double A_diff = A_i - A[j];
            const double sqrt_A_diff = sqrt(TWOPI * A_diff);
            const double diff1 = bound_i - bound_j;
            const double diff2 = bound_i + bound_j;
            const double diff1_A = diff1 - k * A_diff;
            const double diff2_A = diff2 - k * A_diff;
            g1_i += dt / sqrt_A_diff * (
                        g1[j] * exp(-0.5 * diff1_A * diff1_A / A_diff) 
                        * (bound_deriv_i - a2_i * diff1 / A_diff)
                      + g2[j] * exp(-0.5 * diff2_A * diff2_A / A_diff)
                        * (bound_deriv_i - a2_i * diff2 / A_diff));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[i] = std::max(g1_i, 0.0);
        g2[i] = std::max(g1_i * exp(k_2 * bound_i), 0.0);
    }
}


void DMGeneralDeriv::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);
    
    /* precompute some constants */
    const double sqrt_2_pi = 1 / sqrt(2 * PI);
    const double dt_sqrt_2_pi = dt * sqrt_2_pi;
    
    /* cumulative mu and sig2 */
    std::vector<double> cum_drift(n);
    std::vector<double> cum_sig2(n);
    double curr_cum_drift = dt * drift[0];
    cum_drift[0] = curr_cum_drift;
    double curr_cum_sig2 = dt * sig2[0];
    cum_sig2[0] = curr_cum_sig2;
    for (int j = 1; j < n; ++j) {
        curr_cum_drift += dt * drift[j];
        cum_drift[j] = curr_cum_drift;
        curr_cum_sig2 += dt * sig2[j];
        cum_sig2[j] = curr_cum_sig2;
    }
    
    /* fill up g1 and g2 recursively */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double sig2_k = sig2[k];
        const double b_up_k = b_up[k];
        const double b_lo_k = b_lo[k];
        const double cum_drift_k = cum_drift[k];
        const double cum_sig2_k = cum_sig2[k];
        const double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        const double b_up_deriv_k = b_up_deriv[k] - drift[k];
        const double b_lo_deriv_k = b_lo_deriv[k] - drift[k];
        
        /* initial values */
        double g1_k = -sqrt_2_pi / sqrt_cum_sig2_k * 
                      exp(-0.5 * (b_up_k - cum_drift_k) * (b_up_k - cum_drift_k) / 
                          cum_sig2_k) *
                      (b_up_deriv_k - sig2_k * (b_up_k - cum_drift_k) / cum_sig2_k);
        double g2_k = sqrt_2_pi / sqrt_cum_sig2_k *
                      exp(-0.5 * (b_lo_k - cum_drift_k) * (b_lo_k - cum_drift_k) /
                          cum_sig2_k) *
                      (b_lo_deriv_k - sig2_k * (b_lo_k - cum_drift_k) / cum_sig2_k);
        /* relation to previous values */
        for (int j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            const double cum_sig2_diff_j = cum_sig2_k - cum_sig2[j];
            const double sqrt_cum_sig2_diff_j = sqrt(cum_sig2_diff_j);
            const double cum_drift_diff_j = cum_drift[j] - cum_drift_k;
            const double b_up_k_up_j_diff = b_up_k - b_up[j] + cum_drift_diff_j;
            const double b_up_k_lo_j_diff = b_up_k - b_lo[j] + cum_drift_diff_j;
            const double b_lo_k_up_j_diff = b_lo_k - b_up[j] + cum_drift_diff_j;
            const double b_lo_k_lo_j_diff = b_lo_k - b_lo[j] + cum_drift_diff_j;
            /* add values */
            g1_k += dt_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_up_k_up_j_diff * b_up_k_up_j_diff / 
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_up_k_lo_j_diff * b_up_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_lo_j_diff / cum_sig2_diff_j));
            g2_k -= dt_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_lo_k_up_j_diff * b_lo_k_up_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_lo_k_lo_j_diff * b_lo_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_lo_j_diff / cum_sig2_diff_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = std::max(g1_k, 0.0);
        g2[k] = std::max(g2_k, 0.0);
    }
}


void DMGeneralLeakDeriv::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);
    
    /* precompute some constants */
    const double sqrt_2_pi = 1 / sqrt(2 * PI);
    const double dt_sqrt_2_pi = dt * sqrt_2_pi;
    const double exp_leak = exp(- dt * invleak);
    const double exp2_leak = exp(- 2 * dt * invleak);
    
    /* cumulative mu and sig2, and discount (leak) */
    std::vector<double> cum_drift(n);
    std::vector<double> cum_sig2(n);
    std::vector<double> disc(n);
    double curr_cum_drift = dt * drift[0];
    cum_drift[0] = curr_cum_drift;
    double curr_cum_sig2 = dt * sig2[0];
    cum_sig2[0] = curr_cum_sig2;
    double curr_disc = exp_leak;
    disc[0] = curr_disc;
    for (int j = 1; j < n; ++j) {
        curr_cum_drift = exp_leak * curr_cum_drift + dt * drift[j];
        cum_drift[j] = curr_cum_drift;
        curr_cum_sig2 = exp2_leak * curr_cum_sig2 + dt * sig2[j];
        cum_sig2[j] = curr_cum_sig2;
        curr_disc *= exp_leak;
        disc[j] = curr_disc;
    }
    /* double discount (leak),
     * note that  disc[k - j - 1] = exp(- invleak dt (k - j))
     *           disc2[k - j - 1] = exp(- 2 invleak dt (k - j))
     * such that half of disc can be used to compute disc2 */
    std::vector<double> disc2(n);
    int k = (int) floor(((double) (n - 1)) / 2);
    for (int j = 0; j <= k; ++j)
        disc2[j] = disc[2 * j + 1];
    curr_disc = disc2[k];
    for (int j = k + 1; j < n; ++j) {
        curr_disc *= exp2_leak;
        disc2[j] = curr_disc;
    }
    
    /* fill up g1 and g2 recursively */
    for (k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double sig2_k = sig2[k];
        const double b_up_k = b_up[k];
        const double b_lo_k = b_lo[k];
        const double cum_drift_k = cum_drift[k];
        const double cum_sig2_k = cum_sig2[k];
        const double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        const double b_up_deriv_k = b_up_deriv[k] + invleak * b_up_k - drift[k];
        const double b_lo_deriv_k = b_lo_deriv[k] + invleak * b_lo_k - drift[k];
        
        /* initial values */
        double g1_k = -sqrt_2_pi / sqrt_cum_sig2_k * 
                      exp(-0.5 * (b_up_k - cum_drift_k) * (b_up_k - cum_drift_k) / 
                          cum_sig2_k) *
                      (b_up_deriv_k - sig2_k * (b_up_k - cum_drift_k) / cum_sig2_k);
        double g2_k = sqrt_2_pi / sqrt_cum_sig2_k *
                      exp(-0.5 * (b_lo_k - cum_drift_k) * (b_lo_k - cum_drift_k) /
                          cum_sig2_k) *
                      (b_lo_deriv_k - sig2_k * (b_lo_k - cum_drift_k) / cum_sig2_k);
        /* relation to previous values */
        for (int j = 0; j < k; ++j) {
            /* reducing array access + pre-compute values */
            const double disc_j = disc[k - j - 1];
            const double cum_sig2_diff_j = cum_sig2_k - disc2[k - j - 1] * cum_sig2[j];
            const double sqrt_cum_sig2_diff_j = sqrt(cum_sig2_diff_j);
            const double cum_drift_diff_j = disc_j * cum_drift[j] - cum_drift_k;
            const double b_up_k_up_j_diff = b_up_k - disc_j * b_up[j] + cum_drift_diff_j;
            const double b_up_k_lo_j_diff = b_up_k - disc_j * b_lo[j] + cum_drift_diff_j;
            const double b_lo_k_up_j_diff = b_lo_k - disc_j * b_up[j] + cum_drift_diff_j;
            const double b_lo_k_lo_j_diff = b_lo_k - disc_j * b_lo[j] + cum_drift_diff_j;
            /* add values */
            g1_k += dt_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_up_k_up_j_diff * b_up_k_up_j_diff / 
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_up_k_lo_j_diff * b_up_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_up_deriv_k - 
                      sig2_k * b_up_k_lo_j_diff / cum_sig2_diff_j));
            g2_k -= dt_sqrt_2_pi / sqrt_cum_sig2_diff_j *
                    (g1[j] * exp(-0.5 * b_lo_k_up_j_diff * b_lo_k_up_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_up_j_diff / cum_sig2_diff_j) +
                     g2[j] * exp(-0.5 * b_lo_k_lo_j_diff * b_lo_k_lo_j_diff /
                                 cum_sig2_diff_j) *
                     (b_lo_deriv_k -
                      sig2_k * b_lo_k_lo_j_diff / cum_sig2_diff_j));
        }
        /* avoid negative densities that could appear due to numerical instab. */
        g1[k] = std::max(g1_k, 0.0);
        g2[k] = std::max(g2_k, 0.0);
    }
}


/** normalising the mass, such that (sum(g1) + sum(g2) * delta_t = 1 
 *
 * Function makes sure that g1(t) >= 0, g2(t) >= 0, for all t, and that
 * (sum(g1) + sum(g2) * delta_t) = 1. It does so by eventually adding mass to
 * the last elements of g1 / g2, such that the ratio
 * sum(g1) / (sum(g1) + sum(g2)) (after removing negative values) remains
 * unchanged.
 */
void mnorm(double g1[], double g2[], int n, double delta_t)
{
    /* remove negative elements and compute sum */
    double g1_sum = 0.0, g2_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (g1[i] < 0) g1[i] = 0;
        else g1_sum += g1[i];
        if (g2[i] < 0) g2[i] = 0;
        else g2_sum += g2[i];
    }
    
    /* adjust last elements accoring to ratio */
    double p = g1_sum / (g1_sum + g2_sum);
    g1[n - 1] += p / delta_t - g1_sum;
    g2[n - 1] += (1 - p) / delta_t - g2_sum;
}
