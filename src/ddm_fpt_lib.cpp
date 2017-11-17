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


ExtArray ExtArray::cumsum(value_t f, size_t data_size_out) const
{
    value_t* x = new value_t[data_size_out];
    if (data_size_out > 0) {
        value_t cursum = f * (*this)[0];
        x[0] = cursum;
        for (size_t i = 1; i < data_size_out; ++i) {
            cursum += f * (*this)[i];
            x[i] = cursum;
        }
        return ExtArray(shared_owner(x), cursum, data_size_out);
    } else
        return ExtArray(shared_owner(x), 0, 0);
}


ExtArray ExtArray::deriv(value_t dt) const
{
    if (isconst()) return const_array(0.0);
    if (data_size_ == 1) {
        const value_t dx = (last_ - data_.get()[0]) / (2 * dt);
        value_t* x = new value_t[2];
        x[0] = dx;   /* left and right finite differences */
        x[1] = dx;
        return ExtArray(shared_owner(x), 0.0, 2);
    } else {
        const value_t dt2 = 2 * dt;
        value_t* x = new value_t[data_size_ + 1];
        /* left/right finite difference for edges, otherwise central diff */
        x[0] = (data_.get()[1] - data_.get()[0]) / dt;
        for (size_t i = 1; i < data_size_-1; ++i)
            x[i] = (data_.get()[i + 1] - data_.get()[i - 1]) / dt2;
        if (data_size_ >= 3)
            x[data_size_ - 1] = (last_ - data_.get()[data_size_ - 2]) / dt2;
        x[data_size_] = (last_ - data_.get()[data_size_-1]) / dt;
        return ExtArray(shared_owner(x), 0.0, data_size_ + 1);
    }
}


FastDMSamplingBase* FastDMSamplingBase::create(FastDMSamplingBase::value_t drift)
{   
    if (drift >= 1.0)
        return new FastDMSamplingInvNorm(drift);
    else 
        return new FastDMSamplingNormExp(drift);
}


bool FastDMSamplingBase::acceptt(value_t t, value_t zf, value_t c2)
{
    assert(c2 > 0.06385320297074884); // log(5/3) / 16, req. for convergence
    value_t b = exp(-c2);
    int twok = 3;
    while (true) {
        if (zf >= b) return false;     // above upper bound
        b -= twok * exp(-c2 * (twok * twok));
        if (zf <= b) return true;      // below lower bound
        twok += 2;
        b += twok * exp(-c2 * (twok * twok));
        twok += 2;
    }
}


FastDMSamplingBase::value_t FastDMSamplingNormExp::rand(rngeng_t& rngeng)
{
    std::uniform_real_distribution<double> unif_dist;
    while (true) {
        const value_t P = F1inf_ * unif_dist(rngeng);
        if (P <= CF1st_) {
            // short-time series
            const value_t erfcinvP = erfcinv(P / Cf1s_);
            const value_t t = 1 / (2 * a_ * erfcinvP * erfcinvP);
            if (acceptt(t, exp(- 0.5/(a_ * t) - sqrtamu_ + 
                               mu2_ * t / 2) * unif_dist(rngeng),
                        0.5 / t)) 
                return t;
        } else {
            // long-time series
            const value_t t = -log1p(- (P - CF1st_) / Cf1l_ - F1lt_) / fourmu2pi_;
            const value_t pi2t8 = PI * PI * t / 8;
            if (acceptt(t, exp(-pi2t8) * unif_dist(rngeng), pi2t8)) return t;
        }
    }
}


FastDMSamplingBase::value_t FastDMSamplingNormExp::erfcinv(value_t P)
{
    // transforming from inv ccdf of standard normal to inv erfc
    if (P <= 1) return -0.7071067811865475 * 
        rational_invccdf_approx(sqrt(-2.0*log(0.5 * P)));
    else return 0.7071067811865475 * 
        rational_invccdf_approx(sqrt(-2.0*log(1.0-0.5*P)));
}


// public domain code from http://www.johndcook.com/cpp_phi_inverse.html
FastDMSamplingBase::value_t FastDMSamplingNormExp::rational_invccdf_approx(value_t t)
{
    // Abramowitz and Stegun formula 26.2.23.
    // The absolute value of the error should be less than 4.5 e-4.
    double c[] = {2.515517, 0.802853, 0.010328};
    double d[] = {1.432788, 0.189269, 0.001308};
    return t - ((c[2]*t + c[1])*t + c[0]) / 
               (((d[2]*t + d[1])*t + d[0])*t + 1.0);
}


FastDMSamplingBase::value_t FastDMSamplingInvNorm::rand(rngeng_t& rngeng)
{
    std::uniform_real_distribution<double> unif_dist;
    while (true) {
        const value_t t = randin(rngeng, invabsmu_, invmu2_);
        const value_t one2t = 0.5 / t;
        if (t < 2.5) {
            // short-time series
            if (acceptt(t, exp(-one2t) * unif_dist(rngeng), one2t)) return t;
        } else {
            // long-time series
            constexpr value_t Cl = -0.6773740579341821; // -log(pi/4)-log(2pi)/2;
            if (acceptt(t, exp(Cl - one2t - 3 / 2 * log(t)) * unif_dist(rngeng), 
                        PI * PI * t / 8)) return t;
        }
    }
}


FastDMSamplingBase::value_t FastDMSamplingInvNorm::randin(rngeng_t& rngeng, 
    value_t mu, value_t mu2)
{
    std::normal_distribution<value_t> randn;
    std::uniform_real_distribution<double> unif_dist;
    const value_t z = randn(rngeng);
    const value_t y = z * z;
    const value_t x = mu + (mu2 * y - mu * sqrt((4 * mu + mu2 * y) * y)) / 2;
    return unif_dist(rngeng) <= 1 / (1 + x / mu) ? x : mu2 / x;
}


DMBase::value_t DMBase::pdfu(value_t t)
{
    if (t == 0.0) return 0.0;
    const value_t n = t / dt_;
    const size_t nup = static_cast<int>(ceil(n));
    ExtArray g1(nup), g2(nup);
    pdfseq(nup, g1, g2);
    return lininterp(nup == 1 ? 0.0 : g1[nup-2], g1[nup-1], n - nup + 1.0);
}


DMBase::value_t DMBase::pdfl(value_t t)
{
    if (t == 0.0) return 0.0;
    const value_t n = t / dt_;
    const size_t nup = static_cast<int>(ceil(n));
    ExtArray g1(nup), g2(nup);
    pdfseq(nup, g1, g2);
    return lininterp(nup == 1 ? 0.0 : g2[nup-2], g2[nup-1], n - nup + 1.0);
}


// generic Euler-Maruyama Method implementation, assuming sig2=1, always
DMSample DMBase::rand(rngeng_t& rngeng)
{
    std::normal_distribution<value_t> randn;
    value_t x = drift(0) * dt_ + sqrt_dt_ * randn(rngeng);
    size_t n = 1;
    int cb = crossed_bounds(x, 1);
    while (!cb) {
        x += drift(n) * dt_ + sqrt_dt_ * randn(rngeng);
        n += 1;
        cb = crossed_bounds(x, n);
    }
    return DMSample(n * dt_, cb == 1);
}


void DMBase::mnorm(ExtArray& g1, ExtArray& g2) const
{
    const size_t n = std::max(g1.size(), g2.size());
    /* remove negative elements and compute sum */
    value_t g1_sum = 0.0, g2_sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (g1[i] < 0) g1[i] = 0;
        else g1_sum += g1[i];
        if (g2[i] < 0) g2[i] = 0;
        else g2_sum += g2[i];
    }
    
    /* adjust last elements accoring to ratio */
    double p = g1_sum / (g1_sum + g2_sum);
    g1[n - 1] += p / dt_ - g1_sum;
    g2[n - 1] += (1 - p) / dt_ - g2_sum;
}


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
    const bool infinvleak = std::isinf(invleak);
    if (infinvleak) {
        const bool unit_sig2 = (sig2.isconst() && sig2[0] == 1.0);
        const bool constbounds = (b_lo.isconst() && b_up.isconst());
        if (unit_sig2 && constbounds) {
            const bool symconstbounds = (constbounds && b_up[0] == -b_up[0]);
            if (drift.isconst()) {
                if (symconstbounds)
                    // const drift, const sym bounds, unit variance
                    return new DMConstDriftConstBound(drift[0], b_up[0], dt);
                else
                    // const drift, const asym bounds, unit variance
                    return new DMConstDriftConstABound(drift[0], b_lo[0], b_up[0], dt);
            } else {
                if (symconstbounds)
                    // var drift, const sym bounds, unit variance
                    return new DMVarDriftVarBound(drift, b_up, dt);
                else
                    // var drift, const asym bounds, unit variance
                    return new DMGeneralDeriv(drift, sig2, b_lo, b_up,
                                              b_lo_deriv, b_up_deriv, dt);
            }
        }
        // general case, no leak
        return new DMGeneralDeriv(drift, sig2, b_lo, b_up,
                                  b_lo_deriv, b_up_deriv, dt);
    } else
        // general case, leak
        return new DMGeneralLeakDeriv(drift, sig2, b_lo, b_up,
                                      b_lo_deriv, b_up_deriv, invleak, dt);
}


void DMConstDriftConstBound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    compute_dm_consts();
    const value_t c1 = dm_consts_->c1();
    const value_t c2 = dm_consts_->c2();
    const value_t c3 = dm_consts_->c3();
    const value_t c4 = dm_consts_->c4();

    value_t t = dt_;
    for (int i = 0; i < n; ++i) {
        const value_t g = fpt_symup(t, c1, c2, c3);
        g1[i] = std::max(g, 0.0);
        g2[i] = std::max(c4 * g, 0.0);
        t += dt_;
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


DMSample DMConstDriftConstBound::rand(rngeng_t& rngeng)
{
    compute_dm_consts();
    if (!fpt_sampler_) {
        // initialise sampler at first use
        const double smu = fabs(dm_consts_->c3());
        fpt_sampler_.reset(FastDMSamplingBase::create(smu));
    }
    // use sampler with time re-scaling for non-unit bounds
    double t = fpt_sampler_->rand(rngeng);
    std::uniform_real_distribution<double> unif_dist;
    return DMSample(t * dm_consts_->c1() / 4, 
                    unif_dist(rngeng) <= dm_consts_->c5());
}


void DMConstDriftConstABound::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);

    compute_pdf_consts();
    const value_t c1 = pdf_consts_->c1();
    const value_t c2 = pdf_consts_->c2();
    const value_t c3 = pdf_consts_->c3();
    const value_t c4 = pdf_consts_->c4();
    const value_t w = pdf_consts_->w();

    value_t t = dt_;
    for (int i = 0; i < n; ++i) {
        g1[i] = std::max(fpt_asymup(t, c1, c2, c3, w), 0.0);
        g2[i] = std::max(fpt_asymlo(t, c1, c2, c4, w), 0.0);
        t += dt_;
    }
}


DMSample DMConstDriftConstABound::rand(rngeng_t& rngeng)
{
    value_t t = 0.0;
    value_t x = 0.0;
    std::uniform_real_distribution<double> unif_dist;
    while (true) {
        const double xlo = fabs(x - b_lo_);
        const double xup = fabs(x - b_up_);
        if (fabs(xlo - xup) < 1e-20) {
            // symmetric bounds, diffusion model in [x - xup, x + xup]
            const double mutheta = xup * drift_;
            FastDMSamplingBase* fpt_sampler = FastDMSamplingBase::create(mutheta);
            t += xup * xup * fpt_sampler->rand(rngeng);
            delete fpt_sampler;
            return DMSample(t, unif_dist(rngeng) < 1 / (1 + exp(-2 * mutheta)));
        } else if (xlo > xup) {
            // x closer to upper bound, diffusion model in [x - xup, x + xup]
            const double mutheta = xup * drift_;
            FastDMSamplingBase* fpt_sampler = FastDMSamplingBase::create(mutheta);
            t += xup * xup * fpt_sampler->rand(rngeng);
            delete fpt_sampler;
            if (unif_dist(rngeng) < 1 / (1 + exp(-2 * mutheta)))
                return DMSample(t, true);
            x -= xup;
        } else {
            // x closer to lower bound, diffusion model in [x - xlo, x + xlo]
            const double mutheta = xlo * drift_;
            FastDMSamplingBase* fpt_sampler = FastDMSamplingBase::create(mutheta);
            t += xlo * xlo * fpt_sampler->rand(rngeng);
            delete fpt_sampler;
            if (unif_dist(rngeng) > 1 / (1 + exp(-2 * mutheta)))
                return DMSample(t, false);
            x += xlo;
        }
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

    /* precompute some constants / derivatives */
    const double dt_2 = 2.0 * dt_;
    const double pi_dt_2 = PI * dt_2;
    const double drift_dt = dt_ * drift_;
    const double drift_2 = -2 * drift_;
    auto bound_deriv = bound_.deriv(dt_);

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * dt * (i + 1)) 
       norm_t[i] = 1 / (dt * (i + 1)) */
    std::vector<double> norm_sqrt_t(n);
    std::vector<double> norm_t(n);
    for (int j = 0; j < n; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_dt_2 * (j + 1.0));
        norm_t[j] = 1.0 / (dt_ * (j + 1.0));
    }

    /* fill g1 recursively, g2 is based on g1 */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double bound_k = bound_[k];
        const double bound_deriv_k1 = bound_deriv[k] - drift_;
        const double bound_deriv_k2 = -bound_deriv[k] - drift_;
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
            const double bound_j = bound_[j];
            const double cum_drift_k_j = (k - j) * drift_dt;
            const double diff1 = bound_k - bound_j - cum_drift_k_j;
            const double diff2 = bound_k + bound_j - cum_drift_k_j;
            const double norm_t_j = norm_t[k - j - 1];
            const double norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += dt_ * norm_sqrt_t_j
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
    const double dt_2 = 2.0 * dt_;
    const double pi_dt_2 = PI * dt_2;
    
    /* cumulative drift, and derivative of bound */
    auto cum_drift = drift_.cumsum(dt_, n);
    auto bound_deriv = bound_.deriv(dt_);

    /* norm_sqrt_t[i] = 1 / sqrt(2 * pi * dt * (i + 1)) 
       norm_t[i] = 1 / (dt * (i + 1)) */
    std::vector<double> norm_sqrt_t(n);
    std::vector<double> norm_t(n);
    for (int j = 0; j < n; ++j) {
        norm_sqrt_t[j] = 1.0 / sqrt(pi_dt_2 * (j + 1.0));
        norm_t[j] = 1.0 / (dt_ * (j + 1.0));
    }

    /* fill up g1 and g2 recursively */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double bound_k = bound_[k];
        const double bound_deriv_k1 = bound_deriv[k] - drift_[k];
        const double bound_deriv_k2 = -bound_deriv[k] - drift_[k];
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
            const double bound_j = bound_[j];
            const double cum_drift_k_j = cum_drift_k - cum_drift[j];
            const double diff11 = bound_k - bound_j - cum_drift_k_j;
            const double diff12 = bound_k + bound_j - cum_drift_k_j;
            const double norm_t_j = norm_t[k - j - 1];
            const double norm_sqrt_t_j = norm_sqrt_t[k - j - 1];
            /* add values */
            g1_k += dt_ * norm_sqrt_t_j
                    * (g1[j] * exp(- 0.5 * diff11 * diff11 * norm_t_j)
                       * (bound_deriv_k1 - diff11 * norm_t_j)
                       + g2[j] * exp(- 0.5 * diff12 * diff12 * norm_t_j)
                       * (bound_deriv_k1 - diff12 * norm_t_j));
            const double diff21 = -bound_k - bound_j - cum_drift_k_j;
            const double diff22 = -bound_k + bound_j - cum_drift_k_j;
            g2_k -= dt_ * norm_sqrt_t_j
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
    const double k_2 = -2 * k_;

    /* a2(t) = drift(t)^2, A_t(t) = \int^t a2(s) ds, and derivative of bound */
    std::vector<double> a2(n);
    std::vector<double> A(n);
    double cum_a2 = dt_ * (a2[0] = drift_[0] * drift_[0]);
    A[0] = cum_a2;
    for (int j = 1; j < n; ++j) {
        cum_a2 += dt_ * (a2[j] = drift_[j] * drift_[j]);
        A[j] = cum_a2;
    }
    auto bound_deriv = bound_.deriv(dt_);

    /* fill up g1 and g2 recursively */
    for (int i = 0; i < n; ++i) {
        /* reduce array access */
        const double bound_i = bound_[i];
        const double a2_i = a2[i];
        const double A_i = A[i];
        const double bound_deriv_i = bound_deriv[i];

        /* initial values */
        const double diff1 = bound_i - k_ * A_i;
        const double sqrt_A_i = sqrt(TWOPI * A_i);
        const double tmp = bound_deriv_i - bound_i / A_i * a2_i;
        double g1_i = - exp(-0.5 * diff1 * diff1 / A_i) / sqrt_A_i * tmp;

        /* relation to previous values */
        for (int j = 0; j < i; ++j) {
            /* reduce array access and pre-compute values */
            const double bound_j = bound_[j];
            const double A_diff = A_i - A[j];
            const double sqrt_A_diff = sqrt(TWOPI * A_diff);
            const double diff1 = bound_i - bound_j;
            const double diff2 = bound_i + bound_j;
            const double diff1_A = diff1 - k_ * A_diff;
            const double diff2_A = diff2 - k_ * A_diff;
            g1_i += dt_ / sqrt_A_diff * (
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


DMSample DMWVarDriftVarBound::rand(rngeng_t& rngeng)
{
    // dx = mu(t) (k mu(t) dt + dW) .
    std::normal_distribution<value_t> randn;
    value_t x = drift(0) * (k_ * drift(0) * dt_ + sqrt_dt_ * randn(rngeng));
    size_t n = 1;
    int cb = crossed_bounds(x, 1);
    while (!cb) {
        const value_t drift_n = drift(n);
        x += drift_n * (k_ * drift_n * dt_ + sqrt_dt_ * randn(rngeng));
        n += 1;
        cb = crossed_bounds(x, n);
    }
    return DMSample(n * dt_, cb == 1);
}


void DMGeneralDeriv::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);
    
    /* precompute some constants and cumulatives */
    const double sqrt_2_pi = 1 / sqrt(2 * PI);
    const double dt_sqrt_2_pi = dt_ * sqrt_2_pi;
    auto cum_drift = drift_.cumsum(dt_, n);
    auto cum_sig2 = sig2_.cumsum(dt_, n);
    
    /* fill up g1 and g2 recursively */
    for (int k = 0; k < n; ++k) {
        /* speed increase by reducing array access */
        const double sig2_k = sig2_[k];
        const double b_up_k = b_up_[k];
        const double b_lo_k = b_lo_[k];
        const double cum_drift_k = cum_drift[k];
        const double cum_sig2_k = cum_sig2[k];
        const double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        const double b_up_deriv_k = b_up_deriv_[k] - drift_[k];
        const double b_lo_deriv_k = b_lo_deriv_[k] - drift_[k];
        
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
            const double b_up_k_up_j_diff = b_up_k - b_up_[j] + cum_drift_diff_j;
            const double b_up_k_lo_j_diff = b_up_k - b_lo_[j] + cum_drift_diff_j;
            const double b_lo_k_up_j_diff = b_lo_k - b_up_[j] + cum_drift_diff_j;
            const double b_lo_k_lo_j_diff = b_lo_k - b_lo_[j] + cum_drift_diff_j;
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


DMSample DMGeneralDeriv::rand(rngeng_t& rngeng)
{
    std::normal_distribution<value_t> randn;
    value_t x = drift(0) * dt_ + sqrt_dt_ * sqrt(sig2(0)) * randn(rngeng);
    size_t n = 1;
    int cb = crossed_bounds(x, 1);
    while (!cb) {
        x += drift(n) * dt_ + sqrt_dt_ * sqrt(sig2(n)) * randn(rngeng);
        n += 1;
        cb = crossed_bounds(x, n);
    }
    return DMSample(n * dt_, cb == 1);
}


void DMGeneralLeakDeriv::pdfseq(size_t n, ExtArray& g1, ExtArray& g2)
{
    assert(n > 0);
    
    /* precompute some constants */
    const double sqrt_2_pi = 1 / sqrt(2 * PI);
    const double dt_sqrt_2_pi = dt_ * sqrt_2_pi;
    const double exp_leak = exp(- dt_ * invleak_);
    const double exp2_leak = exp(- 2 * dt_ * invleak_);
    
    /* cumulative mu and sig2, and discount (leak) */
    std::vector<double> cum_drift(n);
    std::vector<double> cum_sig2(n);
    std::vector<double> disc(n);
    double curr_cum_drift = dt_ * drift_[0];
    cum_drift[0] = curr_cum_drift;
    double curr_cum_sig2 = dt_ * sig2_[0];
    cum_sig2[0] = curr_cum_sig2;
    double curr_disc = exp_leak;
    disc[0] = curr_disc;
    for (int j = 1; j < n; ++j) {
        curr_cum_drift = exp_leak * curr_cum_drift + dt_ * drift_[j];
        cum_drift[j] = curr_cum_drift;
        curr_cum_sig2 = exp2_leak * curr_cum_sig2 + dt_ * sig2_[j];
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
        const double sig2_k = sig2_[k];
        const double b_up_k = b_up_[k];
        const double b_lo_k = b_lo_[k];
        const double cum_drift_k = cum_drift[k];
        const double cum_sig2_k = cum_sig2[k];
        const double sqrt_cum_sig2_k = sqrt(cum_sig2_k);
        const double b_up_deriv_k = b_up_deriv_[k] + invleak_ * b_up_k - drift_[k];
        const double b_lo_deriv_k = b_lo_deriv_[k] + invleak_ * b_lo_k - drift_[k];
        
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
            const double b_up_k_up_j_diff = b_up_k - disc_j * b_up_[j] + cum_drift_diff_j;
            const double b_up_k_lo_j_diff = b_up_k - disc_j * b_lo_[j] + cum_drift_diff_j;
            const double b_lo_k_up_j_diff = b_lo_k - disc_j * b_up_[j] + cum_drift_diff_j;
            const double b_lo_k_lo_j_diff = b_lo_k - disc_j * b_lo_[j] + cum_drift_diff_j;
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


DMSample DMGeneralLeakDeriv::rand(rngeng_t& rngeng)
{
    // dx = (- x(t)/tau(t) + mu(t)) dt + sig(t) dW
    std::normal_distribution<value_t> randn;
    value_t x = drift(0) * dt_ + sqrt_dt_ * sqrt(sig2(0)) * randn(rngeng);
    size_t n = 1;
    int cb = crossed_bounds(x, 1);
    while (!cb) {
        x += (drift(n) - invleak_ * x) * dt_ + sqrt_dt_ * sqrt(sig2(n)) * randn(rngeng);
        n += 1;
        cb = crossed_bounds(x, n);
    }
    return DMSample(n * dt_, cb == 1);
}

