/**
 * Copyright (c) 2013, 2014 Jan Drugowitsch
 * All rights reserved.
 * See the file LICENSE for licensing information.
 *   
 * ddm_ftp_lib.h
 **/

#ifndef _DDM_FPT_LIB_H_
#define _DDM_FPT_LIB_H_

#include<cstring>
#include<cmath>
#include<algorithm>
#include<memory>
#include<limits>
#include<stdexcept>

/**
 * An array of doubles that returns values beyond its size.
 *
 * For example, if array x has n elements, x[0] ... x[n-1], x[i] for any
 * i >= n will return the same element.
 *
 * The array data is managed by a std::shared_ptr<double>, such that ownership
 * is shared between all shared_ptr-holders. Such ownership requires a custom
 * deleter to call delete[] rather than delete, which the convenience method
 * shared_owner() provides. Another method shared_noowner() turns a value_t
 * array into a shared_ptr without claiming ownership.
 **/
class ExtArray {
public:
    typedef double value_t;
    typedef std::shared_ptr<value_t> data_t;
    typedef int size_t;

    // turns value_t array into shared_prt<value_t> with ownership
    static data_t shared_owner(value_t* x)
    { return data_t(x, [](value_t* p) { delete[] p; }); }

    // turns value_t array into shared_ptr<value_t> without ownership
    static data_t shared_noowner(value_t* x)
    { return data_t(x, [](value_t* p) { }); }

    // returns constant array at x
    static ExtArray const_array(value_t x)
    { return ExtArray(shared_noowner(nullptr), x, 0); }

    ExtArray(data_t d, size_t data_size)
    : data(d), last(data_size > 0 ? d.get()[data_size-1] : 0),
      data_size(data_size) {}

    ExtArray(data_t d, value_t last, size_t data_size)
    : data(d), last(last), data_size(data_size) {}

    size_t size() const 
    { return data_size; }

    bool isconst() const
    { return data_size == 0 || (data_size == 1 && data.get()[0] == last); }

    // [idx] for idx >= size() assigns last element
    value_t& operator[](size_t idx)
    { return idx >= data_size ? last : data.get()[idx]; }
    value_t operator[](size_t idx) const
    { return idx >= data_size ? last : data.get()[idx]; }

    // returns cumulative sum of f * (x[0], ..., x[size()-1])
    ExtArray cumsum(value_t f) const
    {
        value_t* x = new value_t[data_size];
        if (data_size > 0) {
            value_t cursum = f * data.get()[0];
            x[0] = cursum;
            for (size_t i = 1; i < data_size; ++i) {
                cursum += f * data.get()[i];
                x[i] = cursum;
            }
            return ExtArray(shared_owner(x), cursum, data_size);
        } else
            return ExtArray(shared_owner(x), 0, 0);
    }

private:
    data_t data;
    value_t last;
    size_t data_size;
};


// diffusion model base class
class DMBase {
public:
    typedef double value_t;
    typedef int size_t;

    virtual ~DMBase() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2) = 0;
    virtual value_t pdfu(value_t t) = 0;
    virtual value_t pdfl(value_t t) = 0;

    // factory functions
    static DMBase* create(const ExtArray& drift, const ExtArray& bound,
                          value_t dt);
    static DMBase* createw(const ExtArray& drift, const ExtArray& bound,
                           value_t k, value_t dt);
    static DMBase* create(const ExtArray& drift, const ExtArray& sig2,
                          const ExtArray& b_lo, const ExtArray& b_up,
                          const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                          value_t dt,
                          value_t invleak = std::numeric_limits<value_t>::infinity());

protected:
    static constexpr double PI = 3.14159265358979323846;
    static constexpr double TWOPI = 2 * PI;
    static constexpr double PISQR = PI * PI;
};


// TODO: refactor DMConstDriftConstBound and DMConstDriftConstABound to have
// a common parent class that handles some of the series expansion code.
// TODO: for both classes, introduce caching of c1,c2,.. constants.

// diffusion model with constant drift and constant symmetric bounds
class DMConstDriftConstBound : public DMBase {
public:
    DMConstDriftConstBound(value_t drift, value_t bound, value_t dt)
    : drift(drift), bound(bound), dt(dt) {}

    virtual ~DMConstDriftConstBound() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { return fpt_symup(t, 4 * bound * bound, drift * drift / 2, drift * bound); }
    virtual value_t pdfl(value_t t)
    { return exp(- 2 * drift * bound) * pdfu(t); }

private:
    value_t drift;
    value_t bound;
    value_t dt;

    static constexpr double SERIES_ACC = 1e-29;

    // choose between two series expansions, from Navarro & Fuss (2009), Eq. (13)
    static bool useshorttseries(value_t t, value_t tol)
    { return (2.0 + sqrt(-2 * t * log(2 * tol * sqrt(TWOPI * t))) < 
             sqrt(- 2 * log(PI * t * tol) / (t * PISQR))) ? 1 : 0; }
    /** fpt_symup - fpt density at upper boundary, symmetric bounds
     * The required arguments are
     * c1 = 4 * bound^2
     * c2 = mu^2 / 2
     * c3 = mu * bound
     * The density at the lower bound is exp(-2 mu bound) times the upper density
     **/
    static value_t fpt_symup(value_t t, value_t c1, value_t c2, value_t c3)
    { return exp(c3 - c2 * t) / c1 * fpt_symfastseries(t / c1, SERIES_ACC); }
    /** fpt_symfastseries - fpt lower density, mu=0, bounds {0,1}, starting at 0.5
     * The function chooses between the faster of two series expansions, depending
     * on the given t. It returns the lower bound density at t.
     **/
    static value_t fpt_symfastseries(value_t t, value_t tol)
    { if (t == 0.0) return 0.0;
      return useshorttseries(t, tol) ?
             fpt_symseries(t, 1 / (8 * t), 1 / sqrt(8 * PI * pow(t, 3)), tol) :
             fpt_symseries(t, t * PISQR / 2, PI, tol); }

    static value_t fpt_symseries(value_t t, value_t a, value_t b, value_t tol);
};


// diffusion model with constant drift and constant asymmetric bounds
class DMConstDriftConstABound : public DMBase {
public:
    DMConstDriftConstABound(value_t drift, value_t b_lo, value_t b_up, value_t dt)
    : drift(drift), b_up(b_up), b_lo(b_lo), dt(dt) {}

    virtual ~DMConstDriftConstABound() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { return fpt_asymup(t, pow(b_up - b_lo, 2), drift * drift / 2,
                        drift * b_up, -b_lo / (b_up - b_lo));  }
    virtual value_t pdfl(value_t t)
    { return fpt_asymlo(t, pow(b_up - b_lo, 2), drift * drift / 2,
                        drift * b_lo, -b_lo / (b_up - b_lo)); }

private:
    value_t drift;
    value_t b_up, b_lo;
    value_t dt;

    static constexpr double SERIES_ACC = 1e-29;

    // choose between two series expansions, from Navarro & Fuss (2009), Eq. (13)
    static bool useshorttseries(value_t t, value_t tol)
    { return (2.0 + sqrt(-2 * t * log(2 * tol * sqrt(TWOPI * t))) < 
             sqrt(- 2 * log(PI * t * tol) / (t * PISQR))) ? 1 : 0; }
    /** fpt_asymlo - fpt for upper bound, for const drift/bounds
     * The required arguments are
     * c1 = (bu - bl)^2
     * c2 = mu^2 / 2
     * c3 = mu * bu
     * w = -bl / (bu - bl)
     * where mu = drift, bu and bl are upper and lower bounds.
     **/
    static value_t fpt_asymup(value_t t, value_t c1, value_t c2, value_t c3, value_t w)
    { return exp(c3 - c2 * t) / c1 * fpt_asymfastseries(t / c1, 1 - w, SERIES_ACC); }

    /** fpt_asymlo - fpt for lower bound, for const drift/bounds
     * The required arguments are as for fpt_asymup, except c4, which is
     * c4 = mu * bl
     **/
    static value_t fpt_asymlo(value_t t, value_t c1, value_t c2, value_t c4, value_t w)
    { return exp(c4 - c2 * t) / c1 * fpt_asymfastseries(t / c1, w, SERIES_ACC); }
    /** fpt_asymfastseries - fpt lower density, mu=0, bounds {0,1}, starting at w
     * The function chooses between the faster of two series expansions, depending
     * on the given t. It returns the lower bound density at t.
     **/
    static value_t fpt_asymfastseries(value_t t, value_t w, value_t tol)
    { if (t == 0.0) return 0.0;
      return useshorttseries(t, tol) ? fpt_asymshortt(t, w, tol) :
                                       fpt_asymlongt(t, w, tol); }

    static value_t fpt_asymshortt(value_t t, value_t w, value_t tol);
    static value_t fpt_asymlongt(value_t t, value_t w, value_t tol);
};


// diffusion model with constant drift and varying symmetric bounds
class DMConstDriftVarBound : public DMBase {
public:
    DMConstDriftVarBound(value_t drift, const ExtArray& bound, value_t dt)
    : drift(drift), bound(bound), dt(dt) {}

    virtual ~DMConstDriftVarBound() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { throw std::runtime_error("pdfu() not available for DMConstDriftVarBound"); }
    virtual value_t pdfl(value_t t)
    { throw std::runtime_error("pdfl() not available for DMConstDriftVarBound"); }

private:
    value_t drift;
    ExtArray bound;
    value_t dt;
};


// diffusion model with varying drift and varying symmetric bounds
class DMVarDriftVarBound : public DMBase {
public:
    DMVarDriftVarBound(const ExtArray& drift, const ExtArray& bound, value_t dt)
    : drift(drift), bound(bound), dt(dt) {}

    virtual ~DMVarDriftVarBound() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { throw std::runtime_error("pdfu() not available for DMVarDriftVarBound"); }
    virtual value_t pdfl(value_t t)
    { throw std::runtime_error("pdfl() not available for DMVarDriftVarBound"); }

private:
    ExtArray drift, bound;
    value_t dt;
};


// diffusion model with varying drift, varying symmetric bound, and weighted inp.
class DMWVarDriftVarBound : public DMBase {
public:
    DMWVarDriftVarBound(const ExtArray& drift, const ExtArray& bound, 
                        value_t k, value_t dt)
    : drift(drift), bound(bound), k(k), dt(dt) {}

    virtual ~DMWVarDriftVarBound() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { throw std::runtime_error("pdfu() not available for DMWVarDriftVarBound"); }
    virtual value_t pdfl(value_t t)
    { throw std::runtime_error("pdfl() not available for DMWVarDriftVarBound"); }

private:
    ExtArray drift, bound;
    value_t k, dt;
};


// diffusion model with varying drift / variance / bounds, and known
// bound derivatives
class DMGeneralDeriv : public DMBase {
public:
    DMGeneralDeriv(const ExtArray& drift, const ExtArray& sig2,
                   const ExtArray& b_lo, const ExtArray& b_up,
                   const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                   value_t dt)
    : drift(drift), sig2(sig2), b_lo(b_lo), b_up(b_up),
      b_lo_deriv(b_lo_deriv), b_up_deriv(b_up_deriv), dt(dt) {}

    virtual ~DMGeneralDeriv() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { throw std::runtime_error("pdfu() not available for DMGeneralDeriv"); }
    virtual value_t pdfl(value_t t)
    { throw std::runtime_error("pdfl() not available for DMGeneralDeriv"); }

private:
    ExtArray drift, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv;
    value_t dt;
};


// diffusion model with time-varying drift / variance / bounds / leak / and known
// bound derivatives
class DMGeneralLeakDeriv : public DMBase {
public:
    DMGeneralLeakDeriv(const ExtArray& drift, const ExtArray& sig2,
                       const ExtArray& b_lo, const ExtArray& b_up,
                       const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                       value_t invleak, value_t dt)
    : drift(drift), sig2(sig2), b_lo(b_lo), b_up(b_up),
      b_lo_deriv(b_lo_deriv), b_up_deriv(b_up_deriv), invleak(invleak), dt(dt) {}

    virtual ~DMGeneralLeakDeriv() {}

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { throw std::runtime_error("pdfu() not available for DMGeneralLeakDeriv"); }
    virtual value_t pdfl(value_t t)
    { throw std::runtime_error("pdfl() not available for DMGeneralLeakDeriv"); }

private:
    ExtArray drift, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv;
    value_t invleak, dt;
};


/** normalising the mass, such that (sum(g1) + sum(g2) * delta_t = 1 
 *
 * Function makes sure that g1(t) >= 0, g2(t) >= 0, for all t, and that
 * (sum(g1) + sum(g2) * delta_t) = 1. It does so by eventually adding mass to
 * the last elements of g1 / g2, such that the ratio
 * sum(g1) / (sum(g1) + sum(g2)) (after removing negative values) remains
 * unchanged.
 */
void mnorm(double g1[], double g2[], int n, double delta_t);

#endif
