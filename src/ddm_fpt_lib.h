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
#include<random>

typedef std::mt19937 dm_rngeng_t;
typedef double dm_value_t;

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
    typedef dm_value_t value_t;
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

    ExtArray(size_t data_size)
    : data_(shared_owner(new value_t[data_size])), last_(0.0),
      data_size_(data_size) {} 

    ExtArray(data_t d, size_t data_size)
    : data_(d), last_(data_size > 0 ? d.get()[data_size-1] : 0),
      data_size_(data_size) {}

    ExtArray(data_t d, value_t last, size_t data_size)
    : data_(d), last_(last), data_size_(data_size) {}

    size_t size() const 
    { return data_size_; }

    bool isconst() const
    { return data_size_ == 0 || (data_size_ == 1 && data_.get()[0] == last_); }

    // [idx] for idx >= size() assigns last element
    value_t& operator[](size_t idx)
    { return idx >= data_size_ ? last_ : data_.get()[idx]; }
    value_t operator[](size_t idx) const
    { return idx >= data_size_ ? last_ : data_.get()[idx]; }

    // returns cumulative sum of f * (x[0], ..., x[data_size_out-1])
    ExtArray cumsum(value_t f, size_t data_size_out) const;

    // returns the finite difference derivative, assuming steps of dt
    ExtArray deriv(value_t dt) const;

private:
    data_t data_;
    value_t last_;
    size_t data_size_;
};


// base class for fast diffusion model sampling
class FastDMSamplingBase {
public:
    typedef dm_value_t value_t;
    typedef dm_rngeng_t rngeng_t;

    virtual ~FastDMSamplingBase() { }

    virtual value_t rand(rngeng_t& rngeng) = 0;

    static FastDMSamplingBase* create(value_t drift);

protected:
    static constexpr double PI = 3.14159265358979323846;

    static bool acceptt(value_t t, value_t zf, value_t C2);
};


// fast diffusion model fpt sampling with normal/exponential proposal density
class FastDMSamplingNormExp : public FastDMSamplingBase {
public:
    FastDMSamplingNormExp(value_t drift)
    : mu2_(drift * drift)
    {  const value_t t = 0.12 + exp(- fabs(drift) / 3) / 2;
       a_ = (3 + sqrt(9 + 4 * mu2_)) / 6;
       sqrtamu_ = sqrt((a_ - 1) * mu2_ / a_);
       fourmu2pi_ = (4 * mu2_ + PI * PI) / 8;
       Cf1s_ = sqrt(a_) * exp(-sqrtamu_);
       Cf1l_ = PI / (4 * fourmu2pi_);
       CF1st_ = Cf1s_ * erfc(1 / sqrt(2 * a_ * t));
       F1lt_ = - expm1(-t * fourmu2pi_);
       F1inf_ = CF1st_ + Cf1l_ * (1 - F1lt_);
    }

    virtual value_t rand(rngeng_t& rngeng);

private:
    value_t mu2_, a_, sqrtamu_, fourmu2pi_;
    value_t Cf1s_, CF1st_, Cf1l_, F1lt_, F1inf_;

    // returns the inverse of the complementary error function
    static value_t erfcinv(value_t P);
    static value_t rational_invccdf_approx(value_t t);
};


// fast diffusion model fpt sampling with inverse-normal proposal density
class FastDMSamplingInvNorm : public FastDMSamplingBase {
public:
    FastDMSamplingInvNorm(value_t drift)
    : invabsmu_(1 / fabs(drift)), invmu2_(1 / (drift * drift)) {}

    virtual value_t rand(rngeng_t& rngeng);
private:
    value_t invabsmu_, invmu2_;

    // samples inverse-normal with lambda = 1, mean = mu, mu > 0
    static value_t randin(rngeng_t& rngeng, value_t mu, value_t mu2);
};


// single sample from diffusion model
class DMSample {
public:
    typedef dm_value_t value_t;

    DMSample(value_t t, bool upper_bound)
    : t_(t), upper_bound_(upper_bound) {}

    value_t t() const           { return t_; }
    bool upper_bound() const    { return upper_bound_; }
private:
    value_t t_;
    bool upper_bound_;
};


// diffusion model base class
class DMBase {
public:
    typedef dm_value_t value_t;
    typedef int size_t;
    typedef dm_rngeng_t rngeng_t;

    DMBase(value_t dt)
    : dt_(dt), sqrt_dt_(sqrt(dt)) { }

    virtual ~DMBase() {}

    // getters
    value_t dt() const                         { return dt_; }
    virtual value_t sig2(size_t n) const       { return 1.0; }
    virtual value_t drift(size_t n) const = 0;
    virtual value_t b_lo(size_t n) const = 0;
    virtual value_t b_up(size_t n) const = 0;

    // functions to compute first-passage time density
    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2) = 0;
    virtual value_t pdfu(value_t t);
    virtual value_t pdfl(value_t t);

    // functions to sample first-passage times and choices
    virtual DMSample rand(rngeng_t& rngeng);

    /** normalising the mass, such that (sum(g1) + sum(g2) * delta_t = 1 
     *
     * Function makes sure that g1(t) >= 0, g2(t) >= 0, for all t, and that
     * (sum(g1) + sum(g2) * delta_t) = 1. It does so by eventually adding mass to
     * the last elements of g1 / g2, such that the ratio
     * sum(g1) / (sum(g1) + sum(g2)) (after removing negative values) remains
     * unchanged.
     */
    void mnorm(ExtArray& g1, ExtArray& g2) const;

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
    value_t dt_;
    value_t sqrt_dt_;

    // 0 - not crossed, 1 - upper bound, -1 - lower bound
    int crossed_bounds(value_t x, size_t n) const
    { return x >= b_up(n) ? 1 : (x <= b_lo(n) ? -1 : 0); }

private:
    static value_t lininterp(value_t x1, value_t x2, value_t w)
    { return x1 + w * (x2  - x1); }
};


// base class for diffusion models with constant drift/bound
class DMConstBase : public DMBase {
public:
    DMConstBase(value_t dt)
    : DMBase(dt) {}
protected:
    static constexpr double SERIES_ACC = 1e-29;

    // choose between two series expansions, from Navarro & Fuss (2009), Eq. (13)
    static bool useshorttseries(value_t t, value_t tol)
    { return (2.0 + sqrt(-2 * t * log(2 * tol * sqrt(TWOPI * t))) < 
             sqrt(- 2 * log(PI * t * tol) / (t * PISQR))) ? 1 : 0; }
};


// diffusion model with constant drift and constant symmetric bounds
class DMConstDriftConstBound : public DMConstBase {
public:
    DMConstDriftConstBound(value_t drift, value_t bound, value_t dt)
    : DMConstBase(dt), drift_(drift), bound_(bound), fpt_sampler_(), dm_consts_() {}

    virtual ~DMConstDriftConstBound() {}

    virtual value_t drift(size_t n) const  { return drift_; }
    virtual value_t b_lo(size_t n) const   { return -bound_; }
    virtual value_t b_up(size_t n) const   { return bound_; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { compute_dm_consts(); 
      return fpt_symup(t, dm_consts_->c1(), dm_consts_->c2(), dm_consts_->c3()); }
    virtual value_t pdfl(value_t t)
    { compute_dm_consts(); 
      return dm_consts_->c4() * 
             fpt_symup(t, dm_consts_->c1(), dm_consts_->c2(), dm_consts_->c3()); }

    virtual DMSample rand(rngeng_t& rngeng);

private:
    value_t drift_;
    value_t bound_;
    std::shared_ptr<FastDMSamplingBase> fpt_sampler_;

    class DMConsts {
    public:
        DMConsts(value_t drift, value_t bound)
        : c1_(4 * bound * bound), c2_(drift * drift / 2),
          c3_(drift * bound)
        { c4_ = exp(-2 * c3_);  c5_ = 1 / (1 + exp(-2 * c3_)); }
        value_t c1() const { return c1_; }
        value_t c2() const { return c2_; }
        value_t c3() const { return c3_; }
        value_t c4() const { return c4_; }
        value_t c5() const { return c5_; }
    private:
        value_t c1_, c2_, c3_, c4_, c5_;
    };

    std::shared_ptr<DMConsts> dm_consts_;

    void compute_dm_consts()
    { if (!dm_consts_) dm_consts_.reset(new DMConsts(drift_, bound_)); }

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
class DMConstDriftConstABound : public DMConstBase {
public:
    DMConstDriftConstABound(value_t drift, value_t b_lo, value_t b_up, value_t dt)
    : DMConstBase(dt), drift_(drift), b_up_(b_up), b_lo_(b_lo), pdf_consts_() {}

    virtual ~DMConstDriftConstABound() {}

    virtual value_t drift(size_t n) const  { return drift_; }
    virtual value_t b_lo(size_t n) const   { return b_lo_; }
    virtual value_t b_up(size_t n) const   { return b_up_; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);
    virtual value_t pdfu(value_t t)
    { compute_pdf_consts();
      return fpt_asymup(t, pdf_consts_->c1(), pdf_consts_->c2(), 
                        pdf_consts_->c3(), pdf_consts_->w()); }
    virtual value_t pdfl(value_t t)
    { compute_pdf_consts();
      return fpt_asymlo(t, pdf_consts_->c1(), pdf_consts_->c2(), 
                        pdf_consts_->c4(), pdf_consts_->w()); }

    virtual DMSample rand(rngeng_t& rngeng);

private:
    value_t drift_;
    value_t b_up_, b_lo_;

    class PDFConsts {
    public:
        PDFConsts(value_t drift, value_t b_lo, value_t b_up)
        : c2_(drift * drift / 2), c3_(drift * b_up), c4_(drift * b_lo)
        { const value_t bdiff = b_up - b_lo; 
          c1_ = bdiff * bdiff;
          w_ = -b_lo / bdiff; }
        value_t c1() const { return c1_; }
        value_t c2() const { return c2_; }
        value_t c3() const { return c3_; }
        value_t c4() const { return c4_; }
        value_t w()  const { return w_; }
    private:
        value_t c1_, c2_, c3_, c4_, w_;
    };

    std::shared_ptr<PDFConsts> pdf_consts_;

    void compute_pdf_consts()
    { if (!pdf_consts_) pdf_consts_.reset(new PDFConsts(drift_, b_lo_, b_up_)); }

    /** fpt_asymup - fpt for upper bound, for const drift/bounds
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
    : DMBase(dt), drift_(drift), bound_(bound) {}

    virtual ~DMConstDriftVarBound() {}

    virtual value_t drift(size_t n) const  { return drift_; }
    virtual value_t b_lo(size_t n) const   { return -bound_[n]; }
    virtual value_t b_up(size_t n) const   { return bound_[n]; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);

private:
    value_t drift_;
    ExtArray bound_;
};


// diffusion model with varying drift and varying symmetric bounds
class DMVarDriftVarBound : public DMBase {
public:
    DMVarDriftVarBound(const ExtArray& drift, const ExtArray& bound, value_t dt)
    : DMBase(dt), drift_(drift), bound_(bound) {}

    virtual ~DMVarDriftVarBound() {}

    virtual value_t drift(size_t n) const  { return drift_[n]; }
    virtual value_t b_lo(size_t n) const   { return -bound_[n]; }
    virtual value_t b_up(size_t n) const   { return bound_[n]; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);

private:
    ExtArray drift_, bound_;
};


// diffusion model with varying drift, varying symmetric bound, and weighted inp.
class DMWVarDriftVarBound : public DMBase {
public:
    DMWVarDriftVarBound(const ExtArray& drift, const ExtArray& bound, 
                        value_t k, value_t dt)
    : DMBase(dt), drift_(drift), bound_(bound), k_(k) {}

    virtual ~DMWVarDriftVarBound() {}

    virtual value_t drift(size_t n) const  { return drift_[n]; }
    virtual value_t b_lo(size_t n) const   { return -bound_[n]; }
    virtual value_t b_up(size_t n) const   { return bound_[n]; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);

    virtual DMSample rand(rngeng_t& rngeng);

private:
    ExtArray drift_, bound_;
    value_t k_;
};


// diffusion model with varying drift / variance / bounds, and known
// bound derivatives
class DMGeneralDeriv : public DMBase {
public:
    DMGeneralDeriv(const ExtArray& drift, const ExtArray& sig2,
                   const ExtArray& b_lo, const ExtArray& b_up,
                   const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                   value_t dt)
    : DMBase(dt), drift_(drift), sig2_(sig2), b_lo_(b_lo), b_up_(b_up),
      b_lo_deriv_(b_lo_deriv), b_up_deriv_(b_up_deriv) {}

    virtual ~DMGeneralDeriv() {}

    virtual value_t sig2(size_t n) const   { return sig2_[n]; }
    virtual value_t drift(size_t n) const  { return drift_[n]; }
    virtual value_t b_lo(size_t n) const   { return b_lo_[n]; }
    virtual value_t b_up(size_t n) const   { return b_up_[n]; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);

    virtual DMSample rand(rngeng_t& rngeng);

private:
    ExtArray drift_, sig2_, b_lo_, b_up_, b_lo_deriv_, b_up_deriv_;
};


// diffusion model with time-varying drift / variance / bounds / leak / and known
// bound derivatives
class DMGeneralLeakDeriv : public DMBase {
public:
    DMGeneralLeakDeriv(const ExtArray& drift, const ExtArray& sig2,
                       const ExtArray& b_lo, const ExtArray& b_up,
                       const ExtArray& b_lo_deriv, const ExtArray& b_up_deriv,
                       value_t invleak, value_t dt)
    : DMBase(dt), drift_(drift), sig2_(sig2), b_lo_(b_lo), b_up_(b_up),
      b_lo_deriv_(b_lo_deriv), b_up_deriv_(b_up_deriv), invleak_(invleak) {}

    virtual ~DMGeneralLeakDeriv() {}

    virtual value_t sig2(size_t n) const   { return sig2_[n]; }
    virtual value_t drift(size_t n) const  { return drift_[n]; }
    virtual value_t b_lo(size_t n) const   { return b_lo_[n]; }
    virtual value_t b_up(size_t n) const   { return b_up_[n]; }

    virtual void pdfseq(size_t n, ExtArray& g1, ExtArray& g2);

    virtual DMSample rand(rngeng_t& rngeng);

private:
    ExtArray drift_, sig2_, b_lo_, b_up_, b_lo_deriv_, b_up_deriv_;
    value_t invleak_;
};


#endif
