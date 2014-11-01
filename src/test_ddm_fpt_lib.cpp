#include<iostream>
#include<memory>
#include<thread>
#include<cmath>
#include<algorithm>

#include "ddm_fpt_lib.h"

double max_pdfseq_diff(DMBase& dm1, DMBase& dm2, int n)
{
    ExtArray g11(n);
    ExtArray g12(n);
    ExtArray g21(n);
    ExtArray g22(n);
    dm1.pdfseq(n, g11, g12);
    dm2.pdfseq(n, g21, g22);
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, fabs(g11[i] - g21[i]));
        max_diff = std::max(max_diff, fabs(g12[i] - g22[i]));
    }
    return max_diff;
}

void test_pdfseq_diff()
{
    const double drift = 0.9;
    const double bound = 1.1;
    const double dt = 0.005;
    const int n = static_cast<int>(ceil(2.0 / dt));

    DMConstDriftConstBound dmconstsym(drift, bound, dt);
    DMConstDriftConstABound dmconstasym(drift, -bound, bound, dt);
    DMConstDriftVarBound dmvarbound(drift, ExtArray::const_array(bound), dt);
    DMVarDriftVarBound dmvar(ExtArray::const_array(drift), ExtArray::const_array(bound), dt);
    DMGeneralDeriv dmgeneral(ExtArray::const_array(drift), ExtArray::const_array(1.0), 
                             ExtArray::const_array(-bound), ExtArray::const_array(bound),
                             ExtArray::const_array(0.0), ExtArray::const_array(0.0), dt);

    std::cout << "-- Testing consistency of pdf across classes:" << std::endl <<
        "Reported value is maximum absolute difference between pdf's" << std::endl;
    std::cout << "DMConstDriftConstBound vs. DMConstDriftConstABound: " << 
        max_pdfseq_diff(dmconstsym, dmconstasym, n) << std::endl; 
    std::cout << "DMConstDriftConstBound vs. DMConstDriftVarBound:    " << 
        max_pdfseq_diff(dmconstsym, dmvarbound, n) << std::endl; 
    std::cout << "DMConstDriftConstBound vs. DMVarDriftVarBound:      " << 
        max_pdfseq_diff(dmconstsym, dmvar, n) << std::endl;
    std::cout << "DMConstDriftConstBound vs. DMGeneralDeriv:          " << 
        max_pdfseq_diff(dmconstsym, dmgeneral, n) << std::endl;
}


void test_mnorm()
{
    const double drift = 0.9;
    const double bound = 1.1;
    const double dt = 0.005;
    const int n = static_cast<int>(ceil(2.0 / dt));
    DMConstDriftConstBound dm(drift, bound, dt);
    ExtArray g1(n), g2(n);
    dm.pdfseq(n, g1, g2);
    dm.mnorm(g1, g2);
    double gsum = 0.0;
    for (int i = 0; i < n; ++i)
        gsum += g1[i] + g2[i];
    std::cout << "-- Testing probability density normalisation" << std::endl;
    std::cout << "Total probability mass after normalisation: " << gsum * dt << std::endl;
}


void avg_t_bound(DMBase& dm, int n, double& et, double& eb)
{
    std::mt19937 rngeng(1234);
    et = 0.0;  eb = 0.0;
    for (int i = 0; i < n; ++i) {
        DMSample s = dm.rand(rngeng);
        et += s.t();
        if (s.upper_bound()) eb += 1.0;
    }
    et /= n;
    eb /= n;
}


void test_rand()
{
    const double drift = -0.9;
    const double bound = 1.1;
    const double dt = 0.001;
    const int n = 10000;

    DMConstDriftConstBound dmconstsym(drift, bound, dt);
    DMConstDriftConstABound dmconstasym(drift, -bound, bound, dt);
    DMConstDriftVarBound dmvarbound(drift, ExtArray::const_array(bound), dt);
    DMVarDriftVarBound dmvar(ExtArray::const_array(drift), ExtArray::const_array(bound), dt);
    DMGeneralDeriv dmgeneral(ExtArray::const_array(drift), ExtArray::const_array(1.0), 
                             ExtArray::const_array(-bound), ExtArray::const_array(bound),
                             ExtArray::const_array(0.0), ExtArray::const_array(0.0), dt);

    double et, eb;
    std::cout << "-- Testing sampling from diffusion models" << std::endl;
    std::cout << "Expected                <T> = " << bound / drift * tanh(bound * drift) 
        << ",  <B> = " << 1 / (1 + exp(-2 * drift * bound)) << std::endl;
    avg_t_bound(dmconstsym, n, et, eb);
    std::cout << "DMConstDriftConstBound  <T> = " << et << ",  <B> = " << eb << std::endl;
    avg_t_bound(dmconstasym, n, et, eb);
    std::cout << "DMConstDriftConstABound <T> = " << et << ",  <B> = " << eb << std::endl;
    avg_t_bound(dmvarbound, n, et, eb);
    std::cout << "DMConstDriftVarBound    <T> = " << et << ",  <B> = " << eb << std::endl;
    avg_t_bound(dmvar, n, et, eb);
    std::cout << "DMVarDriftVarBound      <T> = " << et << ",  <B> = " << eb << std::endl;
    avg_t_bound(dmgeneral, n, et, eb);
    std::cout << "DMGeneralDeriv          <T> = " << et << ",  <B> = " << eb << std::endl;

}




int main()
{
    test_pdfseq_diff();
    std::cout << std::endl;
    test_mnorm();
    std::cout << std::endl;
    test_rand();

    //std::cout << measure<>::execution(test_fun);
}