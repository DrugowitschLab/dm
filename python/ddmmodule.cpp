/*
 * Python wrapper to ddm_fpt_lib
 *
 * Provides the following functions:
 *
 * ----- (g1, g2) = fpt(mu, bound, dt, tmax[, mnorm])
 *
 * computes the first-passage time distribution of a diffusion model given
 * drift mu and bounds at -bound and bound. dt is the time steps in which this
 * distribution is computed, until time tmax.
 *
 * mu and bound need to be NumPy arrays of type Float. If they are shorter
 * than the returned distribution arrays, then their last element is repeated.
 *
 * dt and tmax are Float-type scalar values.
 *
 * If mnorm (boolean, default false) is given and true, the returned
 * distributions are normalised before being returned.
 *
 *
 * ----- (g1, g2) = fpt_w(a, k, bound, dt, tmax[, mnorm])
 *
 * computes the first-passage time distribution assuming weighted accumulation
 * with weights given by vector a. k is a scalar that determines the
 * proportionality constant.
 *
 * a and bound need to be NumPy arrays of type Float. If they are shorter than
 * the returned distribution arrays, then their last element is repeated.
 *
 * k, dt, and tmax are Float-type scalar values.
 *
 * mnorm has the same effect as for fpt(.).
 *
 * ----- (g1, g2) = fpt_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv, dt, tmax,
 *                           [inv_leak, mnorm])
 *
 * computed the first-passage time distribution for drift vector mu, variance
 * vector sig2, lower and upper bound vectors b_lo and b_up, their time
 * derivatives in vectors b_lo_deriv and b_up_deriv, in steps of dt until time
 * tmax. If given, the inverse integrator time-constant is inv_leak.
 *
 * All vectors are NumPy arrays of type Float, which will be extended, if
 * necessary. All scalars are of type Float.
 *
 * mnorm has the same effect as for fpt(.).
 **/


#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "../ddm_fpt_lib/ddm_fpt_lib.h"

/* evaluates if given array is non-empty 1D NumPy with double elements */
int is1DDoubleArray(PyArrayObject* x)
{
    if (!(PyArray_NDIM(x) == 1 && PyArray_DIM(x, 0) > 0 &&
          PyArray_DESCR(x)->type_num == NPY_DOUBLE)) {
        PyErr_SetString(PyExc_ValueError,
                        "Array(s) must be non-empty 1D array of type Float");
        return 0;
    } else
        return 1;
}


/* method fpt(mu, bound, dt, tmax, ...) */
static PyObject* ddmmod_fpt(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *py_mu, *py_bound;
    double *mu_data, *bound_data;
    PyObject* mnorm_obj = NULL;
    double dt, t_max;
    static char* kwlist[] = {"mu", "bound", "dt", "tmax", "mnorm", NULL};
    int n_max, mnorm_bool;
    npy_intp out_size[1];
    PyObject *py_g1, *py_g2, *py_tuple;

    /* process arguments */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!dd|O", kwlist, 
                                     &PyArray_Type, &py_mu, &PyArray_Type, &py_bound,
                                     &dt, &t_max, &mnorm_obj))
        return NULL;
    if (!(is1DDoubleArray(py_mu) && is1DDoubleArray(py_bound)))
        return NULL;
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt needs to be larger than 0");
        return NULL;
    }
    if (t_max <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "tmax needs to be larger than 0");
        return NULL;
    }
    mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);
    mu_data = (double*) PyArray_DATA(py_mu);
    bound_data = (double*) PyArray_DATA(py_bound);

    /* get output length and reserve space */
    n_max = (int) ceil(t_max / dt);
    out_size[0] = n_max;
    py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }

    /* decide which function to call, based on sizes of inputs */
    if (PyArray_DIM(py_mu, 0) == 1) {
        if (PyArray_DIM(py_bound, 0) == 1) {
            /* size of mu == 1, bound == 1 */
            ddm_fpt_const(mu_data[0], bound_data[0], dt, n_max,
                          (double*) PyArray_DATA((PyArrayObject*) py_g1),
                          (double*) PyArray_DATA((PyArrayObject*) py_g2));
        } else {
            /* size of mu == 1, bound > 1 - replicate last element of bound */
            double* bound_ext;
            const int bound_size = PyArray_DIM(py_bound, 0);
            int err;

            bound_ext = extend_vector(bound_data, bound_size, n_max,
                                      bound_data[bound_size - 1]);
            if (bound_ext == NULL) goto memory_fail;

            err = ddm_fpt_const_mu(mu_data[0], bound_ext, dt, n_max,
                (double*) PyArray_DATA((PyArrayObject*) py_g1),
                (double*) PyArray_DATA((PyArrayObject*) py_g2));
            free(bound_ext);
            if (err == -1) goto memory_fail;
        }
    } else {
        /* size of mu > 1 - replicate last element of both bound and mu */
        double *bound_ext, *mu_ext;
        const int mu_size = PyArray_DIM(py_mu, 0);
        const int bound_size = PyArray_DIM(py_bound, 0);
        int err;

        mu_ext = extend_vector(mu_data, mu_size, n_max, mu_data[mu_size - 1]);
        bound_ext = extend_vector(bound_data, bound_size, n_max, bound_data[bound_size - 1]);
        if (mu_ext == NULL || bound_ext == NULL) {
            free(mu_ext);  /* in case only one of the allocations failed */
            free(bound_ext);
            goto memory_fail;
        }

        err = ddm_fpt(mu_ext, bound_ext, dt, n_max,
                      (double*) PyArray_DATA((PyArrayObject*) py_g1),
                      (double*) PyArray_DATA((PyArrayObject*) py_g2));
        free(mu_ext);
        free(bound_ext);
        if (err == -1) goto memory_fail;
    }

    if (mnorm_bool)
        mnorm((double*) PyArray_DATA((PyArrayObject*) py_g1),
              (double*) PyArray_DATA((PyArrayObject*) py_g2), n_max, dt);

    /* create tuple to return */
    py_tuple = PyTuple_New(2);
    if (py_tuple == NULL) goto memory_fail;
    PyTuple_SET_ITEM(py_tuple, 0, py_g1);
    PyTuple_SET_ITEM(py_tuple, 1, py_g2);
    return py_tuple;

memory_fail:
    Py_DECREF(py_g1);
    Py_DECREF(py_g2);
    PyErr_SetString(PyExc_MemoryError, "out of memory");
    return NULL;
}


/* method fpt_w(a, k, bound, dt, tmax, ...) */
static PyObject* ddmmod_fpt_w(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *py_a, *py_bound;
    double *a_ext, *bound_ext;
    PyObject* mnorm_obj = NULL;
    double k, dt, t_max;
    static char* kwlist[] = {"a", "k", "bound", "dt", "tmax", "mnorm", NULL};
    int n_max, mnorm_bool, err;
    npy_intp out_size[1];
    PyObject *py_g1, *py_g2, *py_tuple;

    /* process arguments */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!dO!dd|O", kwlist, 
                                     &PyArray_Type, &py_a, &k, &PyArray_Type, &py_bound,
                                     &dt, &t_max, &mnorm_obj))
        return NULL;
    if (!(is1DDoubleArray(py_a) && is1DDoubleArray(py_bound)))
        return NULL;
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt needs to be larger than 0");
        return NULL;
    }
    if (t_max <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "tmax needs to be larger than 0");
        return NULL;
    }
    mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);

    /* get output length and reserve space */
    n_max = (int) ceil(t_max / dt);
    out_size[0] = n_max;
    py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }

    /* extend arrays, if too short, and compute fpt */
    a_ext = extend_vector((double*) PyArray_DATA(py_a), PyArray_DIM(py_a, 0), n_max,
                          ((double*) PyArray_DATA(py_a))[PyArray_DIM(py_a, 0) - 1]);
    bound_ext = extend_vector((double*) PyArray_DATA(py_bound), PyArray_DIM(py_bound, 0), n_max,
                              ((double*) PyArray_DATA(py_bound))[PyArray_DIM(py_bound, 0) - 1]);
    if (a_ext == NULL || bound_ext == NULL) {
        free(a_ext);
        free(bound_ext);
        goto memory_fail;
    }

    err = ddm_fpt_w(a_ext, bound_ext, k, dt, n_max, 
                    (double*) PyArray_DATA((PyArrayObject*) py_g1),
                    (double*) PyArray_DATA((PyArrayObject*) py_g2));
    free(a_ext);
    free(bound_ext);
    if (err == -1) goto memory_fail;

    if (mnorm_bool)
        mnorm((double*) PyArray_DATA((PyArrayObject*) py_g1),
              (double*) PyArray_DATA((PyArrayObject*) py_g2), n_max, dt);

    /* create tuple to return */
    py_tuple = PyTuple_New(2);
    if (py_tuple == NULL) goto memory_fail;
    PyTuple_SET_ITEM(py_tuple, 0, py_g1);
    PyTuple_SET_ITEM(py_tuple, 1, py_g2);
    return py_tuple;

memory_fail:
    Py_DECREF(py_g1);
    Py_DECREF(py_g2);
    PyErr_SetString(PyExc_MemoryError, "out of memory");
    return NULL;
}


/* method fpt_full(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv, dt, tmax,
 *                 [inv_leak, mnorm]) */
static PyObject* ddmmod_fpt_full(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyArrayObject *py_mu, *py_sig2, *py_b_lo, *py_b_up, *py_b_lo_deriv, *py_b_up_deriv;
    double *mu_ext, *sig2_ext, *b_lo_ext, *b_up_ext, *b_lo_deriv_ext, *b_up_deriv_ext;
    PyObject* mnorm_obj = NULL;
    double dt, t_max, inv_leak = 0.0;
    static char* kwlist[] = {"mu", "sig2", "b_lo", "b_up", "b_lo_deriv",
                             "b_up_deriv", "dt", "tmax", "inv_leak", "mnorm", NULL};
    int n_max, mnorm_bool, err;
    npy_intp out_size[1];
    PyObject *py_g1, *py_g2, *py_tuple;

    /* process arguments */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!O!O!dd|dO", kwlist, 
                                     &PyArray_Type, &py_mu,
                                     &PyArray_Type, &py_sig2,
                                     &PyArray_Type, &py_b_lo,
                                     &PyArray_Type, &py_b_up,
                                     &PyArray_Type, &py_b_lo_deriv,
                                     &PyArray_Type, &py_b_up_deriv,
                                     &dt, &t_max, &inv_leak, &mnorm_obj))
        return NULL;
    if (!(is1DDoubleArray(py_mu) && is1DDoubleArray(py_sig2) &&
          is1DDoubleArray(py_b_lo) && is1DDoubleArray(py_b_up) &&
          is1DDoubleArray(py_b_lo_deriv) && is1DDoubleArray(py_b_up_deriv)))
        return NULL;
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt needs to be larger than 0");
        return NULL;
    }
    if (t_max <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "tmax needs to be larger than 0");
        return NULL;
    }
    if (inv_leak < 0.0) {
        PyErr_SetString(PyExc_ValueError, "inv_leak needs to be non-negative");
        return NULL;
    }
    mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);

    /* get output length and reserve space */
    n_max = (int) ceil(t_max / dt);
    out_size[0] = n_max;
    py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }

    /* extend arrays, if too short, and compute fpt */
    /* copying all these vectors is not particularly efficient -
       in a future version, it would be better to create an extended
       vector class (in C++) that extends the array on the fly. */
    mu_ext = extend_vector((double*) PyArray_DATA(py_mu), PyArray_DIM(py_mu, 0), n_max,
                          ((double*) PyArray_DATA(py_mu))[PyArray_DIM(py_mu, 0) - 1]);
    sig2_ext = extend_vector((double*) PyArray_DATA(py_sig2), PyArray_DIM(py_sig2, 0), n_max,
                          ((double*) PyArray_DATA(py_sig2))[PyArray_DIM(py_sig2, 0) - 1]);
    b_lo_ext = extend_vector((double*) PyArray_DATA(py_b_lo), PyArray_DIM(py_b_lo, 0), n_max,
                             ((double*) PyArray_DATA(py_b_lo))[PyArray_DIM(py_b_up, 0) - 1]);
    b_up_ext = extend_vector((double*) PyArray_DATA(py_b_up), PyArray_DIM(py_b_up, 0), n_max,
                             ((double*) PyArray_DATA(py_b_up))[PyArray_DIM(py_b_up, 0) - 1]);
    b_lo_deriv_ext = extend_vector((double*) PyArray_DATA(py_b_lo_deriv),
                                   PyArray_DIM(py_b_lo_deriv, 0), n_max, 0.0);
    b_up_deriv_ext = extend_vector((double*) PyArray_DATA(py_b_up_deriv),
                                    PyArray_DIM(py_b_up_deriv, 0), n_max, 0.0);
    if (mu_ext == NULL || sig2_ext == NULL || b_lo_ext == NULL ||
        b_up_ext == NULL || b_lo_deriv_ext == NULL || b_up_deriv_ext == NULL) {
        free(mu_ext);
        free(sig2_ext);
        free(b_lo_ext);
        free(b_up_ext);
        free(b_lo_deriv_ext);
        free(b_up_deriv_ext);
        goto memory_fail;
    }

    if (inv_leak > 0.0)
        err = ddm_fpt_full_leak(mu_ext, sig2_ext, b_lo_ext, b_up_ext,
                                b_lo_deriv_ext, b_up_deriv_ext,
                                inv_leak, dt, n_max,
                                (double*) PyArray_DATA((PyArrayObject*) py_g1),
                                (double*) PyArray_DATA((PyArrayObject*) py_g2));
    else
        err = ddm_fpt_full(mu_ext, sig2_ext, b_lo_ext, b_up_ext,
                           b_lo_deriv_ext, b_up_deriv_ext, dt, n_max,
                           (double*) PyArray_DATA((PyArrayObject*) py_g1),
                           (double*) PyArray_DATA((PyArrayObject*) py_g2));
    free(mu_ext);
    free(sig2_ext);
    free(b_lo_ext);
    free(b_up_ext);
    free(b_lo_deriv_ext);
    free(b_up_deriv_ext);
    if (err == -1) goto memory_fail;

    if (mnorm_bool)
        mnorm((double*) PyArray_DATA((PyArrayObject*) py_g1),
              (double*) PyArray_DATA((PyArrayObject*) py_g2), n_max, dt);

    /* create tuple to return */
    py_tuple = PyTuple_New(2);
    if (py_tuple == NULL) goto memory_fail;
    PyTuple_SET_ITEM(py_tuple, 0, py_g1);
    PyTuple_SET_ITEM(py_tuple, 1, py_g2);
    return py_tuple;

memory_fail:
    Py_DECREF(py_g1);
    Py_DECREF(py_g2);
    PyErr_SetString(PyExc_MemoryError, "out of memory");
    return NULL;
}


/* methods table */
static PyMethodDef DDMMethods[] = {
    {"fpt", (PyCFunction) ddmmod_fpt, METH_VARARGS | METH_KEYWORDS,
     "Computes the first-passage time distributions given mu and bound"},
    {"fpt_w", (PyCFunction) ddmmod_fpt_w, METH_VARARGS | METH_KEYWORDS,
     "Computes the first-passage time distributions with weighted accumulation"},
    {"fpt_full", (PyCFunction) ddmmod_fpt_full, METH_VARARGS | METH_KEYWORDS,
     "Computes the first-passage time distributions given mu, sig2, and bound"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
    /* module initialisation for Python 3 */
    static struct PyModuleDef ddmmodule = {
       PyModuleDef_HEAD_INIT,
       "ddm",   /* name of module */
       "Module to compute first-passage time densities of diffusion models",
       -1,
       DDMMethods
    };

    PyMODINIT_FUNC PyInit_ddm(void)
    {
        PyObject *m = PyModule_Create(&ddmmodule);
        import_array();
        return m;
    }
#else
    /* module initialisation for Python 2 */
    PyMODINIT_FUNC initddm(void)
    {
        Py_InitModule("ddm", DDMMethods);
        import_array();
    }
#endif
