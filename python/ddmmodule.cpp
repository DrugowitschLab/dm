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

#include "../src/ddm_fpt_lib.h"

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
    /* process arguments */
    PyArrayObject *py_mu, *py_bound;
    double dt, t_max;
    PyObject* mnorm_obj = NULL;
    static char* kwlist[] = {"mu", "bound", "dt", "tmax", "mnorm", NULL};
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
    bool mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);
    ExtArray mu(ExtArray::shared_noowner((double*) PyArray_DATA(py_mu)),
                PyArray_DIM(py_mu, 0));
    ExtArray bound(ExtArray::shared_noowner((double*) PyArray_DATA(py_bound)),
                   PyArray_DIM(py_bound, 0));

    /* get output length and reserve space */
    int n_max = (int) ceil(t_max / dt);
    npy_intp out_size[1] = { n_max };
    //out_size[0] = n_max;
    PyObject* py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    PyObject* py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }
    ExtArray g1(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g1)), n_max);
    ExtArray g2(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g2)), n_max);

    /* compute pdf */
    DMBase* dm = DMBase::create(mu, bound, dt);
    dm->pdfseq(n_max, g1, g2);
    if (mnorm_bool) dm->mnorm(g1, g2);
    delete dm;

    /* create tuple to return */
    PyObject* py_tuple = PyTuple_New(2);
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
    /* process arguments */
    static char* kwlist[] = {"a", "k", "bound", "dt", "tmax", "mnorm", NULL};
    PyArrayObject *py_a, *py_bound;
    double k, dt, t_max;
    PyObject* mnorm_obj = NULL;
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
    bool mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);
    ExtArray a(ExtArray::shared_noowner((double*) PyArray_DATA(py_a)),
                PyArray_DIM(py_a, 0));
    ExtArray bound(ExtArray::shared_noowner((double*) PyArray_DATA(py_bound)),
                   PyArray_DIM(py_bound, 0));

    /* get output length and reserve space */
    int n_max = (int) ceil(t_max / dt);
    npy_intp out_size[1] = { n_max };
    PyObject* py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    PyObject* py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }
    ExtArray g1(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g1)), n_max);
    ExtArray g2(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g2)), n_max);

    /* compute fpt */
    DMBase* dm = DMBase::createw(a, bound, k, dt);
    dm->pdfseq(n_max, g1, g2);
    if (mnorm_bool) dm->mnorm(g1, g2);
    delete dm;

    /* create tuple to return */
    PyObject* py_tuple = PyTuple_New(2);
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
    /* process arguments */
    static char* kwlist[] = {"mu", "sig2", "b_lo", "b_up", "b_lo_deriv",
                             "b_up_deriv", "dt", "tmax", "inv_leak", "mnorm", NULL};
    PyArrayObject *py_mu, *py_sig2, *py_b_lo, *py_b_up, *py_b_lo_deriv, *py_b_up_deriv;
    double dt, t_max, inv_leak = 0.0;
    PyObject* mnorm_obj = NULL;
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
    bool mnorm_bool = (mnorm_obj != NULL && PyObject_IsTrue(mnorm_obj) == 1);
    ExtArray mu(ExtArray::shared_noowner((double*) PyArray_DATA(py_mu)),
                PyArray_DIM(py_mu, 0));
    ExtArray sig2(ExtArray::shared_noowner((double*) PyArray_DATA(py_sig2)),
                  PyArray_DIM(py_sig2, 0));
    ExtArray b_lo(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_lo)),
                  PyArray_DIM(py_b_lo, 0));
    ExtArray b_up(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_up)),
                  PyArray_DIM(py_b_up, 0));
    ExtArray b_lo_deriv(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_lo_deriv)),
                        PyArray_DIM(py_b_lo_deriv, 0));
    ExtArray b_up_deriv(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_up_deriv)),
                        PyArray_DIM(py_b_up_deriv, 0));

    /* get output length and reserve space */
    int n_max = (int) ceil(t_max / dt);
    npy_intp out_size[1] = { n_max };
    PyObject* py_g1 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g1 == NULL) return NULL;
    PyObject* py_g2 = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_g2 == NULL) {
        Py_DECREF(py_g1);
        return NULL;
    }
    ExtArray g1(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g1)), n_max);
    ExtArray g2(ExtArray::shared_noowner(
        (double*) PyArray_DATA((PyArrayObject*) py_g2)), n_max);


    /* compute fpt */
    DMBase* dm = nullptr;
    if (inv_leak > 0.0)
        dm = DMBase::create(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv,
                            dt, inv_leak);
    else
        dm = DMBase::create(mu, sig2, b_lo, b_up, b_lo_deriv, b_up_deriv, dt);
    dm->pdfseq(n_max, g1, g2);
    if (mnorm_bool) dm->mnorm(g1, g2);
    delete dm;

    /* create tuple to return */
    PyObject* py_tuple = PyTuple_New(2);
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
