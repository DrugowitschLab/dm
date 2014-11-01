/*
 * Python wrapper to ddm_fpt_lib
 *
 * See README.md for documentation.
 */

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


/* method (t, b) = rand_sym(mu, bound, dt, n[, seed]) */
static PyObject* ddmmod_rand_sym(PyObject* self, PyObject* args)
{
    /* process arguments */
    PyArrayObject *py_mu, *py_bound;
    double dt;
    int n, rngseed = 0;
    if (!PyArg_ParseTuple(args, "O!O!di|i", &PyArray_Type, &py_mu,
                          &PyArray_Type, &py_bound, &dt, &n, &rngseed))
        return NULL;
    if (!(is1DDoubleArray(py_mu) && is1DDoubleArray(py_bound)))
        return NULL;
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt needs to be larger than 0");
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n needs to be larger than 0");
        return NULL;
    }
    ExtArray mu(ExtArray::shared_noowner((double*) PyArray_DATA(py_mu)),
                PyArray_DIM(py_mu, 0));
    ExtArray bound(ExtArray::shared_noowner((double*) PyArray_DATA(py_bound)),
                   PyArray_DIM(py_bound, 0));

    /* get output length and reserve space */
    npy_intp out_size[1] = { n };
    PyObject* py_t = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_t == NULL) return NULL;
    PyObject* py_b = PyArray_SimpleNew(1, out_size, NPY_BOOL);
    if (py_b == NULL) {
        Py_DECREF(py_t);
        return NULL;
    }
    npy_double* t = (npy_double*) PyArray_DATA((PyArrayObject*) py_t);
    npy_bool* b = (npy_bool*) PyArray_DATA((PyArrayObject*) py_b);

    /* perform sampling */
    DMBase* dm = DMBase::create(mu, bound, dt);
    DMBase::rngeng_t rngeng;
    if (rngseed == 0) rngeng.seed(std::random_device()());
    else rngeng.seed(rngseed);
    for (int i = 0; i < n; ++i) {
        DMSample s = dm->rand(rngeng);
        t[i] = s.t();
        b[i] = s.upper_bound() ? NPY_TRUE : NPY_FALSE;
    }
    delete dm;

    /* create tuple to return */
    PyObject* py_tuple = PyTuple_New(2);
    if (py_tuple == NULL) goto memory_fail;
    PyTuple_SET_ITEM(py_tuple, 0, py_t);
    PyTuple_SET_ITEM(py_tuple, 1, py_b);
    return py_tuple;

memory_fail:
    Py_DECREF(py_t);
    Py_DECREF(py_b);
    PyErr_SetString(PyExc_MemoryError, "out of memory");
    return NULL;
}


/* method (t, b) = rand_asym(mu, b_lo, b_up, dt, n[, seed]) */
static PyObject* ddmmod_rand_asym(PyObject* self, PyObject* args)
{
    /* process arguments */
    PyArrayObject *py_mu, *py_b_lo, *py_b_up;
    double dt;
    int n, rngseed = 0;
    if (!PyArg_ParseTuple(args, "O!O!O!di|i", &PyArray_Type, &py_mu,
                          &PyArray_Type, &py_b_lo, &PyArray_Type, &py_b_up,
                          &dt, &n, &rngseed))
        return NULL;
    if (!(is1DDoubleArray(py_mu) && 
          is1DDoubleArray(py_b_lo) && is1DDoubleArray(py_b_up)))
        return NULL;
    if (dt <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "dt needs to be larger than 0");
        return NULL;
    }
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "n needs to be larger than 0");
        return NULL;
    }
    ExtArray mu(ExtArray::shared_noowner((double*) PyArray_DATA(py_mu)),
                PyArray_DIM(py_mu, 0));
    ExtArray b_lo(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_lo)),
                   PyArray_DIM(py_b_lo, 0));
    ExtArray b_up(ExtArray::shared_noowner((double*) PyArray_DATA(py_b_up)),
                   PyArray_DIM(py_b_up, 0));
    if (b_lo[0] >= 0.0) {
        PyErr_SetString(PyExc_ValueError, "b_lo[0] needs to be negative");
        return NULL;
    }
    if (b_up[0] <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "b_up[0] needs to be positive");
        return NULL;
    }

    /* get output length and reserve space */
    npy_intp out_size[1] = { n };
    PyObject* py_t = PyArray_SimpleNew(1, out_size, NPY_DOUBLE);
    if (py_t == NULL) return NULL;
    PyObject* py_b = PyArray_SimpleNew(1, out_size, NPY_BOOL);
    if (py_b == NULL) {
        Py_DECREF(py_t);
        return NULL;
    }
    npy_double* t = (npy_double*) PyArray_DATA((PyArrayObject*) py_t);
    npy_bool* b = (npy_bool*) PyArray_DATA((PyArrayObject*) py_b);

    /* perform sampling */
    DMBase* dm = DMBase::create(mu, ExtArray::const_array(1.0), b_lo, b_up,
                                ExtArray::const_array(0.0), ExtArray::const_array(0.0), dt);
    DMBase::rngeng_t rngeng;
    if (rngseed == 0) rngeng.seed(std::random_device()());
    else rngeng.seed(rngseed);
    for (int i = 0; i < n; ++i) {
        DMSample s = dm->rand(rngeng);
        t[i] = s.t();
        b[i] = s.upper_bound() ? NPY_TRUE : NPY_FALSE;
    }
    delete dm;

    /* create tuple to return */
    PyObject* py_tuple = PyTuple_New(2);
    if (py_tuple == NULL) goto memory_fail;
    PyTuple_SET_ITEM(py_tuple, 0, py_t);
    PyTuple_SET_ITEM(py_tuple, 1, py_b);
    return py_tuple;

memory_fail:
    Py_DECREF(py_t);
    Py_DECREF(py_b);
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
    {"rand_sym", (PyCFunction) ddmmod_rand_sym, METH_VARARGS,
     "Draws first-passage time and boundary samples from symmetric diff. model"},
    {"rand_asym", (PyCFunction) ddmmod_rand_asym, METH_VARARGS,
     "Draws first-passage time and boundary samples from asymmetric diff. model"},
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
