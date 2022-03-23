/*************************************************************************************************
Solbert -- Copyright (c) 2022, Markus Iser, KIT - Karlsruhe Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************************************************/

#include <cstdio>
#include <vector>

#include "src/util/PyUtil.h"
#include "src/util/ResourceLimits.h"

#include "src/apps/PrimeImplicants.h"
#include "src/apps/EnumerateModels.h"

#include "src/apps/ModelIterator.h"
#include "src/apps/PrimeImplicants2.h"



static PyObject* compute_prime_implicants(PyObject* self, PyObject* arg) {
    PyObject* pyformula;
    PyObject* pyinputs;
    unsigned rlim = 0, mlim = 0;
    PyArg_ParseTuple(arg, "OO|II", &pyformula, &pyinputs, &rlim, &mlim);
    std::vector<std::vector<int>> formula = list_to_formula(pyformula);
    std::vector<int> inputs = list_to_vec(pyinputs);

    ResourceLimits limits(rlim, mlim);
    limits.set_rlimits();
    try {
        // compute prime implicants guarded
        std::vector<std::vector<int>> pis = get_prime_implicants(formula, inputs);
        PyObject* obj = pylist();
        for (std::vector<int>& pi : pis) {
            PyObject* obj2 = pylist();
            for (int lit : pi) {
                pylist(obj2, lit);
            }
            pylist(obj, obj2);
        }
        return obj;
    } catch (TimeLimitExceeded& e) {
        return pytype("timeout");
    } catch (MemoryLimitExceeded& e) {
        return pytype("memout");
    }
}


static PyObject* compute_prime_implicants2(PyObject* self, PyObject* arg) {
    PyObject* pyformula;
    PyObject* pyinputs;
    unsigned rlim = 0, mlim = 0;
    PyArg_ParseTuple(arg, "OO|II", &pyformula, &pyinputs, &rlim, &mlim);

    ResourceLimits limits(rlim, mlim);
    limits.set_rlimits();
    try {
        // compute prime implicants guarded
        std::vector<std::vector<int>> formula = list_to_formula(pyformula);
        std::vector<int> inputs = list_to_vec(pyinputs);
        
        std::vector<std::vector<int>> pis = get_prime_implicants2(formula, inputs);
        PyObject* obj = pylist();
        for (std::vector<int>& pi : pis) {
            PyObject* obj2 = pylist();
            for (int lit : pi) {
                pylist(obj2, lit);
            }
            pylist(obj, obj2);
        }
        return obj;
    } catch (TimeLimitExceeded& e) {
        return pytype("timeout");
    } catch (MemoryLimitExceeded& e) {
        return pytype("memout");
    }
}


static PyObject* enumerate_models(PyObject* self, PyObject* arg) {
    PyObject* pyformula;
    PyObject* pyinputs;
    unsigned rlim = 0, mlim = 0;
    PyArg_ParseTuple(arg, "OO|II", &pyformula, &pyinputs, &rlim, &mlim);
    std::vector<std::vector<int>> formula = list_to_formula(pyformula);
    std::vector<int> inputs = list_to_vec(pyinputs);

    ResourceLimits limits(rlim, mlim);
    limits.set_rlimits();
    try {
        // enumerate models guarded
        std::vector<std::vector<int>> pis = get_models(formula, inputs);
        PyObject* obj = pylist();
        for (std::vector<int>& pi : pis) {
            PyObject* obj2 = pylist();
            for (int lit : pi) {
                pylist(obj2, lit);
            }
            pylist(obj, obj2);
        }
        return obj;
    } catch (TimeLimitExceeded& e) {
        return pytype("timeout");
    } catch (MemoryLimitExceeded& e) {
        return pytype("memout");
    }
}


static PyMethodDef methods[] = {
    {"compute_prime_implicants", compute_prime_implicants, METH_VARARGS, "Compute Prime Implicants"},
    {"compute_prime_implicants2", compute_prime_implicants2, METH_VARARGS, "Compute Prime Implicants"},
    {"enumerate_models", enumerate_models, METH_VARARGS, "Enumerate Models"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef solbert = {
    PyModuleDef_HEAD_INIT, 
    "solbert", "Python Wrapper for Incremental SAT Applications", -1, methods
};

PyMODINIT_FUNC PyInit_solbert(void) {
    PyObject* mod = PyModule_Create(&solbert);

    Py_INCREF((PyObject*) &ModelIteratorType);
    PyModule_AddObject(mod, "model_iterator", (PyObject*) &ModelIteratorType);

    return mod;
}
