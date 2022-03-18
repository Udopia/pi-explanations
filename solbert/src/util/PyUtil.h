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

#ifndef SRC_UTIL_PY_UTIL_H_
#define SRC_UTIL_PY_UTIL_H_

#include "Python.h"

static PyObject* pytype(int val) {
    return Py_BuildValue("i", val);
}

static PyObject* pytype(unsigned val) {
    return Py_BuildValue("I", val);
}

static PyObject* pytype(const char* val) {
    return Py_BuildValue("s", val);
}

static PyObject* pytype(float val) {
    return PyFloat_FromDouble(static_cast<double>(val));
}

static PyObject* pydict() {
    return PyDict_New();
}

template<typename T>
static void pydict(PyObject* dict, const char* key, T val) {
    PyDict_SetItem(dict, pytype(key), pytype(val));
}

static PyObject* pylist() {
    return PyList_New(0);
}

template<typename T>
static void pylist(PyObject* list, T val) {
    PyList_Append(list, pytype(val));
}

static void pylist(PyObject* list, PyObject* val) {
    PyList_Append(list, val);
}


static std::vector<int> list_to_vec(PyObject* list) {
    std::vector<int> vec;
    for (Py_ssize_t i = 0; i < PyList_Size(list); i++) {
        PyObject* elem = PyList_GetItem(list, i);
        long lit = PyLong_AsLong(elem);
        vec.push_back(static_cast<int>(lit));
    }
    return vec;
}

static std::vector<std::vector<int>> list_to_formula(PyObject* list) {
    std::vector<std::vector<int>> formula;
    for (Py_ssize_t i = 0; i < PyList_Size(list); i++) {
        PyObject* elem = PyList_GetItem(list, i);
        formula.push_back(list_to_vec(elem));
    }
    return formula;
}

#endif  // SRC_UTIL_PY_UTIL_H_
