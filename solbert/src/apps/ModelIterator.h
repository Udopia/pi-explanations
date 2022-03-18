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

#include <iostream>

#include <vector>

#include "lib/ipasir.h"

#include "src/util/PyUtil.h"

#ifndef SRC_APPS_MODELITERATOR_H_
#define SRC_APPS_MODELITERATOR_H_

typedef struct ModelIterator {
    PyObject_HEAD
    void* solver;
    std::vector<int> projection;
} ModelIterator;

static PyObject* model_iterator_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject* pyformula;
    PyObject* pyinputs;

    PyArg_ParseTuple(args, "OO", &pyformula, &pyinputs);

    std::vector<std::vector<int>> formula = list_to_formula(pyformula);

    ModelIterator* mit = (ModelIterator*) type->tp_alloc(type, 0);

    // init sat solver
    mit->solver = ipasir_init();
    for (std::vector<int>& clause : formula) {
        for (int lit : clause) {
            ipasir_add(mit->solver, lit);
        }
        ipasir_add(mit->solver, 0);
    }

    mit->projection = list_to_vec(pyinputs);

    return (PyObject*) mit;
}

static void model_iterator_delete(ModelIterator* mit) {
    ipasir_release(mit->solver);
    // Py_TYPE(mit)->tp_free(mit);  // segfaults (TODO: study)
}

static PyObject* model_iterator_next(PyObject *self) {
    ModelIterator* mit = (ModelIterator*) self;

    if (ipasir_solve(mit->solver) == 10) {
        std::vector<int> model;

        for (int var : mit->projection) {
            if (ipasir_val(mit->solver, var) >= 0) {  // TODO: replace >= by > (and test difference)
                model.push_back(var);
            }
        }

        for (int lit : model) {
            ipasir_add(mit->solver, -lit);
        }
        ipasir_add(mit->solver, 0);

        PyObject* pym = pylist();
        for (int lit : model) {
            pylist(pym, lit);
        }

        return pym;
    } else {
        /* Raising of standard StopIteration exception with empty value. */
        PyErr_SetNone(PyExc_StopIteration);
        return nullptr;
    }
}

static PyTypeObject ModelIteratorType = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "solbert.ModelIterator", /*tp_name*/
    sizeof(ModelIterator), /*tp_basicsize*/
    0, /*tp_itemsize*/ 
    (destructor) model_iterator_delete, /*tp_dealloc*/ 
    0, /*tp_print*/ 0, /*tp_getattr*/ 0, /*tp_setattr*/ 0, /*tp_compare*/ 0, /*tp_repr*/ 0, /*tp_as_number*/ 0, /*tp_as_sequence*/ 
    0, /*tp_as_mapping*/ 0, /*tp_hash */ 0, /*tp_call*/ 0, /*tp_str*/ 0, /*tp_getattro*/ 0, /*tp_setattro*/ 0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to use tp_iter and tp_iternext fields. */
    "solbert model iterator object.", /* tp_doc */
    0, /* tp_traverse */ 0, /* tp_clear */ 0, /* tp_richcompare */ 0, /* tp_weaklistoffset */
    PyObject_SelfIter, /* tp_iter: __iter__() method */
    (iternextfunc) model_iterator_next, /* tp_iternext: next() method */
    0, /* tp_methods */ 0, /* tp_members */ 0, /* tp_getset */ 0, /* tp_base */ 0, /* tp_dict */
    0, /* tp_descr_get */ 0, /* tp_descr_set */ 0, /* tp_dictoffset */ 0, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    model_iterator_new, /* tp_new */
};

#endif  // SRC_APPS_MODELITERATOR_H_