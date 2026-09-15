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
#include <algorithm>

#include "structmember.h"

#include "lib/ipasir.h"

#include "src/util/PyUtil.h"

#ifndef SRC_APPS_MONOTONICCIRCUIT_H_
#define SRC_APPS_MONOTONICCIRCUIT_H_

typedef struct MonotonicCircuit {
    PyObject_HEAD
    int max_var;
    void* solver;
    std::vector<int> inputs;
    std::vector<std::vector<int>> prime_implicants;
} MonotonicCircuit;

/**
 * @brief Create new monotonic circuit object
 * 
 * Initialize with given CNF formula and list of input literals
 * 
 * @param type 
 * @param args formula (list of list of int), inputs (list of int)
 * @param kwargs 
 * @return PyObject* 
 */
static PyObject* monotonic_circuit_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    PyObject* pyformula;
    PyObject* pyinputs;

    PyArg_ParseTuple(args, "OO", &pyformula, &pyinputs);

    std::vector<std::vector<int>> formula = list_to_formula(pyformula);

    MonotonicCircuit* self = (MonotonicCircuit*) type->tp_alloc(type, 0);

    // init sat solver
    self->max_var = 0;
    self->solver = ipasir_init();
    for (std::vector<int>& clause : formula) {
        for (int lit : clause) {
            self->max_var = std::max(self->max_var, abs(lit));
            ipasir_add(self->solver, lit);
        }
        ipasir_add(self->solver, 0);
    }

    // init implication chain of disjoint root conjunctions:
    ipasir_add(self->solver, self->max_var);
    ipasir_add(self->solver, 0);

    self->inputs = list_to_vec(pyinputs);

    return (PyObject*) self;
}

static void monotonic_circuit_delete(MonotonicCircuit* self) {
    ipasir_release(self->solver);
    // Py_TYPE(self)->tp_free(self);  // segfaults (TODO: study)
}

/**
 * @brief Append root conjunction
 * 
 * The root gate of this monotonic circuit is an (extensible) disjunction of conjunctions.
 * This method appends a conjunction to the root "DNF" gate.
 * 
 * @param self 
 * @param args conjunction (list of int)
 * @return PyObject* 
 */
static PyObject* monotonic_circuit_append_root(MonotonicCircuit* self, PyObject* args) {
    PyObject* pyinputs;
    PyArg_ParseTuple(args, "O", &pyinputs);
    std::vector<int> root = list_to_vec(pyinputs);

    // extend root disjunction
    ipasir_add(self->solver, -self->max_var);
    int enc = ++(self->max_var);
    ipasir_add(self->solver, enc);
    ipasir_add(self->solver, ++(self->max_var));
    ipasir_add(self->solver, 0);

    // encode root conjunction
    for (int var : root) {
        ipasir_add(self->solver, -enc);
        ipasir_add(self->solver, var);
        ipasir_add(self->solver, 0);
    }

    Py_RETURN_NONE;
}

// vectors are sorted
static bool is_subset(std::vector<int> subset, std::vector<int> set) {
    if (subset.size() >= set.size()) {
        return false;
    }
    auto it = subset.begin();
    for (int elem : set) {
        if (*it < elem) {
            break;
        } else if (*it == elem) {
            ++it;
            if (it == subset.end()) {
                return true;
            }
        }
    }
    return false;
}


/**
 * @brief Enumerate prime implicants projected to input variables. 
 * Exploit that we have a positive monotonic circuit.
 * 
 * @param self 
 * @return PyObject* list of prime implicants
 */
static PyObject* update_prime_implicants(MonotonicCircuit* self, PyObject* Py_UNUSED(ignored)) {
    ipasir_assume(self->solver, -self->max_var); // fix end of extensible disjunction
    bool res = (ipasir_solve(self->solver) == 10);

    while (res) { // determine models
        while (res) { // minimize model
            std::vector<int> minim;
            std::vector<int> facts;

            for (int var : self->inputs) {
                if (ipasir_val(self->solver, var) >= 0) {
                    minim.push_back(-var);
                } else {
                    facts.push_back(-var);
                }
            }

            for (int lit : minim) {
                ipasir_add(self->solver, lit);
            }
            ipasir_add(self->solver, 0);

            for (int lit : facts) {
                ipasir_assume(self->solver, lit);
            }

            ipasir_assume(self->solver, -self->max_var); // fix end of extensible disjunction
            res = (ipasir_solve(self->solver) == 10);
            if (!res) {
                /*std::cout << "Found Prime Implicant: ";
                for (int lit : minim) std::cout << lit << " ";
                std::cout << std::endl;
                for (auto pi : self->prime_implicants) {
                    if (is_subset(minim, pi)) {
                        std::cout << " -> Subsumes: ";
                        for (int lit : pi) std::cout << lit << " ";
                        std::cout << std::endl;
                    }
                }*/
                auto end = std::remove_if(self->prime_implicants.begin(), self->prime_implicants.end(), [minim](std::vector<int>& elem) { return is_subset(minim, elem); });
                if (end < self->prime_implicants.end()) { 
                    std::cout << "Found new prime implicant. Subsuming " << std::distance(end, self->prime_implicants.end()) << " previous prime implicants" << std::endl;
                    self->prime_implicants.erase(end, self->prime_implicants.end());
                }
                self->prime_implicants.push_back(minim);
            }
        }
        ipasir_assume(self->solver, -self->max_var); // fix end of extensible disjunction
        res = (ipasir_solve(self->solver) == 10);
    }

    Py_RETURN_NONE;
}

static PyObject* get_primp(MonotonicCircuit* self, PyObject* Py_UNUSED(ignored)) {
    PyObject* obj = pylist();
    for (std::vector<int>& pi : self->prime_implicants) {
        PyObject* obj2 = pylist();
        for (int lit : pi) {
            pylist(obj2, lit);
        }
        pylist(obj, obj2);
    }

    return obj;
}
static PyMethodDef mci_methods[] = {
    { "append_root", (PyCFunction) monotonic_circuit_append_root, METH_VARARGS, "append a conjunction of root nodes" },
    { "update_prime_implicants", (PyCFunction) update_prime_implicants, METH_NOARGS, "update prime implicants" },
    { "get_primp", (PyCFunction) get_primp, METH_NOARGS, "get prime implicants" },
    { NULL }  /* Sentinel */
};

static PyTypeObject MonotonicCircuitType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "solbert.MonotonicCircuit", /*tp_name*/
    sizeof(MonotonicCircuit), /*tp_basicsize*/
    0, /*tp_itemsize*/ 
    (destructor) monotonic_circuit_delete, /*tp_dealloc*/ 
    0, /*tp_print*/ 0, /*tp_getattr*/ 0, /*tp_setattr*/ 0, /*tp_compare*/ 0, /*tp_repr*/ 0, /*tp_as_number*/ 0, /*tp_as_sequence*/ 
    0, /*tp_as_mapping*/ 0, /*tp_hash */ 0, /*tp_call*/ 0, /*tp_str*/ 0, /*tp_getattro*/ 0, /*tp_setattro*/ 0, /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /* tp_flags: Py_TPFLAGS_HAVE_ITER tells python to use tp_iter and tp_iternext fields. */
    PyDoc_STR("solbert monotonic circuit object."), /* tp_doc */
    0, /* tp_traverse */ 0, /* tp_clear */ 0, /* tp_richcompare */ 0, /* tp_weaklistoffset */
    0, /* tp_iter: __iter__() method */ 0, /* tp_iternext: next() method */
    mci_methods, /* tp_methods */ 
    0, /* tp_members */ 
    0, /* tp_getset */ 0, /* tp_base */ 0, /* tp_dict */
    0, /* tp_descr_get */ 0, /* tp_descr_set */ 0, /* tp_dictoffset */ 
    0, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    monotonic_circuit_new, /* tp_new */
};

#endif  // SRC_APPS_MONOTONICCIRCUIT_H_