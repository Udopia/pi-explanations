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

#ifndef SRC_APPS_ENUMERATEMODELS_H_
#define SRC_APPS_ENUMERATEMODELS_H_

std::vector<std::vector<int>> get_models(std::vector<std::vector<int>> formula, std::vector<int> projection) {
    // initialize solver
    void* S = ipasir_init();
    for (std::vector<int>& clause : formula) {
        for (int lit : clause) {
            ipasir_add(S, lit);
        }
        ipasir_add(S, 0);
    }

    std::vector<std::vector<int>> models;

    while (ipasir_solve(S) == 10) {
        std::vector<int> model;

        for (int var : projection) {
            if (ipasir_val(S, var) >= 0) {
                model.push_back(var);
            }
        }

        for (int var : model) {
            ipasir_add(S, -var);
        }
        ipasir_add(S, 0);

        // std::cout << "Found Model " << models.size() << ": ";
        // for (int lit : model) std::cout << lit << " ";
        // std::cout << std::endl;
        models.push_back(model);
    }

    ipasir_release(S);

    return models;
}

#endif  // SRC_APPS_ENUMERATEMODELS_H_