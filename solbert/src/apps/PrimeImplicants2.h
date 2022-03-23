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

#ifndef SRC_APPS_PRIMEIMPLICANTS2_H_
#define SRC_APPS_PRIMEIMPLICANTS2_H_

std::vector<std::vector<int>> get_prime_implicants2(std::vector<std::vector<int>> formula, std::vector<int> inputs) {
    // initialize enumerating solver
    void* S = ipasir_init();
    for (std::vector<int>& clause : formula) {
        for (int lit : clause) {
            ipasir_add(S, lit);
        }
        ipasir_add(S, 0);
    }

    std::vector<std::vector<int>> prime_implicants;

    bool result = (ipasir_solve(S) == 10);
    while (result) {
        // initialize minimizing solver
        void* S2 = ipasir_init();
        for (std::vector<int>& clause : formula) {
            for (int lit : clause) {
                if (ipasir_val(S, lit) >= 0) {
                    ipasir_add(S2, abs(lit));
                }
            }
            ipasir_add(S2, 0);
        }
        // minimize model
        result = (ipasir_solve(S2) == 10);
        while (result) {
            std::vector<int> minim;
            std::vector<int> facts;
            for (int var : inputs) {
                if (ipasir_val(S2, var) >= 0) {
                    minim.push_back(-var);
                } else {
                    facts.push_back(-var);
                }
            }

            for (int lit : minim) {
                ipasir_add(S2, lit);
            }
            ipasir_add(S2, 0);

            for (int lit : facts) {
                ipasir_add(S2, lit);
                ipasir_add(S2, 0);
            }

            result = (ipasir_solve(S2) == 10);
            if (!result) {
                ipasir_release(S2);
                std::vector<int> prim;
                for (int lit : minim) {
                    if (ipasir_val(S, lit) >= 0) {
                        prim.push_back(lit);
                    } else {
                        prim.push_back(-lit);
                    }
                }
                prime_implicants.push_back(prim);

                for (int lit : prim) {
                    //std::cout << lit << " ";
                    ipasir_add(S, -lit);
                }
                //std::cout << std::endl;
                ipasir_add(S, 0);
            }
        }
        result = (ipasir_solve(S) == 10);
    }

    ipasir_release(S);

    return prime_implicants;
}

#endif  // SRC_APPS_PRIMEIMPLICANTS2_H_