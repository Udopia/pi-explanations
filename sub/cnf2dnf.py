#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Determine Prime Implicants of Random Forest Classifiers
# Copyright (C) 2022 Markus Iser, Karlsruhe Institute of Technology (KIT)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pysat.formula import CNF

import argparse
import multiprocessing
import pebble
from concurrent.futures import as_completed
import os
import numpy as np

from solbert import compute_prime_implicants2


def print_clauses(clauses):
    for clause in clauses:
        print("{} 0".format(" ".join(map(str, clause))))


def file_type(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError('{} is not a regular file'.format(path))
    if os.access(path, os.R_OK):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError('{} is not readable'.format(path))

    

# jeroslaw wang per literal
def nbest_by_jwl(formula, nvars, n):
    occp = [ 0, ] * (nvars + 1)
    occn = [ 0, ] * (nvars + 1)
    for clause in formula:
        if len(clause) > 1:
            for lit in clause:
                if lit < 0:
                    occn[abs(lit)] = occn[abs(lit)] + 1.0 / len(clause)
                else:
                    occp[abs(lit)] = occp[abs(lit)] + 1.0 / len(clause)
    occ = [ min(a, b) / max(a, b) if a + b != 0 else 0 for a, b in zip(occp, occn) ]
    return list(np.argpartition(occ, -n)).reverse()


def main():
    parser = argparse.ArgumentParser(description='Replace Subformulas by Prime Implicants')

    parser.add_argument('file', type=file_type, help='DIMACS CNF file to process')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--vars', type=int, nargs='+', help='List of variables to eliminate in given order')
    group.add_argument('-n', '--num', type=int, help='Number of variables to eliminate starting with most frequent variable')
    parser.add_argument('-t', '--tlim', type=int, default=10, help='Time-limit per variable (seconds)')
    parser.add_argument('-m', '--mlim', type=int, default=500, help='Memory-limit per variable (megabyte)')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Size of process pool')
    args = parser.parse_args()

    cnf = CNF(from_file=args.file)

    occc = [ [] for _ in range(cnf.nv+1) ]
    occv = [ set() for _ in range(cnf.nv+1) ]
    for clause in cnf.clauses:
        for lit in clause:
            occc[abs(lit)].append(clause)
            occv[abs(lit)].update([ abs(lit) for lit in clause ])

    variables = []
    if args.vars:
        variables = args.vars        
    else:
        occs = [ 0 for _ in range(cnf.nv+1) ]
        for clause in cnf.clauses:
            for lit in clause:
                occs[abs(lit)] = occs[abs(lit)] + 1.0 / (len(clause)**2)
        variables = list(np.argsort(occs)[1:])[::-1]

    replaced = []
    
    with pebble.ProcessPool(max_workers=min(multiprocessing.cpu_count(), args.jobs), max_tasks=1) as p:
        futures = { p.schedule(compute_prime_implicants2, (occc[v], list(occv[v]), args.tlim, args.mlim)): v for v in variables }
        for f in as_completed(list(futures.keys())):
            try:
                prim = f.result()
                v = futures[f]
                if prim == "timeout" or prim == "memout":
                    print("c skipped elimination of variable {} due to {}".format(v, prim))
                else:
                    print("c found {} prime implicants for {} clauses containing variable {}".format(len(prim), len(occc[v]), v))
                    dnf = [ "{} 0".format(" ".join(map(str, term))) for term in prim ]
                    print("DNF {} 0".format(" ".join(dnf)))
                    replaced.extend([v, -v])
                    if len(replaced) >= 2*args.num:
                        break
            except pebble.ProcessExpired as e:
                f.cancel()
                print("{}: {}".format(e.__class__.__name__, e))
            except Exception as e:
                f.cancel()
                print("{}: {}".format(e.__class__.__name__, e))

        for f in futures.keys():
            f.cancel()

    for clause in cnf.clauses:
        if not any(lit in clause for lit in replaced):
            print("{} 0".format(" ".join(map(str, clause))))

    


if __name__ == '__main__':
    main()