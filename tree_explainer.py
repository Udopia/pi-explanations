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

import numpy as np
import pandas as pd
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from gbd_tool.gbd_api import GBD
from gbd_tool.util import eprint

from tree_encoder import DecisionTreeEncoder
from tree_wrapper import DecisionTreeWrapper

from matplotlib import pyplot as plt


class DecisionTreeExplainer:

    def __init__(self, query, api: GBD, wrapper: DecisionTreeWrapper):
        self.query = query
        self.api = api
        self.wrapper = wrapper
        self.encoder = DecisionTreeEncoder(wrapper)
        self.cats = self.wrapper.class_names
        self.implicants = self.encoder.explain()
        self.nleafs = dict() # category -> n leafs
        self.nprime = dict() # category -> n prime implicants
        self.depths = dict() # category -> [depths]
        self.nsplits = dict() # category -> [nsplits by prime implicants]
        self.nsamples_leafs = dict() # category -> [samples per leaf]
        self.queries = dict() # category -> queries from prime implicants
        self.nsamples_prime = dict() # category -> [samples per prime implicant]
        for cat in self.cats:
            eprint("Explaining category: {}".format(cat))
            leafs = self.wrapper.leaf_nodes(cat)
            implicants = self.implicants[cat]
            self.nleafs[cat] = len(leafs)
            self.nprime[cat] = len(implicants)
            self.depths[cat] = sorted([ self.wrapper.node_depth(leaf) for leaf in leafs ])
            self.nsplits[cat] = sorted([self.encoder.decode(imp)["cases"] for imp in implicants])
            self.nsamples_leafs[cat] = sorted([self.wrapper.node_samples_total(leaf) for leaf in leafs])
            self.queries[cat] = [ self.encoder.decode(imp)["query"] for imp in implicants ]
            self.nsamples_prime[cat] = sorted([len(self.api.query_search(self.query + " and " + query)) for query in self.queries[cat]])

    def report(self):
        self.report_depth_vs_size()
        #self.report_numbers_of_samples()
        self.report_queries()

    def report_depth_vs_size(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Leaf or PI")
        ax.set_ylabel("Leaf Depth / PI Size")
        for cat in self.cats:
            plt.title(cat)
            plt.plot(self.depths[cat], label="Leaf Depths")
            plt.plot(self.nsplits[cat], label="PI Splits")
            plt.legend()
            plt.show()

    def report_numbers_of_samples(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Leaf or PI")
        ax.set_ylabel("Numbers of Samples")
        for cat in self.cats:
            plt.title(cat)
            plt.plot(self.nsamples_leafs[cat])
            plt.plot(self.nsamples_prime[cat])
            plt.legend(loc="upper left")
            plt.show()

    def report_queries(self):
        for cat in self.cats:
            print("PI Queries for", cat)
            print("\n".join(self.queries[cat]))



    def plot(self, leaf_data, imp_data):
        from statistics import mean
        sizes = []
        ncd_ratios = []
        for i in range(len(self.cats)):
            L = leaf_data[i]
            I = imp_data[i]
            ncd_leafs = [ leaf[0] for leaf in L ]
            cov_leafs = [ leaf[1] for leaf in L ]
            ncd_imps = [ imp[0] for imp in I ]
            sizes.append(sum(cov_leafs))
            ncd_ratios.append(mean(ncd_imps) / mean(ncd_leafs))
        fig, ax = plt.subplots()
        ax.set_xlabel("Family Size (Number of Samples)")
        ax.set_ylabel("NCD Ratio")
        plt.title("NCD Ratio vs. Family Size")
        plt.scatter(sizes, ncd_ratios, marker='x')
        plt.legend(loc='upper right')
        plt.show()

        for i in range(len(self.cats)):
            L = leaf_data[i]
            I = imp_data[i]
            cat = self.cats[i]
            ncd_leafs = [ leaf[0] for leaf in L ]
            ncd_imps = [ imp[0] for imp in I ]
            if sizes[i] > 0:#350:
                K = ["NCD (Leafs)", "NCD (Implicants)"]
                df = pd.concat([pd.Series(ncd_leafs), pd.Series(ncd_imps)], keys=K, ignore_index=True, axis=1)
                print(df)
                plt.title("Family: {}, NCD Ratio: {:2f}".format(cat.upper(), mean(ncd_imps) / mean(ncd_leafs)))
                plt.boxplot(df)
                plt.xticks(range(1, len(K)+1), K)
                plt.show()