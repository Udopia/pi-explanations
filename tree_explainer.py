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


class DecisionTreeExplainer:

    def __init__(self, query, api: GBD, wrapper: DecisionTreeWrapper):
        self.query = query
        self.api = api
        self.wrapper = wrapper
        self.encoder = DecisionTreeEncoder(wrapper)
        self.cats = self.wrapper.class_names
        self.implicants = self.encoder.explain()


    def print_implicants(self):
        cat_leafs = []
        cat_imps = []
        for cat in self.cats:
            (leafs, imps) = self.explain(cat)
            cat_leafs.append(leafs)
            cat_imps.append(imps)
        self.plot(cat_leafs, cat_imps)


    def explain(self, cat):
        imps = self.implicants[cat]
        eprint("-" * 42)
        eprint("Explaining category: {}".format(cat))
        leafs = self.wrapper.leaf_nodes(cat)
        eprint("Number of Leaf Nodes: {}".format(len(leafs)))
        eprint("Number of Prime Implicants: {}".format(len(imps)))
        cases = []
        for i, imp in enumerate(imps):
            explanation = self.encoder.decode(imp)
            cases.append(explanation["cases"])
        eprint("Leaf Depths:                 {}".format(str(sorted([ self.wrapper.node_depth(leaf) for leaf in leafs ]))))
        eprint("Implicant Case Distinctions: {}".format(str(sorted(cases))))
        # leaf depths and sample numbers
        L = []
        for leaf in leafs:
            depth = self.wrapper.node_depth(leaf)
            samples = self.wrapper.node_samples_total(leaf)
            L.append((depth, samples))
        L.sort(key = lambda x: x[1], reverse=True)
        # implicant size and sample numbers
        I = []
        for i, imp in enumerate(imps):
            explanation = self.encoder.decode(imp)
            hashes = self.api.query_search(self.query + " and " + explanation["query"])
            size = explanation["features"]
            samples = len(hashes)
            I.append((size, samples))
        I.sort(key = lambda x: x[1], reverse=True)
        print("Leaf Depths and Samples: " + str(L))
        print("Implicant Size and Samples: " + str(I))
        return (L, I)


    def plot(self, leaf_data, imp_data):
        from matplotlib import pyplot as plt
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