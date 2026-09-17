#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Determine Prime Implicants of Random Forest Classifiers
# Copyright (c) 2022 Ashlin Iser, Karlsruhe Institute of Technology (KIT)
# SPDX-License-Identifier: MIT

from .._logging import eprint

from .encoder import DecisionTreeEncoder
from .wrapper import DecisionTreeWrapper


class DecisionTreeExplainer:

    def __init__(self, query, api, wrapper: DecisionTreeWrapper):
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
        #self.plot(cat_leafs, cat_imps)

    def explain_prediction(self, sample):
        prediction = self.wrapper.clf.predict([sample])[0]
        class_id = list(self.wrapper.clf.classes_).index(prediction)
        category = self.wrapper.class_name(class_id)
        reasons = self.encoder.explain_prediction(
            sample, self.implicants[category]
        )
        return category, reasons


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
            result = self.api.query(self.query + " and " + explanation["query"])
            eprint(explanation)
            eprint("Samples: {}".format(len(result)))
            size = explanation["features"]
            samples = len(result)
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
                plt.title("Family: {}, NCD Ratio: {:2f}".format(cat.upper(), mean(ncd_imps) / mean(ncd_leafs)))
                plt.boxplot([ncd_leafs, ncd_imps])
                plt.xticks(range(1, len(K)+1), K)
                plt.show()