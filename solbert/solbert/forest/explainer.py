#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Determine Prime Implicants of Random Forest Classifiers
# Copyright (c) 2022 Ashlin Iser, Karlsruhe Institute of Technology (KIT)
# SPDX-License-Identifier: MIT

import time

from .._logging import eprint

from .encoder import RandomForestEncoder
from .wrapper import RandomForestWrapper


class RandomForestExplainer:

    def __init__(self, query, api, wrapper: RandomForestWrapper):
        self.query = query
        self.api = api
        self.wrapper = wrapper
        self.encoder = RandomForestEncoder(wrapper)
        self.cats = self.wrapper.class_names
        self.implicants = {}

    def _implicants_for(self, category):
        if category not in self.implicants:
            start = time.time()
            self.implicants[category] = self.encoder.explain_class(category)
            end = time.time()
            eprint("\n -> Seconds to explain: {}\n".format(round(end - start)))
        return self.implicants[category]


    def print_implicants(self):
        #cat_leafs = []
        #cat_imps = []
        for cat in self.cats:
            eprint("-" * 42)
            eprint("Explaining category: {}".format(cat))
            #(leafs, imps) = self.explain(cat)
            #cat_leafs.append(leafs)
            #cat_imps.append(imps)
            eprint("Number of prime implicants: {}".format(len(self._implicants_for(cat))))
        #self.plot(cat_leafs, cat_imps)

    def explain_prediction(self, sample):
        prediction = self.wrapper.clf.predict([sample])[0]
        class_id = list(self.wrapper.clf.classes_).index(prediction)
        category = self.wrapper.class_name(class_id)
        reasons = self.encoder.explain_prediction(
            sample, self._implicants_for(category)
        )
        return category, reasons


    def explain(self, cat):
        imps = self._implicants_for(cat)
        I = []
        for imp in imps:
            explanation = self.encoder.decode(imp)
            result = self.api.query(self.query + " and " + explanation["query"])
            size = explanation["features"]
            samples = len(result)
            I.append((size, samples))
        I.sort(key = lambda x: x[1], reverse=True)        
        print("Class {} Implicant Size and Samples: {}".format(cat, str(I)))
        return ([], I)


    def plot(self, leaf_data, imp_data):
        from matplotlib import pyplot
        for i in range(len(self.cats)):
            #L = leaf_data[i]
            I = imp_data[i]
            imp_x = range(len(I))
            imp_y = [ imp[1] for imp in I ]
            fig, ax = pyplot.subplots()
            ax.set_xlim((0, max(imp_x, default=0) + 1))
            ax.set_ylim((0, max(imp_y, default=0) + 1))
            ax.set_ylabel("Number of Covered Training Samples")
            ax.set_xlabel("Prime Implicant")        
            pyplot.title(self.cats[i].upper())
            pyplot.scatter(imp_x, imp_y, marker='x')
            pyplot.legend(loc='upper right')
            pyplot.show()
