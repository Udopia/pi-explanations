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
from pyparsing import nested_expr
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from gbd_tool.gbd_api import GBD
from gbd_tool.util import eprint

from tree_explainer import DecisionTreeExplainer
from tree_wrapper import DecisionTreeWrapper
from forest_explainer import RandomForestExplainer
from forest_wrapper import RandomForestWrapper


class Explainer:

    def __init__(self, api: GBD, df: pd.DataFrame, target, query):
        self.api = api
        self.target = target
        self.query = query
        self.lhs = df #self.df.drop(self.df[self.df.hash.isin(exclude_hashes)].index)
        self.lhs.drop(["hash"], axis=1, inplace=True)
        self.rhs = self.lhs.pop(self.target).astype("category")
        self.x = np.nan_to_num(self.lhs.to_numpy().astype(np.float32), nan=-1)
        self.y = self.rhs.cat.codes.to_numpy()

    def train_test_accuracy(self, seed=0):
        eprint("Testing ...")
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, test_size=0.2, random_state=seed)
        model = tree.DecisionTreeClassifier(random_state=seed)
        model.fit(xtrain, ytrain)
        ypred=model.predict(xtest)
        acc = accuracy_score(ytest, ypred)
        print("Accuracy: {}".format(acc))

    def explain(self, seed=0):
        eprint("Training ...")
        model = tree.DecisionTreeClassifier(random_state=seed)
        model.fit(self.x, self.y)
        wrapper = DecisionTreeWrapper(model, self.lhs, self.rhs)
        explainer = DecisionTreeExplainer(wrapper)

        implicants = dict()
        for cat in list(self.rhs.cat.categories):
            eprint("Calculating prime implicants for category {}".format(str(cat)))
            implicants[cat] = explainer.explain([cat])
            implicants[cat].sort(key=len)

        for cat, imps in implicants.items():
            eprint("-" * 42)
            eprint("Explaining category: {}".format(cat))
            leafs = wrapper.leaf_nodes(cat)
            eprint("Number of Leaf Nodes: {}".format(len(leafs)))
            eprint("Number of Prime Implicants: {}".format(len(imps)))
            cases = []
            for i, imp in enumerate(imps):
                explanation = explainer.decode(imp)
                cases.append(explanation["cases"])
            eprint("Leaf Depths:                 {}".format(str(sorted([ wrapper.node_depth(leaf) for leaf in leafs ]))))
            eprint("Implicant Case Distinctions: {}".format(str(sorted(cases))))
            # leaf depths and sample numbers
            L = []
            for leaf in leafs:
                depth = wrapper.node_depth(leaf)
                samples = wrapper.node_samples_total(leaf)
                L.append((depth, samples))
            L.sort(key = lambda x: x[1], reverse=True)
            # implicant size and sample numbers
            I = []
            for i, imp in enumerate(imps):
                explanation = explainer.decode(imp)
                hashes = self.api.query_search(self.query + " and " + explanation["query"])
                size = explanation["features"]
                samples = len(hashes)
                I.append((size, samples))
            I.sort(key = lambda x: x[1], reverse=True)
            print("Leaf Depths and Samples: " + str(L))
            print("Implicant Size and Samples: " + str(I))
            #if len(L) >= 18:
            self.plot(L, I, cat)
                #eprint("-" * 21)
                #eprint("Implicant {}:".format(i))
                #eprint("Number of constrained features: {}".format(explanation["features"]))
                #eprint("Number of case distinctions: {}".format(explanation["cases"]))
                #eprint("Query: {}".format(explanation["query"]))
                #eprint("Number of Samples: {}".format(len(hashes)))

    def plot(self, L, I, cat):
        from matplotlib import pyplot
        leaf_x = [ leaf[0] for leaf in L ]
        leaf_y = [ leaf[1] for leaf in L ]
        imp_x = [ imp[0] for imp in I ]
        imp_y = [ imp[1] for imp in I ]
        #n = max(max(leaf_x), max(imp_x))
        fig, ax = pyplot.subplots()
        ax.set_xlim([0, max(max(leaf_x), max(imp_x)) + 1])
        ax.set_ylim([0, max(max(leaf_y), max(imp_y)) + 1])
        ax.set_xlabel("Number of Case Distinctions")
        ax.set_ylabel("Number of Training Samples")        
        pyplot.title(cat.upper())
        pyplot.scatter(leaf_x, leaf_y, marker='+', label='Leaf Nodes')
        pyplot.scatter(imp_x, imp_y, marker='x', label='Prime Implicants')
        pyplot.legend(loc='upper right')
        pyplot.show()


    def train_test_accuracy_forest(self, seed=0):
        eprint("Testing ...")
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, test_size=0.2, random_state=seed)
        model = ensemble.RandomForestClassifier(random_state=seed, n_estimators=3)
        model.fit(xtrain, ytrain)
        ypred=model.predict(xtest)
        acc = accuracy_score(ytest, ypred)
        print("Accuracy: {}".format(acc))

    def explain_forest(self, seed=0):
        eprint("Training ...")
        model = ensemble.RandomForestClassifier(random_state=seed, n_estimators=3)
        model.fit(self.x, self.y)
        wrapper = RandomForestWrapper(model, self.lhs, self.rhs)
        explainer = RandomForestExplainer(wrapper)
        #explainer.enumerate_valid_combinations()

        implicants = dict()
        for cat in list(self.rhs.cat.categories):
            eprint("Calculating prime implicants for category {}".format(str(cat)))
            implicants[cat] = explainer.explain([cat])
            implicants[cat].sort(key=len)

        for cat, imps in implicants.items():
            eprint("-" * 42)
            eprint("Explaining category: {}".format(cat))
            leafs_list = wrapper.leaf_nodes(cat)
            for i, leafs in enumerate(leafs_list):
                eprint("Number of Leaf Nodes for Tree {}: {}".format(i, len(leafs)))
            eprint("Number of prime implicants: {}".format(len(imps)))

            I = []
            for i, imp in enumerate(imps):
                explanation = explainer.decode(imp)
                hashes = self.api.query_search(self.query + " and " + explanation["query"])
                size = explanation["features"]
                samples = len(hashes)
                I.append((size, samples))
            I.sort(key = lambda x: x[1], reverse=True)
            print("Implicant Size and Samples: " + str(I))
            #if len(I) >= 20:
            self.plot_forest(I, cat)

            cases = []
            for i, imp in enumerate(imps):
                explanation = explainer.decode(imp)
                cases.append(explanation["cases"])
            #eprint("Leaf Depths:                 {}".format(str(sorted([ wrapper.node_depth(leaf) for leaf in leafs ]))))
            #eprint("Implicant Case Distinctions: {}".format(str(sorted(cases))))
            #count = 0
            #for i, imp in enumerate(imps):
            #    explanation = explainer.decode(imp)
            #    hashes = self.api.query_search(self.query + " and " + explanation["query"])
            #    if len(hashes) > 0:
            #        count = count + 1
            #        eprint("-" * 21)
            #        explanation = explainer.decode(imp)
            #        eprint("Implicant {}:".format(i))
            #        eprint("Number of constrained features: {}".format(explanation["features"]))
            #        eprint("Number of case distinctions: {}".format(explanation["cases"]))
            #        eprint("Query: {}".format(explanation["query"]))
            #        hashes = self.api.query_search(self.query + " and " + explanation["query"])
            #        eprint("Number of Samples: {}".format(len(hashes)))
            #eprint("Number of prime implicants with non-zero training instances: {}".format(count))

    def plot_forest(self, I, cat):
        from matplotlib import pyplot
        #leaf_x = [ leaf[0] for leaf in L ]
        #leaf_y = [ leaf[1] for leaf in L ]
        #imp_x = [ imp[0] for imp in I ]
        imp_x = range(len(I))
        imp_y = [ imp[1] for imp in I ]
        #n = max(max(leaf_x), max(imp_x))
        fig, ax = pyplot.subplots()
        #ax.set_xlim([0, max(max(leaf_x), max(imp_x)) + 1])
        #ax.set_ylim([0, max(max(leaf_y), max(imp_y)) + 1])
        ax.set_xlim([0, max(imp_x) + 1])
        ax.set_ylim([0, max(imp_y) + 1])
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel("Prime Implicant")        
        pyplot.title(cat.upper())
        #pyplot.scatter(leaf_x, leaf_y, marker='+', label='Leaf Nodes')
        pyplot.scatter(imp_x, imp_y, marker='x')
        pyplot.legend(loc='upper right')
        pyplot.show()


class FamilyExplainer(Explainer):

    def __init__(self, api: GBD):
        query = "track like %20% and family != unknown and family != agile and family unlike %random%"
        source = api.get_features("base_db") + api.get_features("gate_db")
        df = api.query_search2(query, [], source + [ "family" ], replace=[ ("timeout", np.inf), ("memout", np.inf), ("empty", np.nan), ("failed", np.inf) ])
        Explainer.__init__(self, api, df, "family", query)


class PortfolioExplainer(Explainer):

    def __init__(self, api: GBD, solvers):
        notout = " or ".join([ "({s} != timeout and {s} != memout)".format(s=solver) for solver in solvers ])
        query = "track = main_2020 and ({})".format(notout)
        source = api.get_features("base_db") + api.get_features("gate_db")
        df = api.query_search2(query, [], source + solvers, replace=[ ("timeout", np.inf), ("memout", np.inf), ("empty", np.nan), ("failed", np.inf) ])
        #conditions = [ df[s] == min(df[solvers]) for s in solvers]
        #df['solver'] = np.select(conditions, solvers)
        df["solver"] = "empty"
        for s in solvers:
            for idx, row in df.iterrows():
                if float(row[s]) == min(row[solvers].astype(float)):
                    row["solver"] = s
        df.drop(solvers, axis=1, inplace=True)
        Explainer.__init__(self, api, df, "solver", query)
