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

from tree_wrapper import DecisionTreeWrapper
from tree_explainer import DecisionTreeExplainer

from forest_wrapper import RandomForestWrapper
from forest_explainer import RandomForestExplainer


class Explainer:

    def __init__(self, model_getter, api: GBD, df: pd.DataFrame, target, query):
        self.get_model = model_getter
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
        model = self.get_model()
        model.fit(xtrain, ytrain)
        ypred=model.predict(xtest)
        acc = accuracy_score(ytest, ypred)
        print("Accuracy: {}".format(acc))

    def explain(self):
        eprint("Training ...")
        model = self.get_model()
        model.fit(self.x, self.y)
        if isinstance(model, tree.DecisionTreeClassifier):
            wrapper = DecisionTreeWrapper(model, self.lhs, self.rhs)
            explainer = DecisionTreeExplainer(self.query, self.api, wrapper)
            explainer.report()
        elif isinstance(model, ensemble.RandomForestClassifier):
            wrapper = RandomForestWrapper(model, self.lhs, self.rhs)
            explainer = RandomForestExplainer(self.query, self.api, wrapper)
            explainer.report()
        else:
            eprint("Cannot explain models of type {}".format(type(model)))


class FamilyExplainer(Explainer):

    def __init__(self, model_getter, api: GBD):
        query = "track like %20% and family != unknown and family != agile and family unlike %random%"
        source = api.get_features("base_db") # + api.get_features("gate_db")
        df = api.query_search2(query, [], source + [ "family" ], replace=[ ("timeout", np.inf), ("memout", np.inf), ("empty", np.nan), ("failed", np.inf) ])
        Explainer.__init__(self, model_getter, api, df, "family", query)


class PortfolioExplainer(Explainer):

    def __init__(self, model_getter, api: GBD, solvers):
        notout = " or ".join([ "({s} != timeout and {s} != memout)".format(s=solver) for solver in solvers ])
        query = "track = main_2020 and ({})".format(notout)
        source = api.get_features("base_db") # + api.get_features("gate_db")
        df = api.query_search2(query, [], source + solvers, replace=[ ("timeout", np.inf), ("memout", np.inf), ("empty", np.nan), ("failed", np.inf) ])
        df["solver"] = "empty"
        for s in solvers:
            for idx, row in df.iterrows():
                if float(row[s]) == min(row[solvers].astype(float)):
                    row["solver"] = s
        df.drop(solvers, axis=1, inplace=True)
        Explainer.__init__(self, model_getter, api, df, "solver", query)
