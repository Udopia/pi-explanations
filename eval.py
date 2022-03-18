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

from gbd_tool.gbd_api import GBD
from sklearn import tree, ensemble
from explain import FamilyExplainer, PortfolioExplainer


def explain_portfolio(model_getter, api: GBD):
    ex = PortfolioExplainer(model_getter, api, [ "kissat_unsat", "relaxed_newtech" ])
    ex.train_test_accuracy()
    ex.explain()


def explain_family(model_getter, api: GBD):
    ex = FamilyExplainer(model_getter, api)
    ex.train_test_accuracy()
    ex.explain()


def main():
    databases = [
        "/home/iser/git/gbd-data/meta.db",
        "/home/iser/git/gbd-data/base.db",
        "/home/iser/git/gbd-data/gate.db",
        "/home/iser/git/gbd-data/sc2020.db"
    ]

    with GBD(databases, jobs=8) as api:
        seed = 0
        trees = 3
        get_decision_tree = lambda : tree.DecisionTreeClassifier(random_state=seed)
        get_random_forest = lambda : ensemble.RandomForestClassifier(random_state=seed, n_estimators=trees)
        #explain_portfolio(get_decision_tree, api)
        explain_portfolio(get_random_forest, api)
        #explain_family(get_decision_tree, api)
        #explain_family(get_random_forest, api)

if __name__ == '__main__':
    main()