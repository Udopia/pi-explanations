#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Determine Prime Implicants of Random Forest Classifiers
# Copyright (C) 2022 Ashlin Iser, Karlsruhe Institute of Technology (KIT)
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

from argparse import ArgumentParser
from ssl import ALERT_DESCRIPTION_UNEXPECTED_MESSAGE
from gbd_tool.gbd_api import GBD
from sklearn import tree, ensemble
from explain import FamilyExplainer, PortfolioExplainer, InterestingExplainer


def explain_portfolio(model_getter, api: GBD):
    ex = PortfolioExplainer(model_getter, api, [ "kissat_unsat", "relaxed_newtech" ])
    ex.train_test_accuracy()
    ex.explain()


def explain_family(model_getter, api: GBD):
    ex = FamilyExplainer(model_getter, api)
    ex.train_test_accuracy()
    ex.explain()


def explain_interesting(model_getter, api: GBD):
    ex = InterestingExplainer(model_getter, api)
    ex.train_test_accuracy()
    ex.explain()


def main():
    databases = [
        "/home/iser/git/gbd-data/meta.db",
        "/home/iser/git/gbd-data/base.db",
        "/home/iser/git/gbd-data/gate.db",
        "/home/iser/git/gbd-data/sc2020.db",
        "/home/iser/git/gbd-data/minisat.db"
    ]

    parser = ArgumentParser(description='Solbert')
    parser.add_argument("num", type=int)
    args = parser.parse_args()

    num = args.num or 1

    print(num)

    with GBD(databases, jobs=8) as api:
        seed = 0
        trees = 2
        model = lambda : tree.DecisionTreeClassifier(random_state=seed)
        if num != 1:
            model = lambda : ensemble.RandomForestClassifier(random_state=seed, n_estimators=num)
        #explain_portfolio(model, api)
        explain_family(model, api)
        #explain_interesting(model, api)

if __name__ == '__main__':
    main()