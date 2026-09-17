#!/usr/bin/python3

# Determine Prime Implicants of Random Forest Classifiers
# Copyright (c) 2022 Ashlin Iser, Karlsruhe Institute of Technology (KIT)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
from gbd_core.api import GBD
from sklearn import ensemble, tree
from explain import FamilyExplainer, InterestingExplainer, PortfolioExplainer

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
        "/home/iser/git/gbd/gbd-data/cnf/meta.db",
        "/home/iser/git/gbd/gbd-data/cnf/base.db",
        "/home/iser/git/gbd/gbd-data/cnf/gate.db",
        # "/home/iser/git/gbd/gbd-data/sc2020.db",
        # "/home/iser/git/gbd/gbd-data/minisat.db"
    ]

    parser = ArgumentParser(description='Solbert')
    parser.add_argument("num", type=int)
    args = parser.parse_args()

    num = args.num or 1

    print(num)

    with GBD(databases) as api:
        seed = 0
        model = lambda : tree.DecisionTreeClassifier(random_state=seed)
        if num != 1:
            model = lambda : ensemble.RandomForestClassifier(random_state=seed, n_estimators=num)
        #explain_portfolio(model, api)
        explain_family(model, api)
        #explain_interesting(model, api)

if __name__ == '__main__':
    main()