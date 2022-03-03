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
from explain import FamilyExplainer, PortfolioExplainer

def main():
    databases = [
        "/home/iser/git/gbd-data/meta.db",
        "/home/iser/git/gbd-data/base.db",
        "/home/iser/git/gbd-data/gate.db",
        "/home/iser/git/gbd-data/sc2020.db"
    ]

    with GBD(databases, jobs=8) as api:
        #ex = FamilyExplainer(api)
        ex = PortfolioExplainer(api, [ "kissat_unsat", "relaxed_newtech" ])
        #ex.train_test_accuracy()
        #ex.explain()
        ex.train_test_accuracy_forest()
        ex.explain_forest()

if __name__ == '__main__':
    main()