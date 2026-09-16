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

import numpy as np
import polars as pl
from gbd_core.api import GBD
from gbd_core.util import eprint
from sklearn import ensemble, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from solbert.forest import RandomForestExplainer, RandomForestWrapper
from solbert.tree import DecisionTreeExplainer, DecisionTreeWrapper


REPLACEMENTS = {
    "timeout": np.inf,
    "memout": np.inf,
    "empty": np.nan,
    "failed": np.inf,
}


class Explainer:

    def __init__(self, model_getter, api: GBD, df: pl.DataFrame, target, query):
        self.get_model = model_getter
        self.api = api
        self.target = target
        self.query = query
        self.lhs = df.drop("hash", self.target)
        self.rhs = df.get_column(self.target).cast(pl.Categorical)
        self.x = np.nan_to_num(
            self.lhs.with_columns(
                pl.col(pl.String).replace_strict(
                    REPLACEMENTS,
                    default=pl.col(pl.String).cast(pl.Float32, strict=False),
                    return_dtype=pl.Float32,
                )
            )
            .cast(pl.Float32, strict=False)
            .to_numpy(),
            nan=-1,
        )
        self.y = self.rhs.to_physical().to_numpy()


    def train_test_accuracy(self, seed=0):
        eprint("Testing ...")
        xtrain, xtest, ytrain, ytest = train_test_split(self.x, self.y, test_size=0.2, random_state=seed)
        model = self.get_model()
        model.fit(xtrain, ytrain)
        ypred=model.predict(xtest)
        acc = accuracy_score(ytest, ypred)
        print(f"Accuracy: {acc}")

    def explain(self):
        eprint("Training ...")
        model = self.get_model()
        model.fit(self.x, self.y)
        if isinstance(model, tree.DecisionTreeClassifier):
            wrapper = DecisionTreeWrapper(model, self.lhs, self.rhs)
            explainer = DecisionTreeExplainer(self.query, self.api, wrapper)
            explainer.print_implicants()
        elif isinstance(model, ensemble.RandomForestClassifier):
            wrapper = RandomForestWrapper(model, self.lhs, self.rhs)
            explainer = RandomForestExplainer(self.query, self.api, wrapper)
            #explainer.print_implicants()
        else:
            eprint(f"Cannot explain models of type {type(model)}")


class InterestingExplainer(Explainer):

    def __init__(self, model_getter, api: GBD):
        query = "minisat1m != emtpy"
        source = api.get_features("base_db") #+ api.get_features("gate_db")
        df = api.query(query, resolve=source + ["minisat1m"])
        Explainer.__init__(self, model_getter, api, df, "minisat1m", query)


class FamilyExplainer(Explainer):

    def __init__(self, model_getter, api: GBD):
        query = "track like %20% and family != unknown and family != agile and family unlike %random%"# and family like b%"
        source = api.get_features("base_db") #+ api.get_features("gate_db")
        df = api.query(query, resolve=source + ["family"])
        Explainer.__init__(self, model_getter, api, df, "family", query)


class PortfolioExplainer(Explainer):

    def __init__(self, model_getter, api: GBD, solvers):
        notout = " or ".join(
            f"({solver} != timeout and {solver} != memout)" for solver in solvers
        )
        query = f"track = main_2020 and ({notout})"
        source = api.get_features("base_db") + api.get_features("gate_db")
        df = api.query(query, resolve=source + solvers)
        solver_values = [
            pl.col(solver)
            .cast(pl.String)
            .replace_strict(
                REPLACEMENTS,
                default=pl.col(solver).cast(pl.Float64, strict=False),
                return_dtype=pl.Float64,
            )
            for solver in solvers
        ]
        df = df.with_columns(
            pl.concat_list(solver_values)
            .list.arg_min()
            .replace_strict(dict(enumerate(solvers)))
            .alias("solver")
        ).drop(solvers)
        Explainer.__init__(self, model_getter, api, df, "solver", query)
