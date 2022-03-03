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

from pysat.solvers import Glucose4 as Solver

from tree_wrapper import DecisionTreeWrapper

from solbert import compute_prime_implicants


class VariableProducer:
    def __init__(self):
        self.vars = 0

    def new_var(self):
        self.vars = self.vars + 1
        return self.vars


class DecisionTreeExplainer:

    def __init__(self, tree: DecisionTreeWrapper, vprod: VariableProducer = None):
        self.dtw = tree
        self.vprod = vprod if vprod != None else VariableProducer()
        self.vars = 0
        # class variables:
        self.vclasses = [ self.new_var() for _ in range(self.dtw.n_classes()) ]
        # node variables:
        self.vnodestrue = [ self.new_var() for _ in range(self.dtw.n_nodes()) ]
        self.vnodesfalse = [ self.new_var() for _ in range(self.dtw.n_nodes()) ]
        # value variables:
        self.vintervals = []
        for feat_id in range(self.dtw.n_features()):
            self.vintervals.append([ self.new_var() for _ in self.dtw.feature_values(feat_id) ])
        self.vintervall = []
        for feat_id in range(self.dtw.n_features()):
            self.vintervall.extend(self.vintervals[feat_id])
        self.clauses = self.encode()


    def new_var(self):
        return self.vprod.new_var()


    def class2var(self, class_id):
        return self.vclasses[class_id]

    def var2class(self, var_id):
        return self.vclasses.index(var_id)


    def node2var(self, node_id: int, tip: bool):
        return self.vnodestrue[node_id] if tip else self.vnodesfalse[node_id]

    def var2node(self, var_id, tip: bool):
        return self.vnodestrue.index(var_id) if tip else self.vnodesfalse.index(var_id)        


    def feat2vars(self, feat_id):
        return self.vintervals[feat_id]

    def var2val(self, var_id):
        for feat_id in range(self.dtw.n_features()):
            if var_id in self.vintervals[feat_id]:
                return (feat_id, self.vintervals[feat_id].index(var_id))
        assert False, "variable {} not found".format(var_id)


    def explain(self, targetclasses):
        targetclause = [ self.class2var(self.dtw.class_id(name)) for name in targetclasses ]

        return compute_prime_implicants(self.clauses + [ targetclause ], self.vintervall)


    def decode(self, implicant):
        query = []
        nfeats = 0
        ncases = 0
        for feat_id in range(self.dtw.n_features()):
            feat = self.dtw.feature_name(feat_id)
            prev = ncases
            for i in range(len(self.vintervals[feat_id])-1):
                m0 = -self.vintervals[feat_id][i] in implicant
                m1 = -self.vintervals[feat_id][i+1] in implicant
                if m0 != m1:
                    ncases = ncases + 1
                    thre = self.dtw.feature_value(feat_id, i)
                    form = "{:7f}".format(thre).rstrip('0').rstrip('.')
                    query.append("{} {} {}".format(feat, ">" if m0 else "<=", form))
            if prev != ncases:
                nfeats = nfeats + 1
        result = { "features": nfeats, "cases": ncases }
        if len(query) > 10:
            split = int(len(query)/2)
            query1 = " and ".join(query[:split])
            query2 = " and ".join(query[split:])
            result["query"] = "(" + query1 + ") and (" + query2 + ")"
        else:
            result["query"] = " and ".join(query)
        return result


    def encode(self):
        class_constraints = self.encode_class_constraints()
        node_constraints = self.encode_node_constraints()
        value_constraints = self.encode_value_constraints()
        return class_constraints + node_constraints + value_constraints


    # class implies leafs
    def encode_class_constraints(self):
        clauses = [ [ -v ] for v in self.vclasses ]
        for node in range(self.dtw.n_nodes()):
            if not self.dtw.is_inner_node(node):
                class_id = self.dtw.node_class(node)
                clauses[class_id].append(self.node2var(node, True))
        return clauses


    # child implies parent
    def encode_node_constraints(self):
        clauses = []
        for node in range(self.dtw.n_nodes()):
            left = self.dtw.left_child(node)
            right = self.dtw.right_child(node)
            if self.dtw.is_inner_node(node):
                clauses.append([ -self.node2var(left, True), self.node2var(node, True) ])
                clauses.append([ -self.node2var(left, False), self.node2var(node, True) ])
                clauses.append([ -self.node2var(right, True), self.node2var(node, False) ])
                clauses.append([ -self.node2var(right, False), self.node2var(node, False) ])
        return clauses


    # node disables values
    def encode_value_constraints(self):
        clauses = []
        for node in range(self.dtw.n_nodes()):
            if self.dtw.is_inner_node(node): 
                feat = self.dtw.node_feature(node)
                thre = self.dtw.node_threshold(node)
                split = 1+self.dtw.feature_values(feat).index(thre)
                vleft = self.vintervals[feat][:split]
                vright = self.vintervals[feat][split:]
                for v in vleft: # false disables left values:
                    clauses.append([ -self.node2var(node, False), v ])
                for v in vright: # true disables right values:
                    clauses.append([ -self.node2var(node, True), v ])
        return clauses


    def print(self, model, ranges=True, nodes=False, classes=False):
        if classes:
            for v in self.vclasses:
                if model[v-1] > 0:
                    class_id = self.var2class(v)
                    print("Class: " + self.dtw.class_name(class_id))
        if nodes:
            for v in self.vnodestrue:
                if model[v-1] > 0:
                    node = self.var2node(v, True)
                    feat = self.dtw.node_feature_name(node)
                    thre = self.dtw.node_threshold(node)
                    print("Node {id}: {f} <= {t}".format(id=node, f=feat, t=thre))
            for v in self.vnodesfalse:
                if model[v-1] > 0:
                    node = self.var2node(v, False)
                    feat = self.dtw.node_feature_name(node)
                    thre = self.dtw.node_threshold(node)
                    print("Node {id}: {f} > {t}".format(id=node, f=feat, t=thre))
        if ranges:
            for feat_id in range(self.dtw.n_features()):
                feat = self.dtw.feature_name(feat_id)
                if not all(model[v-1] < 0 for v in self.vintervals[feat_id]):
                    value_ranges = [ [-np.infty, np.infty] ]
                    for i in range(len(self.vintervals[feat_id])-1):
                        m0 = self.sat(model, self.vintervals[feat_id][i])
                        m1 = self.sat(model, self.vintervals[feat_id][i+1])
                        if m0 and not m1:
                            if value_ranges[-1][0] != -np.infty:
                                value_ranges.append([-np.infty, np.infty])
                            value_ranges[-1][0] = self.dtw.feature_value(feat_id, i)
                        elif not m0 and m1:
                            if value_ranges[-1][1] != np.infty:
                                value_ranges.append([-np.infty, np.infty])
                            value_ranges[-1][1] = self.dtw.feature_value(feat_id, i)
                    print("{} in {}".format(feat, ", ".join([ "({}, {}]".format(v[0], v[1]) for v in value_ranges ])))


