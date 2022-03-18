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
import multiprocessing

from forest_wrapper import RandomForestWrapper
from tree_encoder import VariableProducer

from solbert import compute_prime_implicants
from solbert import enumerate_models
from solbert import model_iterator

class RandomForestEncoder:

    def __init__(self, forest: RandomForestWrapper):
        self.rfw = forest
        self.vprod = VariableProducer()
        # node variables:
        self.vnodestrue = []
        self.vnodesfalse = []
        for tree_id in range(self.rfw.n_trees()):
            self.vnodestrue.append([ self.new_var() for _ in range(self.rfw.n_nodes(tree_id)) ])
            self.vnodesfalse.append([ self.new_var() for _ in range(self.rfw.n_nodes(tree_id)) ])
        # value variables:
        self.vintervals = []
        for feat_id in range(self.rfw.n_features()):
            self.vintervals.append([ self.new_var() for _ in self.rfw.feature_values(feat_id) ])
        self.vintervall = []
        for feat_id in range(self.rfw.n_features()):
            self.vintervall.extend(self.vintervals[feat_id])
        # deactivation variables:
        self.vdeactivateleft = []
        for feat_id in range(self.rfw.n_features()):
            self.vdeactivateleft.append([ self.new_var() for _ in self.rfw.feature_values(feat_id) ])
        self.vdeactivateright = []
        for feat_id in range(self.rfw.n_features()):
            self.vdeactivateright.append([ self.new_var() for _ in self.rfw.feature_values(feat_id) ])
        # base encoding
        self.clauses = self.encode()
        total_comb = 1
        for tree in self.rfw.trees:
            total_comb = total_comb * tree.n_leafs()
        print("Total Combinations: {}".format(total_comb))
        print("Computing Valid Combinations ...")
        self.comb = [ [ ] for _ in range(self.rfw.n_classes()) ] 
        self.enumerate_valid_combinations()
        print("Valid Combinations: {}".format(sum(len(valid_combs) for valid_combs in self.comb)))
        self.pool = multiprocessing.Pool(processes=4)

    def __del__(self):
        self.pool.terminate()


    def new_var(self):
        return self.vprod.new_var()


    def node2var(self, tree_id, node_id: int, tip: bool):
        return self.vnodestrue[tree_id][node_id] if tip else self.vnodesfalse[tree_id][node_id]

    def var2tree(self, var_id, tip: bool):
        for tree_id in range(self.rfw.n_trees()):
            if tip and var_id in self.vnodestrue[tree_id]:
                return tree_id
            if not tip and var_id in self.vnodesfalse[tree_id]:
                return tree_id

    def var2node(self, tree_id, var_id, tip: bool):
        return self.vnodestrue[tree_id].index(var_id) if tip else self.vnodesfalse[tree_id].index(var_id)


    def feat2vars(self, feat_id):
        return self.vintervals[feat_id]

    def var2val(self, var_id):
        for feat_id in range(self.rfw.n_features()):
            if var_id in self.vintervals[feat_id]:
                return (feat_id, self.vintervals[feat_id].index(var_id))
        assert False, "variable {} not found".format(var_id)


    def explain(self):
        implicants = dict()
        for cat in range(self.rfw.n_classes()):
            target = self.encode_target_class(cat)
            implicants[cat] = compute_prime_implicants(self.clauses + target, self.vintervall)
            implicants[cat].sort(key=len)
        return implicants


    def explain_parallel(self):
        results = list()
        for class_id in range(self.rfw.n_classes()):
            target = self.encode_target_class(class_id)
            res = self.pool.apply_async(compute_prime_implicants, (self.clauses + target, self.vintervall))
            results.append(res)
        implicants = dict()
        for class_id in range(self.rfw.n_classes()):
            cat = self.rfw.class_name(class_id)
            implicants[cat] = results[class_id].get()
            implicants[cat].sort(key=len)
        return implicants


    def encode_target_class(self, class_id):
        root_clause = []
        term_clauses = []
        for term in self.comb[class_id]:
            enc = self.new_var()
            root_clause.append(enc)
            for lit in term:
                term_clauses.append([ -enc, lit ])
        return term_clauses + [ root_clause ]


    def decode(self, implicant):
        query = []
        nfeats = 0
        ncases = 0
        for feat_id in range(self.rfw.n_features()):
            feat = self.rfw.feature_name(feat_id)
            prev = ncases
            for i in range(len(self.vintervals[feat_id])-1):
                m0 = -self.vintervals[feat_id][i] in implicant
                m1 = -self.vintervals[feat_id][i+1] in implicant
                if m0 != m1:
                    ncases = ncases + 1
                    thre = self.rfw.feature_value(feat_id, i)
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
        node_constraints = self.encode_node_constraints()
        value_constraints = self.encode_value_constraints()
        return node_constraints + value_constraints


    # child implies parent
    def encode_node_constraints(self):
        clauses = []
        for tree_id in range(self.rfw.n_trees()):
            for node_id in range(self.rfw.n_nodes(tree_id)):
                if self.rfw.is_inner_node(tree_id, node_id):
                    left = self.rfw.left_child(tree_id, node_id)
                    right = self.rfw.right_child(tree_id, node_id)
                    clauses.append([ -self.node2var(tree_id, left, True), self.node2var(tree_id, node_id, True) ])
                    clauses.append([ -self.node2var(tree_id, left, False), self.node2var(tree_id, node_id, True) ])
                    clauses.append([ -self.node2var(tree_id, right, True), self.node2var(tree_id, node_id, False) ])
                    clauses.append([ -self.node2var(tree_id, right, False), self.node2var(tree_id, node_id, False) ])
        return clauses


    # node disables values
    def encode_value_constraints(self):
        clauses = []
        # encode value constraints
        for tree_id in range(self.rfw.n_trees()):
            for node_id in range(self.rfw.n_nodes(tree_id)):
                if self.rfw.is_inner_node(tree_id, node_id): 
                    feat = self.rfw.node_feature(tree_id, node_id)
                    thre = self.rfw.node_threshold(tree_id, node_id)
                    split = 1+self.rfw.feature_values(feat).index(thre)
                    clauses.append([ -self.node2var(tree_id, node_id, False), self.vdeactivateleft[feat][split-1] ])
                    if split < len(self.vdeactivateright[feat]):
                        clauses.append([ -self.node2var(tree_id, node_id, True), self.vdeactivateright[feat][split] ])
        # encode deactivation constraints
        for feat_id in range(self.rfw.n_features()):
            for i in range(1, len(self.vintervals[feat_id])):
                clauses.append([ self.vdeactivateleft[feat_id][i-1], -self.vdeactivateleft[feat_id][i] ])
                clauses.append([ -self.vdeactivateright[feat_id][i-1], self.vdeactivateright[feat_id][i] ])
        for feat_id in range(self.rfw.n_features()):
            for i in range(len(self.vintervals[feat_id])):
                clauses.append([ -self.vdeactivateleft[feat_id][i], self.vintervals[feat_id][i] ])
                clauses.append([ -self.vdeactivateright[feat_id][i], self.vintervals[feat_id][i] ])
        return clauses


    def get_class(self, comb):
        probs = [ 0, ] * self.rfw.n_classes()
        for v in comb:
            tree_id = self.var2tree(v, True)
            node_id = self.var2node(tree_id, v, True)
            samples_total = self.rfw.node_samples_total(tree_id, node_id)
            samples_per_class = self.rfw.node_samples_per_class(tree_id, node_id)
            for class_id in range(self.rfw.n_classes()):
                proba = samples_per_class[class_id] / samples_total
                probs[class_id] = probs[class_id] + proba
        return np.argmax(probs)


    def get_leaf_vars(self):
        leafs = []
        for tree_id in range(self.rfw.n_trees()):
            for node_id in range(self.rfw.n_nodes(tree_id)):
                if not self.rfw.is_inner_node(tree_id, node_id):
                    leafs.append(self.node2var(tree_id, node_id, True))
        return leafs


    def enumerate_valid_combinations(self):
        clauses = self.clauses + self.encode_combination_constraints()
        project = self.get_leaf_vars()
        for comb in model_iterator(clauses, project):
            class_id = self.get_class(comb)
            self.comb[class_id].append(comb)


    def encode_combination_constraints(self):
        clauses = []
        # at least one leaf per tree:
        for tree_id in range(self.rfw.n_trees()):
            clause = []
            for node_id in range(self.rfw.n_nodes(tree_id)):
                if not self.rfw.is_inner_node(tree_id, node_id):
                    clause.append(self.node2var(tree_id, node_id, True))
            clauses.append(clause)
        # at least one value per feature:
        for feat_id in range(self.rfw.n_features()):
            clause = [ -v for v in self.vintervals[feat_id] ]
            clauses.append(clause)
        return clauses


