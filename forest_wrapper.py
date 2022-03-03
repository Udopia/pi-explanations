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
from sklearn import ensemble
from tree_wrapper import DecisionTreeWrapper

class RandomForestWrapper:

    def __init__(self, clf: ensemble.RandomForestClassifier, lhs: pd.DataFrame, rhs: pd.Categorical):
        self.clf = clf
        self.feature_names = list(lhs)
        self.class_names = list(rhs.cat.categories)
        self.trees = [ DecisionTreeWrapper(tree, lhs, rhs) for tree in self.clf.estimators_ ]
        # calc values:
        self.feature_splits = [ [ np.infty, ] for _ in range(self.n_features()) ]
        for tree in self.trees:
            for feat, splits in enumerate(tree.feature_splits):
                self.feature_splits[feat].extend(splits)
        for values in self.feature_splits:
            values = sorted(set(values))
            #print(len(values))

    def leaf_nodes(self, class_name):
        nodes = []
        for tree in self.trees:
            nodes.append(tree.leaf_nodes(class_name))
        return nodes

    def feature_name(self, feat_id):
        return self.feature_names[feat_id]

    def feature_id(self, feat_name):
        return self.feature_names.index(feat_name)

    def feature_values(self, feat_id):
        return self.feature_splits[feat_id]

    def feature_value(self, feat_id, val_id):
        return self.feature_splits[feat_id][val_id]

    def class_name(self, class_id):
        return self.class_names[class_id]

    def class_id(self, class_name):
        return self.class_names.index(class_name)

    def n_nodes(self, tree_id):
        return self.trees[tree_id].n_nodes()

    def n_features(self):
        return len(self.feature_names)

    def n_classes(self):
        return len(self.class_names)

    def n_trees(self):
        return len(self.trees)

    def tree(self, tree_id):
        return self.trees[tree_id]

    def node_depth(self, tree_id, node_id):
        return self.tree(tree_id).node_depth(node_id)

    def left_child(self, tree_id, node_id):
        return self.tree(tree_id).left_child(node_id)

    def right_child(self, tree_id, node_id):
        return self.tree(tree_id).right_child(node_id)

    def is_inner_node(self, tree_id, node_id):
        return self.tree(tree_id).is_inner_node(node_id)

    def node_feature(self, tree_id, node_id):
        return self.tree(tree_id).node_feature(node_id)

    def node_threshold(self, tree_id, node_id):
        return self.tree(tree_id).node_threshold(node_id)

    def node_samples_total(self, tree_id, node_id):
        return self.tree(tree_id).node_samples_total(node_id)

    def node_samples_per_class(self, tree_id, node_id):
        return self.tree(tree_id).node_samples_per_class(node_id)

    def node_class(self, tree_id, node_id):
        return self.tree(tree_id).node_class(node_id)

    def node_feature_name(self, tree_id, node_id):
        return self.tree(tree_id).node_feature_name(node_id)

    def node_class_name(self, tree_id, node_id):
        return self.tree(tree_id).node_class_name(node_id)

    def print(self):
        for tree in self.trees:
            tree.print()