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
from sklearn import tree

class DecisionTreeWrapper:

    def __init__(self, clf: tree.DecisionTreeClassifier, lhs: pd.DataFrame, rhs: pd.Categorical):
        self.clf = clf
        self.feature_names = list(lhs)
        self.class_names = list(rhs.cat.categories)
        self.depths = [ 0 ] * self.n_nodes()
        # calc depths:
        stack = [ 0 ]
        while len(stack) > 0:
            node = stack.pop()
            left = self.left_child(node)
            right = self.right_child(node)
            if left != right:  # inner node
                self.depths[left] = self.depths[right] = self.depths[node] + 1
                stack.extend((left, right))
        # calc values:
        self.feature_splits = [ [ np.infty, ] for _ in range(self.n_features()) ]
        for node in range(self.n_nodes()):
            if self.is_inner_node(node):
                feat = self.node_feature(node)
                thre = self.node_threshold(node)
                self.feature_splits[feat].append(thre)
        for values in self.feature_splits:
            values.sort()

    def leaf_nodes(self, class_name):
        nodes = []
        class_id = self.class_id(class_name)
        for node in range(self.n_nodes()):
            if not self.is_inner_node(node) and self.node_class(node) == class_id:
                nodes.append(node)
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

    def n_nodes(self):
        return self.clf.tree_.node_count

    def n_leafs(self):
        leafs = [ n for n in range(self.n_nodes()) if not self.is_inner_node(n) ]
        return len(leafs)

    def n_features(self):
        return len(self.feature_names)

    def n_classes(self):
        return len(self.class_names)

    def node_depth(self, node_id):
        return self.depths[node_id]

    def left_child(self, node_id):
        return self.clf.tree_.children_left[node_id]

    def right_child(self, node_id):
        return self.clf.tree_.children_right[node_id]

    def is_inner_node(self, node_id):
        return self.left_child(node_id) != self.right_child(node_id)

    def node_feature(self, node_id):
        return self.clf.tree_.feature[node_id]

    def node_threshold(self, node_id):
        return self.clf.tree_.threshold[node_id]

    def node_samples_total(self, node_id):
        return sum(self.node_samples_per_class(node_id))

    def node_samples_per_class(self, node_id):
        return list(self.clf.tree_.value[node_id][0])

    def node_samples(self, node_id):
        return max(self.node_samples_per_class(node_id))

    def node_class(self, node_id):
        return np.argmax(self.node_samples_per_class(node_id))

    def node_feature_name(self, node_id):
        return self.feature_name(self.node_feature(node_id))

    def node_class_name(self, node_id):
        return self.class_name(self.node_class(node_id))

    def print(self):
        for i in range(self.n_nodes()):
            left = self.left_child(i)
            right = self.right_child(i)
            node_str = "{space}N{node}: {value}".format(space=self.node_depth(i) * " ", node=i, value=self.class_name(self.node_class(i)))
            if left != right:  # inner node
                feat = self.node_feature_name(i)
                thre = self.node_threshold(i)
                node_ite = "({} <= {}) ? N{} : N{}".format(feat, "{:7f}".format(thre).rstrip('0').rstrip('.'), left, right)
                print("{} {}".format(node_str, node_ite))
            else:
                print(node_str)