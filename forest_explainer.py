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

import time

from gbd_tool.gbd_api import GBD
from gbd_tool.util import eprint

from forest_encoder import RandomForestEncoder
from forest_wrapper import RandomForestWrapper

from matplotlib import pyplot as plt


class RandomForestExplainer:

    def __init__(self, query, api: GBD, wrapper: RandomForestWrapper):
        self.query = query
        self.api = api
        self.wrapper = wrapper
        self.encoder = RandomForestEncoder(wrapper)
        self.cats = self.wrapper.class_names
        start = time.time()
        self.implicants = self.encoder.explain_parallel()
        end = time.time()
        eprint("Seconds to explain: {}".format(round(end - start)))
        self.nprime = dict() # category -> n prime implicants
        self.nsamples = dict() # cat -> [nsamples per prime implicant]
        self.pi_sizes = dict() # cat -> [sizes of prime implicants]
        for cat in self.cats:
            self.nsamples[cat] = []
            self.pi_sizes[cat] = []
            self.nprime[cat] = len(self.implicants[cat])
            for imp in self.implicants[cat]:
                explanation = self.encoder.decode(imp)
                size = explanation["features"]
                samples = len(self.api.query_search(self.query + " and " + explanation["query"]))
                self.nsamples[cat].append(samples)
                self.pi_sizes[cat].append(size)
            self.nsamples[cat].sort(reverse=True)
            self.pi_sizes[cat].sort()


    def report(self):
        self.report_pi_sizes()
        self.report_numbers_of_samples()

    def report_pi_sizes(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Prime Implicant")
        ax.set_ylabel("Numbers of Case Distinctions")
        plt.title("Sizes of Prime Implicants")
        for cat in self.cats:
            plt.plot(self.pi_sizes[cat], label=cat)
        plt.legend(loc="upper left")
        plt.show()

    def report_numbers_of_samples(self):
        fig, ax = plt.subplots()
        ax.set_xlabel("Prime Implicant")
        ax.set_ylabel("Numbers of Samples")
        ax.set_xscale('log')
        plt.title("Number of Samples per Prime Implicant")
        for cat in self.cats:
            plt.plot(self.nsamples[cat], label=cat)
        plt.legend(loc="upper left")
        plt.show()


    def plot(self, leaf_data, imp_data):
        from matplotlib import pyplot
        for i in range(len(self.cats)):
            #L = leaf_data[i]
            I = imp_data[i]
            imp_x = range(len(I))
            imp_y = [ imp[1] for imp in I ]
            fig, ax = pyplot.subplots()
            ax.set_xlim([0, max(imp_x) + 1])
            ax.set_ylim([0, max(imp_y) + 1])
            ax.set_ylabel("Number of Covered Training Samples")
            ax.set_xlabel("Prime Implicant")        
            pyplot.title(self.cats[i].upper())
            pyplot.scatter(imp_x, imp_y, marker='x')
            pyplot.legend(loc='upper right')
            pyplot.show()
