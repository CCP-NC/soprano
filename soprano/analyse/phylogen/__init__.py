"""
Module containing functions and classes for phylogenetic clustering of
collections.
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import

from soprano.analyse.phylogen.phylogenclust import PhylogenCluster
from soprano.analyse.phylogen.genes import (Gene, GeneError, GeneDictionary,
                                            load_genefile)
