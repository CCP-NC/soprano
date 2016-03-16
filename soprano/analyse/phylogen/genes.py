"""Definitions for the various genes used by PhylogenCluster"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from collections import namedtuple
from soprano.properties.basic import LatticeCart, LatticeABC, CalcEnergy

Gene = namedtuple('Gene', ['name', 'weight', 'params'])

"""
Gene definitions and parser functions.

The default interface of a gene parser is:

def generic_parser(AtomsCollection, **parameters):
    return np.array [shape = (AtomsCollection.length, :)]

"""

def parsegene_latt_cart(c):
    lattCartProp = LatticeCart(shape=(9,))
    return np.array(lattCartProp(c))

def parsegene_latt_abc(c):
    return np.array(LatticeABC.get(c))[:,0]

def parsegene_latt_ang(c):
    return np.array(LatticeABC.get(c))[:,1]

class GeneDictionary(object):
    """Simple class holding the gene definitions in a slightly safer way"""

    _gene_dictionary = {
        'latt_cart': {
            'default_params': {},
            'parser': parsegene_latt_cart,
            'pair': False
        },
        'latt_abc': {
            'default_params': {},
            'parser': parsegene_latt_abc,
            'pair': False
        },
        'latt_ang': {
            'default_params': {},
            'parser': parsegene_latt_ang,
            'pair': False
        }
    }

    _gene_help = {

        'latt_cart': """
        The lattice parameters of a structure in Cartesian form, inline.

        Parameters: None
        Length: 9
        """,

        'latt_abc': """
        The length of the A, B and C lattice parameters.

        Parameters: None
        Length: 3
        """,

        'latt_ang': """
        The angles alpha, beta and gamma defining the lattice.

        Parameters: None
        Length: 3
        """

    }

    @classmethod
    def get_gene(self, g):
        return dict(self._gene_dictionary[g])

    @classmethod
    def help(self, g):
        return self._gene_help[g]