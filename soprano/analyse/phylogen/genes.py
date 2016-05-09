"""Definitions for the various genes used by PhylogenCluster"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import numpy as np
from collections import namedtuple
from soprano.utils import parse_intlist, parse_floatlist
from soprano.properties.basic import LatticeCart, LatticeABC, CalcEnergy
from soprano.properties.linkage import (LinkageList, MoleculeNumber,
                                        MoleculeMass, MoleculeCOMLinkage,
                                        MoleculeRelativeRotation)


Gene = namedtuple('Gene', ['name', 'weight', 'params'])


class GeneError(Exception):
    pass


"""
Gene definitions and parser functions.

The default interface of a gene parser is:

def generic_parser(AtomsCollection, **parameters):
    return np.array [shape = (AtomsCollection.length, :)]

"""

def parsegene_energy(c):
    return np.array(CalcEnergy.get(c))


def parsegene_latt_cart(c):
    lattCartProp = LatticeCart(shape=(9,))
    return np.array(lattCartProp(c))


def parsegene_latt_abc(c):
    return np.array(LatticeABC.get(c))[:, 0]


def parsegene_latt_ang(c):
    return np.array(LatticeABC.get(c))[:, 1]


def parsegene_linkage_list(c, size=10):
    linkl = LinkageList(size=size)
    return np.array(linkl(c))

def parsegene_mol_num(c):
    return np.array([MoleculeNumber.get(c)])

def parsegene_mol_m(c, Z=0):
    molm = MoleculeMass(size=Z)
    return np.array(molm(c))

def parsegene_mol_com(c, Z=0):
    molc = MoleculeCOMLinkage(size=int(Z*(Z-1)/2))
    return np.array(molc(c))

def parsegene_mol_rot(c, Z=0):
    molr = MoleculeRelativeRotation(size=int(Z*(Z-1)/2))
    return np.array(molr(c))

class GeneDictionary(object):

    """Container class holding gene definitions"""

    _gene_dictionary = {
        'energy': {
            'default_params': {},
            'parser': parsegene_energy,
            'pair': False
        },
        'latt_cart': {
            'default_params': {},
            'parser': parsegene_latt_cart,
            'pair': False
        },
        'latt_abc_len': {
            'default_params': {},
            'parser': parsegene_latt_abc,
            'pair': False
        },
        'latt_abc_ang': {
            'default_params': {},
            'parser': parsegene_latt_ang,
            'pair': False
        },

        'linkage_list': {
            'default_params': {
                'size': int
            },
            'parser': parsegene_linkage_list,
            'pair': False
        },

        'molecule_number': {
            'default_params': {},
            'parser': parsegene_mol_num,
            'pair': False,
        },

        'molecule_mass': {
            'default_params': {
                'Z': int
            },
            'parser': parsegene_mol_m,
            'pair': False,
        },

        'molecule_com_linkage': {
            'default_params': {
                'Z': int
            },
            'parser': parsegene_mol_com,
            'pair': False,
        },

        'molecule_rot_linkage': {
            'default_params': {
                'Z': int
            },
            'parser': parsegene_mol_rot,
            'pair': False,
        },        

    }

    _gene_help = {

        'energy': """
        The calculated total energy of a structure. This is the value returned
        by ase.Atoms.get_potential_energy().

        Parameters: None
        Length: 1
        """,

        'latt_cart': """
        The lattice parameters of a structure in Cartesian form, inline.

        Parameters: None
        Length: 9
        """,

        'latt_abc_len': """
        The length of the A, B and C lattice parameters.

        Parameters: None
        Length: 3
        """,

        'latt_abc_ang': """
        The angles alpha, beta and gamma defining the lattice.

        Parameters: None
        Length: 3
        """,

        'linkage_list': """
        A list of the n shortest interatomic distances in the structure
        (periodic boundaries are taken into account).

        Parameters: 
            size (int): how many distances are used (default = 10)
        Length: [size]
        """,

        'molecule_number': """
        Number of molecules found in the structure.

        Parameters: None
        Length: 1
        """,

        'molecule_mass': """
        Masses of each of the molecules found in the structure.

        Parameters:
            Z (int): expected number of molecules (default = total number of 
                     molecules)
        Length: Z
        """,

        'molecule_com_linkage': """
        A list of the n shortest intermolecular distances in the structure
        (periodic boundaries are taken into account and the molecules are
         considered coincident with their Center Of Mass).

        Parameters:
            Z (int): expected number of molecules (default = total number of 
                     molecules)
        Length: Z*(Z-1)/2
        """,

        'molecule_rot_linkage': """
        A list of the n shortest intermolecular rotational distances in the
        structure (orientation is considered to be the one of the molecules'
        principal inertia axis system, standard 1-|q1.q2| quaternion distance is
        used).

        Parameters:
            Z (int): expected number of molecules (default = total number of 
                     molecules)
        Length: Z*(Z-1)/2
        """,
    }

    @classmethod
    def get_gene(self, g):
        """Get the definition for a given gene"""
        return dict(self._gene_dictionary[g])

    @classmethod
    def help(self, g):
        """Get an help string for a given gene"""
        return self._gene_help[g]


def load_genefile(gfile):
    """Load a gene file and return the (validated) list of genes contained
    within.

    | Args:
    |   gfile (file or str): file to parse

    | Returns:
    |   genelist: a list of genes parsed from the given file, ready to be
    |             passed to a PhylogenCluster

    """

    if isinstance(gfile, file):
        genefile = gfile.read()
    else:
        genefile = open(gfile).read()

    comment_re = re.compile('#[^\n]*\n')
    gene_re = re.compile('\s*([a-zA-Z_]+)\s+([0-9\.]+)(?:\s*{([^}]+)})*')
    block_re = re.compile('\s*([a-zA-Z_]+)\s+([0-9\.]+)')

    # First, remove comments
    genefile = comment_re.sub('\n', genefile)

    # Then actually get the blocks
    gblocks = gene_re.findall(genefile)

    genes = []

    for g in gblocks:
        gname = g[0]
        try:
            gentry = GeneDictionary.get_gene(gname)
        except:
            raise GeneError('{0} is not a valid gene'.format(gname))
        try:
            gweight = float(g[1])
        except:
            raise GeneError('Badly formatted weight definition for gene'
                            ' {0}'.format(gname))
        # Does it have an argument block?
        gargs = {}
        if g[2] != '':
            for arg in block_re.findall(g[2]):
                try:
                    arg_parser = gentry['default_params'][arg[0]]
                except:
                    raise GeneError(('{0} is not a valid parameter for'
                                     ' {1}').format(arg[0], gname))
                try:
                    gargs[arg[0]] = arg_parser(arg[1])
                except:
                    raise GeneError('Invalid value for parameter '
                                    '{0}'.format(arg[0]))

        genes.append(Gene(gname, gweight, gargs))

    return genes
