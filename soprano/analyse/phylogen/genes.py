"""Definitions for the various genes used by PhylogenCluster"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import itertools
import numpy as np
from soprano.utils import parse_intlist, parse_floatlist
from soprano.properties.basic import LatticeCart, LatticeABC, CalcEnergy
from soprano.properties.linkage import (LinkageList, MoleculeNumber,
                                        MoleculeMass, MoleculeCOMLinkage,
                                        MoleculeRelativeRotation,
                                        MoleculeSites,
                                        HydrogenBonds, HydrogenBondsNumber)


class Gene(object):

    """Gene

    A description of a property, a 'gene', characterizing a structure, to be
    used with a PhylogenCluster. A number of default genes is provided, but
    custom ones can be created as well by passing a parser. Only default genes
    can be used in a .genefile with the phylogen.py script though.

    | Args:
    |   name (str): name of the gene. Must be one of the existing ones or a 
    |               custom one (in which case a parser must be provided as
    |               well). Custom names can't conflict with existing ones
    |   weight (float): weight of the gene to be applied, default is 1.0
    |   params (dict): additional parameters to be passed to the gene parser
    |                  function; when not specified, defaults will be used
    |   parser (function<AtomsCollection, **kwargs>
    |           => np.array): parser function to be used when defining custom
    |                         genes. Must return a two-dimensional Numpy array
    |                         (axis 0 for the elements of the collection,
    |                          axis 1 for the values of the gene)
    |   is_pair (bool): False if the gene returns a multi dimensional point
    |                   for each structure, True if it only returns pair 
    |                   distances. Default is False

    """

    def __init__(self, name, weight=1.0, params={}, parser=None, pair=False):

        # Is the gene default or custom?
        try:
            gdef = GeneDictionary.get_gene(name)
            if parser is not None:
                raise ValueError('A default gene of name {0} already exists'
                                 .format(name))
            self._parser = gdef['parser']
            self._pair = gdef['pair']
            # Check the validity of parameters as well
            if any([p not in gdef['default_params'] for p in params]):
                raise ValueError('Invalid parameters passed for gene {0}'
                                 .format(name))
        except KeyError:
            # Custom!
            if parser is None:
                raise RuntimeError('A parser function is required to create'
                                   ' custom gene of name {0}'.format(name))
            self._parser = parser
            self._pair = pair

        self.name = name
        self.weight = weight
        self.params = params

    def __eq__(self, other):

        iseq = False

        try:
            return (self.name == other.name) and\
                   (self.weight == other.weight) and\
                   (self.params == other.params)
        except:
            return False

    @property
    def is_pair(self):
        return self._pair

    def evaluate(self, c):
        return self._parser(c, **self.params)


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


def parsegene_linkage_list(c, size=0):
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


def parsegene_hbonds_totn(c):
    hblen = HydrogenBondsNumber.get(c)
    return np.array([sum(ld.values()) for ld in hblen])


def parsegene_hbonds_fprint(c):
    hblen = HydrogenBondsNumber.get(c)
    return np.array([zip(*sorted(zip(ld.keys(), ld.values())))[1]
                     for ld in hblen])


def parsegene_hbonds_length(c):
    hblist = HydrogenBonds.get(c)
    hbnlist = HydrogenBondsNumber.get(c)
    # Grab the maximum number for each of these
    hbnmax = np.amax(np.array(hbnlist), axis=0)

    # Now actually gather the lengths
    def cap_to(a, n):
        return a + [np.inf]*(n-len(a))
    hblens = []
    for hbn in hblist:
        # Extract lengths, order by key
        hblen = [sorted(cap_to([hb['length'] for hb in hbn[hbs]],
                                hbnmax[hbs]))
                 for hbs in hbn]
        hblen = list(itertools.chain(*hblen))
        hblens.append(hblen)

    return np.array(hblens)

def parsegene_hbonds_angle(c):
    hblist = HydrogenBonds.get(c)
    hbnlist = HydrogenBondsNumber.get(c)
    # Grab the maximum number for each of these
    hbnmax = np.amax(np.array(hbnlist), axis=0)

    # Now actually gather the lengths
    def cap_to(a, n):
        return a + [np.inf]*(n-len(a))
    hblens = []
    for hbn in hblist:
        # Extract lengths, order by key
        hblen = [sorted(cap_to([hb['angle'] for hb in hbn[hbs]],
                                hbnmax[hbs]))
                 for hbs in hbn]
        hblen = list(itertools.chain(*hblen))
        hblens.append(hblen)

    return np.array(hblens)

def parsegene_hbonds_site_compare(c):

    # First, calculate molecules, molecule sites and hydrogen bonds
    MoleculeSites.get(c)
    HydrogenBonds.get(c)

    # Now for the actual comparison we need to compile a list of Hbonds
    # for each structure, expressed in terms of molecular sites

    hblabels = []

    for s in c.structures:
        hbonds = s.info['hydrogen_bonds']
        mols = s.info['molecules']
        hblabels.append([])
        for hbtype in hbonds:
            for hb in hbonds[hbtype]:
                A_i = hb['A'][0]
                H_i = hb['H']
                B_i = hb['B'][0]

                # Check in which molecule they are
                AH_sites = None
                B_sites = None
                for m_i, m in enumerate(mols):
                    if A_i in m:
                        AH_sites = s.info['molecule_sites'][m_i]
                    if B_i in m:
                        B_sites = s.info['molecule_sites'][m_i]

                if AH_sites is None or B_sites is None:
                    raise RuntimeError('Invalid hydrogen bond detected')

                # Now build the proper definition
                hblabel = ('{0}<{1},{2}>'
                           '..{3}<{4}>').format(AH_sites['name'],
                                                  AH_sites['sites'][A_i],
                                                  AH_sites['sites'][H_i],
                                                  B_sites['name'],
                                                  B_sites['sites'][B_i])
                hblabels[-1].append(hblabel)

    # And now to actually create a comparison
    distM = np.zeros((c.length, c.length))

    for hb_i1, hb_lab1 in enumerate(hblabels):
        for hb_i2, hb_lab2 in enumerate(hblabels[hb_i1+1:]):
            hbdiff = list(hb_lab2)
            d = 0.0
            for hb in hb_lab1:
                if hb in hbdiff:
                    hbdiff.remove(hb)
                else:
                    d += 1
            d += len(hbdiff)
            d /= (len(hb_lab1)+len(hb_lab2))*0.5
            distM[hb_i1, hb_i1+hb_i2+1] = d
            distM[hb_i1+hb_i2+1, hb_i1] = d

    return distM

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

        'hbonds_tot_n': {
            'default_params': {},
            'parser': parsegene_hbonds_totn,
            'pair': False
        },

        'hbonds_fingerprint': {
            'default_params': {},
            'parser': parsegene_hbonds_fprint,
            'pair': False
        },

        'hbonds_length': {
            'default_params': {},
            'parser': parsegene_hbonds_length,
            'pair': False
        }, 

        'hbonds_angle': {
            'default_params': {},
            'parser': parsegene_hbonds_angle,
            'pair': False
        },

        'hbonds_site_compare': {
            'default_params': {},
            'parser': parsegene_hbonds_site_compare,
            'pair': True
        }

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
        principal inertia axis system, standard 1-|q1.q2| quaternion distance
        is used).

        Parameters:
            Z (int): expected number of molecules (default = total number of 
                     molecules)
        Length: Z*(Z-1)/2
        """,

        'hbonds_tot_n': """
        Total number of hydrogen bonds in the system. If the hydrogen bonds 
        have already been calculated, the stored data will be used.

        Parameters: None
        Length: 1
        """,

        'hbonds_fingerprint': """
        Hydrogen bonds fingerprint, i.e. number of hydrogen bonds split by
        type. By convention the ordering is alphabetical, with the types
        described as AH..B (A, B replaced by the respective chemical symbols).
        If the hydrogen bonds have already been calculated, the stored data
        will be used.

        Parameters: None
        Length: 1
        """,

        'hbonds_length': """
        Hydrogen bonds lengths, i.e. length of all hydrogen bonds in the
        system, sorted by type and value.

        Parameters: None
        Length: sum of maximum number of hydrogen bonds per type
        """,

        'hbonds_angle': """
        Hydrogen bonds angles, i.e. amplitude of all hydrogen bonds angles in
        the system, sorted by type and value.

        Parameters: None
        Length: sum of maximum number of hydrogen bonds per type
        """,

        'hbonds_site_compare': """
        Comparison between hydrogen bonds in two given systems, site-wise.
        Works for molecular systems. Bonds are compared to see how many share
        the same structure (namely, are established between equivalent sites
        in equivalent molecules). The number of different bonds is treated as
        a 'distance' between pairs of systems.

        Parameters: None
        Length: 1
        """
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

    if hasattr(gfile, '__read__'):
        genefile = gfile.read()
    else:
        gfile = open(gfile)
        genefile = gfile.read()
        gfile.close()

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
