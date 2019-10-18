# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Definitions for the various genes used by PhylogenCluster"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import itertools
import numpy as np
from soprano.selection import AtomSelection
from soprano.utils import list_distance, is_string, compute_asymmetric_distmat
from soprano.properties.basic import LatticeCart, LatticeABC, CalcEnergy
from soprano.properties.linkage import (LinkageList, MoleculeNumber,
                                        MoleculeMass, MoleculeCOMLinkage,
                                        MoleculeRelativeRotation,
                                        HydrogenBonds, HydrogenBondsNumber,
                                        CoordinationHistogram)
from soprano.properties.labeling import (MoleculeSites, HydrogenBondTypes)
from soprano.properties.order import BondOrder


# Useful functions for parsing of complex genes
def _int_array(size=0):
    def parser(s):
        try:
            return np.array([int(x) for x in re.split('[\s,]+', s, size)])
        except:
            raise RuntimeError(('Could not parse line {0}'
                                ' as [int]*{1} array').format(s, size))


def _float_array(size=0):
    def parser(s):
        try:
            return np.array([float(x) for x in re.split('[\s,]+', s, size)])
        except:
            raise RuntimeError(('Could not parse line {0}'
                                ' as [float]*{1} array').format(s, size))


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
    |   parser (function<AtomsCollection, \*\*kwargs>
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

        try:
            return (self.name == other.name) and\
                   (self.weight == other.weight) and\
                   (self.params == other.params)
        except:
            return False

    @property
    def is_pair(self):
        """Whether the gene can only compare a pair of structures or can also
        give an absolute value for each structure individually (required for
        k-means clustering)"""
        return self._pair

    def evaluate(self, c):
        """Evaluate the gene on a given AtomsCollection"""
        val = self._parser(c, **self.params)

        # Check for various possible modes of failure
        if val is None or None in val:
            raise GeneError('Gene {0} has some or all None values'.format(self.name))

        try:
            if np.any(np.isnan(val)):
                raise GeneError('Gene {0} has some or all nan values'.format(self.name))
        except TypeError:
            raise GeneError('Gene {0} has values of a non-numeric type'.format(self.name))

        return val


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
    return np.array(MoleculeNumber.get(c))[:, None]


def parsegene_mol_m(c, Z=0):
    molm = MoleculeMass(size=Z)
    return np.array(molm(c))


def parsegene_mol_com(c, Z=0):
    molc = MoleculeCOMLinkage(size=int(Z*(Z-1)/2))
    return np.array(molc(c))


def parsegene_mol_rot(c, Z=0, twist_axis=None, swing_plane=None):
    molr = MoleculeRelativeRotation(size=int(Z*(Z-1)/2),
                                    twist_axis=twist_axis,
                                    swing_plane=swing_plane)
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


def parsegene_hbonds_site_reference(c, ref=None):

    if ref is None:
        ref = c.structures[0]

    # First, calculate molecules, molecule sites and hydrogen bonds
    MoleculeSites.get(ref)
    HydrogenBonds.get(ref)

    MoleculeSites.get(c)
    HydrogenBonds.get(c)

    # Now for the actual comparison we need to compile a list of Hbonds
    # for each structure, expressed in terms of molecular sites

    reflabels = HydrogenBondTypes.get(ref)
    hblabels = HydrogenBondTypes.get(c)

    # And now to actually create a comparison
    distL = []

    for hb_lab in hblabels:
        distL.append(list_distance(hb_lab, reflabels))

    return np.array(distL)


def parsegene_hbonds_site_compare(c):

    # First, calculate molecules, molecule sites and hydrogen bonds
    MoleculeSites.get(c)
    HydrogenBonds.get(c)

    # Now for the actual comparison we need to compile a list of Hbonds
    # for each structure, expressed in terms of molecular sites

    hblabels = HydrogenBondTypes.get(c)

    # And now to actually create a comparison
    distM = np.zeros((c.length, c.length))

    for hb_i1, hb_lab1 in enumerate(hblabels):
        for hb_i2, hb_lab2 in enumerate(hblabels[hb_i1+1:]):
            d = list_distance(hb_lab1, hb_lab2)
            d /= (len(hb_lab1)+len(hb_lab2))*0.5
            distM[hb_i1, hb_i1+hb_i2+1] = d
            distM[hb_i1+hb_i2+1, hb_i1] = d

    return distM


def parsegene_coord_histogram(c, s1='C', s2='H', max_coord=6):

    chist = CoordinationHistogram(species_1=s1, species_2=s2,
                                  max_coord=max_coord)

    hists = np.zeros((c.length, max_coord+1))
    for i, h in enumerate(chist(c)):
        hists[i] = h[s1][s2]

    return hists.astype(int)


def parsegene_bond_order(c, s1=None, s2=None, channels=10, cutoff_radius=2.0,
                         cutoff_width=0.05, mode='Q'):

    if mode not in ('Q', 'W', 'QW'):
        raise ValueError('Invalid mode argument for parsegene_bond_order')

    l_channels = range(1, channels+1)

    b_ord = []

    def interpret_sel(s, sel):
        if sel is None:
            return AtomSelection.all(s)
        elif is_string(sel):
            return AtomSelection.from_element(s, sel)
        elif hasattr(sel, '__call__'):
            return sel(s)

    for s in c.structures:
        i1 = interpret_sel(s, s1)
        i2 = interpret_sel(s, s2)

        bo = BondOrder(l_channels=l_channels, center_atoms=i1,
                       environment_atoms=i2, cutoff_radius=cutoff_radius,
                       cutoff_width=cutoff_width, compute_W=('W' in mode))
        bo = bo(s)
        bo = np.concatenate([bo[i] for i in mode])

        b_ord.append(bo)

    return np.array(b_ord)


def parsegene_defect_asymmetric_fdist(c, index=0, struct=None):

    if struct is None:
        raise ValueError('defect_asymmetric_fdist gene requires a struct '
                         'argument')

    fp = c.all.get_scaled_positions()[:, index, :]
    return compute_asymmetric_distmat(struct, fp, linearized=False)


def parsegene_defect_asymmetric_fpos(c, index=0, struct=None):

    if struct is None:
        raise ValueError('defect_asymmetric_fdist gene requires a struct '
                         'argument')

    fp = c.all.get_scaled_positions()[:, index, :]
    _, imgs = compute_asymmetric_distmat(struct, fp,
                                         return_images=True)

    return imgs


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
                'Z': int,
                'twist_axis': _float_array(3),
                'swing_plane': _float_array(3),
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

        'hbonds_site_reference': {
            'default_params': {
                'ref': None,
            },
            'parser': parsegene_hbonds_site_reference,
            'pair': False
        },

        'hbonds_site_compare': {
            'default_params': {},
            'parser': parsegene_hbonds_site_compare,
            'pair': True
        },

        'coord_histogram': {
            'default_params': {
                's1': 'C',
                's2': 'H',
                'max_coord': 6
            },
            'parser': parsegene_coord_histogram,
            'pair': False
        },

        'bond_order_pars': {
            'default_params': {
                's1': None,
                's2': None,
                'channels': 10,
                'cutoff_radius': 2.0,
                'cutoff_width': 0.05,
                'mode': 'Q'
            },
            'parser': parsegene_bond_order,
            'pair': False
        },

        'defect_asymmetric_fdist': {
            'default_params': {
                'index': 0,
                'struct': None,
            },
            'parser': parsegene_defect_asymmetric_fdist,
            'pair': True
        },

        'defect_asymmetric_fpos': {
            'default_params': {
                'index': 0,
                'struct': None,
            },
            'parser': parsegene_defect_asymmetric_fpos,
            'pair': False
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
            twist_axis ([float]*3): if present, only compare the Twist
                                    component of quaternion along the given
                                    axis. The Twist/Swing decomposition splits
                                    a quaternion in a rotation around an axis
                                    and one around an orthogonal direction.
                                    Only one between this and swing_plane can
                                    be present.
            swing_plane ([float]*3): if present, only compare the Swing
                                     component of quaternion along the given
                                     axis. The Twist/Swing decomposition
                                     splits a quaternion in a rotation around
                                     an axis and one around an orthogonal
                                     direction. Only one between this and
                                     twist_axis can be present.
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

        'hbonds_site_reference': """
        Comparison between hydrogen bonds in the given systems and a reference
        system, site-wise.
        Works for molecular systems. Bonds are compared to see how many share
        the same structure (namely, are established between equivalent sites
        in equivalent molecules). The number of different bonds is given.

        Parameters:
            ref (ase.Atoms): reference system to compare against (default =
                             first system in the collection)
        Length: 1
        """,

        'hbonds_site_compare': """
        Comparison between hydrogen bonds in two given systems, site-wise.
        Works for molecular systems. Bonds are compared to see how many share
        the same structure (namely, are established between equivalent sites
        in equivalent molecules). The number of different bonds is treated as
        a 'distance' between pairs of systems.

        Parameters: None
        Length: 1
        """,

        'coord_histogram': """
        Coordination histogram for two given species s1 and s2. Gives an array
        whose element of index i contains the number of atoms of species s1
        that are bonded to i atoms of species s2. The parameter max_coord
        controls how many bins are created; any coordination number higher
        than max_coord gets bunched up in the last element. By default this
        means all coordination values of 6 or more.
        As an example, the s1=C, s2=H histogram for a CH3CH2CH=CHCOOH molecule
        would look like: [1,2,1,1,0,0,0].

        Parameters:
            s1 (string): chemical symbol of species whose coordination is to
                         be evaluated. By default C
            s2 (string): chemical symbol of ligand species with s1. By default
                         H
            max_coord (int): maximum coordination number to allow in
                             histogram. By default 6
        Length: max_coord+1 (by default 7)
        """,

        'bond_order_pars': """
        Steinhardt bond order parameters (based on spherical harmonics) for a
        pair of species in the given system. These parameters reflect the
        symmetry of the local environment for each atom by making use of 
        spherical harmonics. Spherical harmonic functions are computed up to 
        the requested angular momentum channel; higher angular momenta
        correspond to more rapidly varying angular functions (and therefore
        will match finer details). There are two types of parameter: 
        Q (second order) and W (third order). Both are invariant to rotation,
        but the W are more expensive to compute and are off by default. A 
        sigmoidal cutoff is applied to weight atoms in the neighbourhood;
        parameters for the sigmoidal function can be passed.

        Parameters:
            s1 (multiple types): selector for the atoms whose environment is
                                 to be evaluated.
                                 If an AtomSelection is used, it will define
                                 the atoms.
                                 If a string is used, it will be
                                 interpreted as a symbol of a chemical
                                 species.
                                 If a callable is used, it must take an 
                                 ase.Atoms object as the only argument, and
                                 return an AtomSelection. 
                                 Default is None, meaning a sum over all
                                 atoms.
            s2 (multiple types): selector for the atoms that contribute to the
                                 environment.
                                 Follows the same rules as s1.
                                 Default is None, meaning a sum over all 
                                 atoms.
            channels (int): number of angular momentum channels. More channels
                            mean finer detail but higher computational cost.
                            Default is 10.
            cutoff_radius (float): cutoff distance for sigmoidal function.
                                   Default is 2 Angstroms.
            cutoff_width (float): scale over which the sigmoidal function
                                  should vanish. Default is 0.05 Angstroms.
            mode (str): mode to compute. Can be Q (only second order), W (only
                        third order) and QW (both orders, concatenated).
                        Default is Q.

        """,

        'defect_asymmetric_fdist': """
        Compute a fractional coordinates distance between single atoms in the
        structures accounting for all effects of symmetry operations. This
        should group together atoms occupying sites that are
        crystallographically equivalent. This is especially useful for defect
        analysis. A pure structure must be passed as an argument to compute
        the symmetry operations in the first place. An installation of spglib
        is required for this computation.

        Parameters:
            index (int): index of the defect for which the distance matrix is
                         to be computed. It must be the same for all
                         structures. Default is 0
            struct (ase.Atoms): pure structure from which the space group and
                                symmetry operations must be computed. Default
                                is None, must be provided for the calculation
                                to work
        """,

        'defect_asymmetric_fpos': """
        Compute an asymmetric fractional coordinates position for a single atom
        in the structures accounting for all effects of symmetry operations.
        This should group together atoms occupying sites that are
        crystallographically equivalent. This is especially useful for defect
        analysis. A pure structure must be passed as an argument to compute
        the symmetry operations in the first place. An installation of spglib
        is required for this computation.

        Parameters:
            index (int): index of the defect for which the distance matrix is
                         to be computed. It must be the same for all
                         structures. Default is 0
            struct (ase.Atoms): pure structure from which the space group and
                                symmetry operations must be computed. Default
                                is None, must be provided for the calculation
                                to work

        """
    }

    @classmethod
    def get_gene(self, g):
        """Get the definition for a given gene"""
        return dict(self._gene_dictionary[g])

    @classmethod
    def help(self, g=None):
        """Get an help string for a given gene"""
        if g is not None:
            return self._gene_help[g]
        else:
            return ('List of available genes:\n' +
                    '\n'.join(self._gene_help.keys()))


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
