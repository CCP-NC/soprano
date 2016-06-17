"""Implementation of some basic AtomsProperty classes"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.utils import cart2abc
from soprano.properties import AtomsProperty


class LatticeCart(AtomsProperty):

    """
    LatticeCart

    Property representing the Cartesian form of a structure's lattice

    | Parameters:
    |   shape (tuple): the shape to give to the array

    """

    default_name = 'lattice_cart'
    default_params = {
            'shape': (3, 3)
    }

    @staticmethod
    def extract(s, shape):
        return s.get_cell().reshape(shape)


class LatticeABC(AtomsProperty):

    """
    LatticeABC

    Property representing the axis-angles form of a structure's lattice

    | Parameters:
    |   shape (tuple): the shape to give to the array
    |   deg (bool): whether to give the angles in degrees instead of radians

    """

    default_name = 'lattice_abc'
    default_params = {
            'shape': (2, 3),
            'deg': False
    }

    @staticmethod
    def extract(s, shape, deg):
        abc = cart2abc(s.get_cell())
        if deg:
            abc[1, :] *= 180.0/np.pi
        return abc.reshape(shape)


class CalcEnergy(AtomsProperty):

    """
    CalcEnergy

    Property representing the energy calculated by an ASE calulator

    """

    default_name = 'calc_energy'
    default_params = {}

    @staticmethod
    def extract(s):
        try:
            return s.get_potential_energy()
        except:
            return None


class NumAtoms(AtomsProperty):

    """
    NumAtoms

    Property representing the number of atoms in a structure

    """

    default_name = 'num_atoms'
    default_params = {}

    @staticmethod
    def extract(s):
        return s.positions.shape[0]