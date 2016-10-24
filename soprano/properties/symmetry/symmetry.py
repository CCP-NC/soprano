"""Implementation of AtomProperties that relate to symmetry"""

try:
    import spglib
except ImportError:
    from pyspglib import spglib

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from soprano.properties import AtomsProperty

class SymmetryDataset(AtomsProperty):

    """
    SymmetryDataset

    Extracts SPGLIB's standard symmetry dataset from a given system, including
    spacegroup symbol, symmetry operations etc.

    | Parameters:
    |   symprec (float): distance tolerance, in Angstroms, applied when
    |                    searching symmetry.

    | Returns:
    |   symm_dataset (dict): dictionary of symmetry information

    """

    default_name = 'symmetry_dataset'
    default_params = {
        'symprec': 1e-5,
    }

    @staticmethod
    def extract(s, symprec):
        symdata = spglib.get_symmetry_dataset(s, symprec=symprec)
        return symdata


