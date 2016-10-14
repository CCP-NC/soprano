"""Implementation of some CASTEP related AtomsProperties"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
from soprano.properties import AtomsProperty


class CastepEnthalpy(AtomsProperty):

    """
    CastepEnthalpy

    Enthalpy as found in the .castep file of a GeometryOptimization
    calculation. If not present, this will fall back on the final free energy.

    | Parameters:
    |   castep_path (str): the path in which the .castep file is to be found.
    |   seedname_info (str): the Atoms.info key that contains the seedname
    |                        of the .castep file. By default is 'name'.

    """

    default_name = 'castep_enthalpy'
    default_params = {
        'castep_path': '.',
        'seedname_info': 'name'
    }

    @staticmethod
    def extract(s, castep_path, seedname_info):
        # Open the file
        fname = os.path.join(castep_path, s.info[seedname_info] + '.castep')
        f = open(fname).read()
        # Parse the expression of interest
        ent_re = re.compile('BFGS: Final Enthalpy\s+=\s+([\+\-\.0-9E]+)\s+eV')
        match = ent_re.findall(f)
        if len(match) > 0:
            try:
                return float(match[-1])
            except:
                raise RuntimeError('Can\'t parse enthalpy:'
                                   'corrupted CASTEP file')
        else:
            # Settle for the final free energy
            nrg_re = re.compile('Final free energy \(E\-TS\)\s+=\s+'
                                '([\+\-\.0-9E]+)\s+eV')
            match = nrg_re.findall(f)
            if len(match) == 0:
                raise RuntimeError('Corrupted or incomplete CASTEP file')
            return float(match[-1])
