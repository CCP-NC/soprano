"""Implementation of AtomsProperties that relate to linkage of atoms"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.utils import minimum_periodic
from soprano.properties import AtomsProperty


class LinkageList(AtomsProperty):

    """
    LinkageList

    Produces an array containing the atomic pair distances in a system,
    reduced to their shortest periodic version and sorted min to max.

    | Parameters: 
    |   num (int): maximum number of distances to include. If not present, all
    |              of them will be included. If present, arrays will be cut or
    |              padded to reach this number

    """

    default_name = 'linkage_list'
    default_params = {
            'num': 0
    }

    @staticmethod
    def extract(s, num):
        # Get the interatomic pair distances
        v = s.get_positions()
        v = v[:, None, :]-v[None, :, :]
        v = v[np.triu_indices(v.shape[0], k=1)]
        # Reduce them
        v = minimum_periodic(v, s.get_cell())
        # And now compile the list
        link_list = np.linalg.norm(v, axis=-1)
        link_list.sort()
        if num > 0:
            if link_list.shape[0] >= num:
                link_list = link_list[:num]
            else:
                link_list = np.pad(link_list,
                                   (0, num-link_list.shape[0]),
                                   mode=str('constant'),
                                   constant_values=np.inf)

        return link_list
