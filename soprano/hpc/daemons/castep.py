"""A basic Daemon to run CASTEP calculations
"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from soprano.hpc.daemons import DaemonHPC


class CastepDaemon(DaemonHPC):

    def set_parameters(self, structures):
        """Set a collection to draw Atoms objects from, and an index keeping
        track of where we are in scanning it.

        | Args:
        |   collection (AtomsCollection): AtomsCollection object containing
        |                                 the atoms to perform calculations
        |                                 on. These must already have a CASTEP
        |                                 calculator set.        

        """

        self.atoms_i = 0
        self.atoms = structures

    def next_processes(self, n=1):

        procs = [{'atoms': a}
                 for a in self.atoms[self.atoms_i:self.atoms_i+n]]
        self.atoms_i += n

        return procs

    def on_complete(self, rval):

        if self.atoms_i < len(self.atoms):
            self.start_processes()

    def run_process(self, loop_id, atoms):
        
        print(atoms.get_potential_energy())

        return "Complete"

