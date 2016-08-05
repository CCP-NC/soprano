#!/usr/bin/env python

"""
SOPRANO: a Python library for generation, manipulation and analysis of large batches of crystalline structures
by Simone Sturniolo
      _
    /|_|\ 
   / / \ \
  /_/   \_\
  \ \   / /
   \ \_/ /
    \|_|/
    
Developed within the CCP-NC project. Copyright STFC 2016


TUTORIAL 2 - Generators, Properties and Calculators

"""

# Basic imports
import os, sys
sys.path.insert(0, os.path.abspath('..')) # This to add the Soprano path to the PYTHONPATH
                                          # so we can load it without installing it

# Other useful imports

import numpy as np

from ase import Atoms
from ase import io as ase_io


"""
1 - USING GENERATORS

Soprano provides a series of generators able to create multiple structures on one go based on simple criteria.
One of these, used here, is the linspaceGen, which interpolates linearly between two extreme structures. Others 
are the rattleGen (generating copies of a given structure with random atomic displacements) and the airssGen
(binding to AIRSS' buildcell executable to generate random structures, only available if AIRSS is installed).
"""
from soprano.collection import AtomsCollection
from soprano.collection.generate import linspaceGen


# Let's use the ammonia molecule switching configurations as an example
nh3coords = np.array([[ 2.5,     2.5,     2.5   ],
                      [ 3.4373,  2.5,     2.1193],
                      [ 2.0314,  3.3117,  2.1193],
                      [ 2.0314,  1.6883,  2.1193]])
nh3l = Atoms('NHHH', nh3coords, cell=[5,5,5]) # The cell is just an empty box
# Now the right version
nh3coords *= [1, 1, -1]
nh3r = Atoms('NHHH', nh3coords, cell=[5,5,5])

# Now let's build a collection of 20 intermediate steps between the two structures
nh3linsp = linspaceGen(nh3l, nh3r, steps=20, periodic=True)
# Generators can be passed directly to the AtomsCollection constructor
nh3coll = AtomsCollection(nh3linsp)


"""
2 - PROPERTIES

Soprano Properties are classes meant to extract complex arrays of information from collections.
A number of these are provided by default, but advanced users can easily create their own class
inheriting from the generic AtomsProperty class to implement particular needs.
"""
from soprano.properties.linkage import LinkageList


# As a first experiment we try using LinkageList, a property meant to return a list of all pair interatomic distances
# in a system. This can serve as a fingerprint to distinguish different structures

# The basic usage is to just call the Property's method "get". In this way the Property is calculated with
# default parameters.
# The three shortest values (varying) are N-H distances, while the constant ones are H-H distances

print "---- Linkage List for all NH3 configurations - Default parameters\n"
print '\n'.join(['{0}'.format(x) for x in LinkageList.get(nh3coll)]), "\n\n"


# If one wants to use parameters, an instance of the Property has to be created.
# For example LinkageList accepts a parameter "size" that limits the number of distances computed.
# This can then just be called on the AtomsCollection

customLL = LinkageList(size=3)

print "---- Linkage List for all NH3 configurations - Custom parameters\n"
print '\n'.join(['{0}'.format(x) for x in customLL(nh3coll)]), "\n\n"


# Now we can try creating a custom property. This one will calculate the center of mass of all Hydrogen atoms.
from soprano.properties import AtomsProperty


class HydrogenCOM(AtomsProperty):
    
    default_name = 'hydrogen_com' # These need to be defined for any property
    default_params = {}
    
    @staticmethod
    def extract(s): # This is where the core of the calculation happens
        # s is a single Atoms object passed to this method
        
        chemsyms = s.get_chemical_symbols()
        h_inds = [i for i, sym in enumerate(chemsyms) if sym == 'H']
        h_pos = s.get_positions()[h_inds]
        com = np.average(h_pos, axis=0)
        
        return com

print "---- Hydrogen COM for all NH3 configurations\n"
print '\n'.join(['{0}'.format(x) for x in HydrogenCOM.get(nh3coll)]), "\n\n"


"""
3 - CALCULATORS

The Atomic Simulation Environment provides bindings to many codes in the form of calculators.
These include ab initio codes like CASTEP and VASP as well as empirical force fields. These calculators can be set
and used in Soprano as well. Here we're going to use the most basic one, the Lennard-Jones force field,
as an example.
"""
from ase.calculators.lj import LennardJones
from soprano.properties.basic import CalcEnergy

nh3coll.set_calculators(LennardJones) # Creates calculators of the given type for all structures

print "---- NH3 Lennard-Jones energy for all configurations ----\n"
print '\n'.join(['{0}'.format(x) for x in CalcEnergy.get(nh3coll)]), "\n\n"