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


TUTORIAL 1 - Basic concepts: using AtomsCollection objects

"""


# Basic imports
import os, sys
# File location
filepath = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, os.path.join(filepath, '..')) # This to add the Soprano path to the PYTHONPATH
                                          		 # so we can load it without installing it                                          		 
datapath = os.path.join(filepath, 'tutorial_data')

# Other useful imports

import glob

import numpy as np

import ase
from ase import io as ase_io

from soprano.collection import AtomsCollection

"""
1 - LOADING STRUCTURES

Soprano can handle multiple structure loading into a single AtomsCollection object.
The structures are loaded singularly as ASE (Atomic Simulation Environment) Atoms objects.
"""

# List all files in the tutorial directory
cifs = glob.glob(os.path.join(datapath, 'struct*.cif'))

aColl = AtomsCollection(cifs, progress=True) # "progress" means we will visualize a loading bar
print

"""
2 - HANDLING COLLECTIONS

Collections are a convenient way of manipulating multiple structures. They allow for many operations that act
collectively on all Atoms objects, or return values from them all at once.
"""

# To access an individual structure, one can simply use indexing:
a0 = aColl.structures[0]
print '---- struct_0.cif positions ----\n'
print a0.get_positions(), '\n\n'

# All properties and methods of Atoms objects are available on an entire collection too, by using
# the meta-element 'all'

print '---- all struct_*.cif positions----\n'
print aColl.all.get_positions(), '\n\n'

print '---- all struct_*.cif info dictionaries----\n'
print aColl.all.info, '\n\n'

# Collections can also be sliced like Numpy arrays for convenience
aColl02 = aColl[0:2]
aColl25 = aColl[2:5]

# Then join them together
aColl05 = aColl02+aColl25

print "---- Collection slice lengths ---- \n"
print "aColl02 = {0}\taColl25 = {1}\taColl05 = {2}\n\n".format(aColl02.length, aColl25.length, aColl05.length)

# Collections can also store "arrays" of data, similarly to Atoms objects in ase
# These arrays' elements are tied each to one structure, and can be used to sort them

arr = range(10, 0, -1) # Let's use this array to reverse the order of a collection

aColl.set_array('reversed_range', arr)

aCollSorted = aColl.sorted_byarray('reversed_range')

print "---- Getting an array from a collection ---- \n"
print "Unsorted: ", aColl.get_array('reversed_range'), "\n"
print "Sorted: ", aCollSorted.get_array('reversed_range'), "\n\n"

# And to make sure
print "---- First vs. last elements ---- \n"
print aColl.structures[0].get_positions(), "\n"
print aCollSorted.structures[-1].get_positions()