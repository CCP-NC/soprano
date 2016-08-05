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


TUTORIAL 3 - AtomSelection and transforms

"""

print


# Basic imports
import os, sys
sys.path.insert(0, os.path.abspath('..')) # This to add the Soprano path to the PYTHONPATH
                                          # so we can load it without installing it

# Other useful imports
import numpy as np

from ase import Atoms


"""
1 - SELECTING ATOMS

Besides allowing to manipulate information about multiple structures, Soprano provides tools to edit them as well.
This is accomplished by combining selection of atoms and transformation operations that change their positions.
As an example we will use again the ammonia molecule.
Selections can be carried with multiple criteria. The basic ones are selection by element, selection of all atoms
in a box, and selection of all atoms in a sphere.
"""
from soprano.selection import AtomSelection

nh3coords = np.array([[ 2.5,     2.5,     2.5   ],
                      [ 3.4373,  2.5,     2.1193],
                      [ 2.0314,  3.3117,  2.1193],
                      [ 2.0314,  1.6883,  2.1193]])
nh3l = Atoms('NHHH', nh3coords, cell=[5,5,5]) # The cell is just an empty box

# Now instead of switching the coordinates by hand let's do this with selections.
nh3Hsel = AtomSelection.from_element(nh3l, 'H') # All H atoms in nh3l

# Selections can be manipulated in interesting ways. To begin with, we can create an Atoms object containing 
# only the selected atoms

h3 = nh3Hsel.subset(nh3l)

print "---- Selected atoms contained in nh3Hsel ----\n"
print h3.get_chemical_symbols(), "\n\n"

# Also, selections can be summed, subtracted, or multiplied (representing intersection)
sel1 = AtomSelection(nh3l, [1]) # A custom generated selection
sel2 = AtomSelection(nh3l, [0, 2]) # A custom generated selection

print "---- Indices of selected atoms for various combinations ----\n"
print "sel1:\t", sel1.indices
print "sel2:\t", sel2.indices
print "nh3Hsel:\t", nh3Hsel.indices
print "sel1+sel2:\t", (sel1+sel2).indices
print "nh3Hsel-sel1:\t", (nh3Hsel-sel1).indices
print "nh3Hsel*sel2:\t", (nh3Hsel*sel2).indices
print "\n\n"

"""
2 - APPLYING TRANSFORMS

Transforms in Soprano are special cases of Properties. They are properties that take in an AtomSelection and some
parameters and return Atoms objects with the transformation applied. These can be used to mass-edit entire
AtomsCollection objects. Basic transforms are Translate, Rotate and Mirror
"""
from soprano.properties.transform import Mirror


mirr = Mirror(selection=nh3Hsel,
              plane=[0, 0, 1, -0.5],
              scaled=True) # Mirror with respect to the XY plane passing through Z=0.5
                           # in fractional coordinates (scaled=True).

# NOTE: the plane is defined by the plane equation cohefficients.
# So for ax+by+cz+d = 0 we have [a,b,c,d]                          

nh3r = mirr(nh3l)

print "---- Coordinates of Hydrogen atoms in left and right versions of the molecule ----\n"
print "nh3l:\n", nh3Hsel.subset(nh3l).get_positions(), "\n"
print "nh3r:\n", nh3Hsel.subset(nh3r).get_positions(), "\n"

"""
3 - SELECTION VALIDATION

As a safety against mistakes, by default, any AtomSelection has its "validate" property initialised to True.
This means that whenever the selection is used to create a subset or make a transform a check is performed to verify
that the chemical symbols of the Atoms object it is operating on is the same as the one on which it was originally
created. In other words, selections shouldn't be able to operate on structures they don't refer to.
"""

# Create two structures
a1 = Atoms('HCO')
a2 = Atoms('FeAgAu')

# Create a selection
sel1 = AtomSelection.from_sphere(a1, [0, 0, 0], 0.1)

# Try using it on the wrong structure
try:
    a0 = sel1.subset(a2)
except ValueError as e:
    print "An error has verified: \n>\t", e