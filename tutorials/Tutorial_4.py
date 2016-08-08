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


TUTORIAL 4 - Clustering analysis

"""

print

# Basic imports
import os, sys
sys.path.insert(0, os.path.abspath('..')) # This to add the Soprano path to the PYTHONPATH
                                          # so we can load it without installing it

# Other useful imports

import glob

import numpy as np

import ase
from ase import io as ase_io

from soprano.collection import AtomsCollection


"""
1 - SETTING UP CLUSTERING

When dealing with multiple structures, Soprano allows to use clustering tools to split them in groups. In this case
we'll use the examples from the first tutorial for a very basic analysis. These examples are slightly randomised 
copies of BCC and FCC iron cells. Telling which one is which is going to be the focus of this exercise.
"""
from soprano.analyse.phylogen import Gene, PhylogenCluster

# List all files in the tutorial directory
cifs = glob.glob('tutorial_data/struct*.cif')

aColl = AtomsCollection(cifs, progress=True) # "progress" means we will visualize a loading bar


# To carry out the analysis we need to define a PhylogenCluster object. This will need as input some Gene objects.
# The phylogenetic nomenclature is just an analogy to the way phylogenetic analysis is carried out in biology.
# Ideally, we're trying to give each structure a "DNA" of sorts, then compare them amongst themselves to find
# which ones are more closely related.
# Finding the right properties to use to distinguish between structures is key here. In this examples it's pretty
# simple but we'll still illustrate a couple different ways to get there

# This gene represents the length of the three lattice parameters
gene_abc = Gene(name='latt_abc_len', weight=1.0, params={}) 

# This gene represents the linkage list property as seen in tutorial 2
gene_lnk = Gene(name='linkage_list', weight=1.0, params={})

# We can try these separately or together
phClust1 = PhylogenCluster(aColl, [gene_abc])
phClust2 = PhylogenCluster(aColl, [gene_lnk])
phClust3 = PhylogenCluster(aColl, [gene_abc, gene_lnk]) # In this case they get chained together,
                                                        # and the relative weights are used
    
# Here's a summary of the generated "genes"
genes, genes_info = phClust3.get_genome_vectors_norm()

print "---- Genetic strings for each structure (normalised) ----\n"
print '\n'.join([str(g) for g in genes]), '\n'
print 'Info:\t', genes_info, '\n\n' # This tells us which genes are present and how long the respective fields are


"""
2 - CLUSTERING METHODS

When clustering structures two algorithms are available: hierarchical and k-means.
Hierarchical clustering builds a tree branching progressively from a single "trunk" containing all structures to 
multiple "leaves" representing each one structure. To turn this into a number of cluster a depth has to be
provided by the user.
K-Means clustering builds a fixed number of clusters. In this case no depth is required but the user still needs
to submit an educated guess about the expected number of clusters. Some times the algorithm can produce less 
clusters than that anyway (i.e. some clusters are in fact left empty).
"""

# First, trying k-means. We know to expect 2 clusters in this case (BCC and FCC)
clust1_inds, clust1_slices = phClust1.get_kmeans_clusters(2) # Returns indices and slices representing the clusters
clust2_inds, clust2_slices = phClust2.get_kmeans_clusters(2)
clust3_inds, clust3_slices = phClust3.get_kmeans_clusters(2)

# Now to compare...
# These should be the same except for the possibility of indices being swapped
print "---- k-means clusters obtained with different genomes ----\n"
print "ABC only:\t", clust1_inds
print "Linkage only:\t", clust2_inds
print "Both:\t\t", clust3_inds
print "\n"

# Now hierarchical clustering
print "---- Hierarchical clusters obtained with different genomes ----\n"
# Variable t (depth of traversing in the tree)
# At the beginning should start with the most branched out version, then leaves should coalesce into clusters
for t in np.linspace(0, 1, 11):
    print "t = {0}:\t\t".format(t), phClust3.get_hier_clusters(t)[0]