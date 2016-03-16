"""Phylogenetic clustering class definitions"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import numpy as np
from soprano.collection import AtomsCollection
from soprano.analyse.phylogen.genes import Gene, GeneDictionary


class PhylogenCluster(object):

    """PhylogenCluster

    An object that, given an AtomsCollection and a series of "genes" and
    weights, will build clusters out of the structures in the collection based
    on their reciprocal positions as points in a multi-dimensional space
    defined by those "genes".
    """

    def __init__(self, coll, genes=None,
                 norm_range=(0.0, 1.0),
                 norm_dist=1.0):
        """Initialize the PhylogenCluster object.

        | Args:
        |   coll (AtomsCollection): an AtomsCollection containing the
        |                           structures that should be classified.
        |                           This will be copied and frozen for the
        |                           entirety of the life of this instance;
        |                           in order to operate on a modified
        |                           collection, a new PhylogenCluster should
        |                           be created.
        |   genes (list[tuple]): list of the genes that should be loaded
        |                        immediately; each gene comes in the form of a
        |                        tuple (name (str), weight (float),
        |                        params (dict)). These can also be loaded
        |                        directly from an existing file with the
        |                        static method loadgenes().
        |   norm_range (list[float?]): ranges to constrain the values of
        |                              single genes in between. Default is
        |                              (0, 1). A value of "None" in either
        |                              place can be used to indicate no
        |                              normalization on one or both sides.
        |   norm_dist (float?): value to normalize distance genes to. These
        |                       are the genes that only make sense on pairs of
        |                       structures. Their minimum value is always 0.
        |                       This number would become their maximum value,
        |                       or can be set to None to avoid normalization.

        """

        # Start by copying the collection
        if not isinstance(coll, AtomsCollection):
            raise TypeError('coll must be an AtomsCollection')
        self._collection = copy.deepcopy(coll)

        # Now initialize all the main fields
        self._genes = []
        self._gene_storage = {}  # Contains all past calculated genes
        # This one stores the norm range used last time
        self._normrng = norm_range
        self._normdist = norm_dist
        # These need to be recalculated every time the genes change
        # They also depend on stuff like normalization conditions
        self._needs_recalc = False

        self._gene_legend = None  # Legend of genes as assigned
        self._gene_vectors_raw = None   # For genes that work as points
        self._gene_matrices_raw = None  # For genes that work as distances
        self._gene_vectors_norm = None  # Normalised version of the above
        self._gene_matrices_norm = None  # Normalised version of the above
        self._distmat = None  # Total distance matrix (to feed to Scipy)

        # Now to actually add the genes
        if genes is not None:
            self.set_genes(genes=genes)

    def set_genes(self, genes):
        """Calculate, store and set a list of genes as used for clustering.

        | Args:
        |   genes (list[soprano.analyse.phylogen.Gene]): a list of Genes to 
        |                                                calculate and store.

        """

        self._genes = []

        for g in genes:
            self._genes.append(g)
            # First, check if it's already stored
            if g.name in self._gene_storage:
                # Is it the same?
                if self._gene_storage[g.name]['def'] == g:
                    continue
            # If it's not, it needs to be calculated
            gene_entry = GeneDictionary.get_gene(g.name)
            gene_params = gene_entry['default_params']
            gene_params.update(g.params if g.params is not None else {})
            self._gene_storage[g.name] = {
                'def': g,
                'val': gene_entry['parser'](self._collection,
                                            **gene_params)
            }

        self._needs_recalc = True

    def _recalc(self):
        """Recalculate all stored variables that depend on genes and ranges"""

        clen = self._collection.length
        self._gene_legend = [[], []]
        self._gene_vectors_raw = np.empty((clen, 0))
        self._gene_matrices_raw = np.empty((clen, clen, 0))

        g_vecs_weights = []
        g_mats_weights = []

        for g in self._genes:
            gene_entry = GeneDictionary.get_gene(g.name)
            gene_val = self._gene_storage[g.name]['val']
            # Now normalization and weights
            if not gene_entry['pair']:
                # Append
                self._gene_vectors_raw = np.append(self._gene_vectors_raw,
                                                   gene_val,
                                                   axis=1)
                gn = gene_val.shape[1]
                g_vecs_weights += [g.weight/np.sqrt(gn)]*gn
                self._gene_legend[0].append((g.name, gn))
            else:
                # Append
                self._gene_matrices_raw = np.append(self._gene_matrices_raw,
                                                    gene_val,
                                                    axis=2)
                gn = gene_val.shape[2]
                g_mats_weights += [g.weight/np.sqrt(gn)]*gn
                self._gene_legend[1].append((g.name, gn))

        # Now calculate the normalized versions
        vnorm = self._gene_vectors_raw.copy()
        if self._normrng != (None, None):
            if self._normrng[0] is None:
                # Only maximum
                vnorm += self._normrng[1] - np.amax(vnorm, axis=0)
            elif self._normrng[1] is None:
                # Only minimum
                vnorm += self._normrng[0] - np.amin(vnorm, axis=0)
            else:
                vmin = np.amin(vnorm, axis=0)
                vmax = np.amax(vnorm, axis=0)
                vnorm = (vnorm-vmin)/(vmax-vmin)
                vnorm *= self._normrng[1]-self._normrng[0]
                vnorm += self._normrng[0]
        self._gene_vectors_norm = vnorm*g_vecs_weights

        mnorm = self._gene_matrices_raw.copy()
        if self._normdist is not None:
            mnorm *= self._normdist/np.amax(mnorm, axis=(0, 1))
        self._gene_matrices_norm = mnorm*g_mats_weights

        # Distmat: start with vectors, add distances
        self._distmat = np.linalg.norm(self._gene_vectors_norm[:, None, :] -
                                       self._gene_vectors_norm[None, :, :],
                                       axis=-1)

        self._distmat = np.linalg.norm(np.append(self._distmat[:, :, None],
                                                 self._gene_matrices_norm,
                                                 axis=-1),
                                       axis=-1)

        self._needs_recalc = False

    def get_genome_vectors(self):
        """ Return the genome vectors in raw form (not normalized).
        The vectors refer to genes that allow to define a specific point for
        each structure.

        | Returns:
        |   genome_vectors (np.ndarray): a (collection.length, gene.length)
        |                                array, containing the whole extent
        |                                of the gene values for each structure
        |                                in the collection on each row
        |   genome_legend (list[tuple]): a list of tuples containing (name,
        |                                length) of the gene fragments in the
        |                                array
        """

        if self._needs_recalc:
            self._recalc()

        return self._gene_vectors_raw.copy(), self._gene_legend[0][:]

    def get_genome_matrices(self):
        """ Return the genome matrices in raw form (not normalized).
        The matrices refer to genes that only allow to define a distance
        between structures. The element at i,j represents the distance
        between said structures. The matrix is symmetric and has
        null diagonal.

        | Returns:
        |   genome_matrix (np.ndarray): a (collection.length,
        |                                collection.length, gene.length)
        |                                array, containing the distances for
        |                                each gene and pair of structures in
        |                                row and column
        |   genome_legend (list[tuple]): a list of tuples containing (name,
        |                                length) of the gene fragments in the
        |                                array
        """

        if self._needs_recalc:
            self._recalc()

        return self._gene_matrices_raw.copy(), self._gene_legend[1][:]

    def get_genome_vectors_norm(self):
        """ Return the genome vectors in normalized and weighted form.
        The vectors refer to genes that allow to define a specific point for
        each structure.

        | Returns:
        |   genome_vectors (np.ndarray): a (collection.length, gene.length)
        |                                array, containing the whole extent
        |                                of the gene values for each structure
        |                                in the collection on each row
        |   genome_legend (list[tuple]): a list of tuples containing (name,
        |                                length) of the gene fragments in the
        |                                array
        """

        if self._needs_recalc:
            self._recalc()

        return self._gene_vectors_norm.copy(), self._gene_legend[0][:]

    def get_genome_matrices_norm(self):
        """ Return the genome matrices in normalized and weighted form.
        The matrices refer to genes that only allow to define a distance
        between structures. The element at i,j represents the distance
        between said structures. The matrix is symmetric and has
        null diagonal.

        | Returns:
        |   genome_matrix (np.ndarray): a (collection.length,
        |                                collection.length, gene.length)
        |                                array, containing the distances for
        |                                each gene and pair of structures in
        |                                row and column
        |   genome_legend (list[tuple]): a list of tuples containing (name,
        |                                length) of the gene fragments in the
        |                                array
        """

        if self._needs_recalc:
            self._recalc()

        return self._gene_matrices_norm.copy(), self._gene_legend[1][:]

    def get_distmat(self):
        """Get the distance matrix between structures in the collection,
        based on the genes currently in use.

        | Returns:
        |   distmat (np.ndarray): a (collection.length, collection.length)
        |                         array, containing the overall distance
        |                         (the norm of all individual gene distances)
        |                         between all pairs of structures.

        """

        if self._needs_recalc:
            self._recalc()

        return self._distmat
