# Soprano - a library to crack crystals! by Simone Sturniolo
# Copyright (C) 2016 - Science and Technology Facility Council

# Soprano is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Soprano is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Phylogenetic clustering class definitions"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import warnings
import numpy as np
from scipy.cluster import hierarchy, vq
from scipy.spatial import distance as spdist
# 2-to-3 compatibility
try:
    import cPickle as pickle
except ImportError:
    import pickle
# Internal imports
from soprano.utils import get_sklearn_clusters
from soprano.collection import AtomsCollection
from soprano.analyse.phylogen.genes import (Gene, GeneDictionary,
                                            GeneError, load_genefile)
from soprano.analyse.phylogen import mapping

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
        |   genes (list[tuple], str, file): list of the genes that should be
        |                                   loaded immediately; each gene
        |                                   comes in the form of a tuple
        |                                   (name (str), weight (float),
        |                                   params (dict)). A path or open
        |                                   file can also be passed for a
        |                                   .gene file, from which the values
        |                                   will be loaded.
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

    def set_genes(self, genes, load_arrays=False):
        """Calculate, store and set a list of genes as used for clustering.

        | Args:
        |   genes (list[soprano.analyse.phylogen.Gene],
        |          file, str): a list of Genes to calculate and store. A path
        |                      or open file can also be passed for a .gene
        |                      file, from which the values will be loaded.
        |   load_arrays (bool): try loading the genes as arrays from the
        |                       collection before generating them. Warning:
        |                       if there are arrays named like genes but with
        |                       different contents this can lead to
        |                       unpredictable results.

        """

        # Check if it can be opened:
        try:
            temp = load_genefile(genes)
            genes = temp
        except TypeError:
            pass

        self._genes = []
        self._has_pairgenes = False

        for g in genes:
            self._genes.append(g)
            # First, check if it's already stored
            if g.name in self._gene_storage:
                # Is it the same?
                if self._gene_storage[g.name]['def'] == g:
                    continue
            # If it's not, it needs to be calculated
            gene_val = None
            self._has_pairgenes = self._has_pairgenes or\
                g.is_pair

            if load_arrays:
                # Check if it is present
                try:
                    gene_val = self._collection.get_array(g.name)
                except ValueError:
                    pass
            # And confirm that there are no NaNs inside
            if gene_val is None or np.any(np.isnan(gene_val)):
                gene_val = g.evaluate(self._collection)

            self._gene_storage[g.name] = {
                'def': g,
                'val': gene_val
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
            gene_val = self._gene_storage[g.name]['val']
            # Here we need to perform a reshaping
            if not g.is_pair and len(gene_val.shape) == 1:
                gene_val = gene_val.reshape((-1, 1))
            elif g.is_pair and len(gene_val.shape) == 2:
                gene_val = gene_val.reshape((gene_val.shape[0],
                                             gene_val.shape[1],
                                             1))
            # Now normalization and weights
            if not g.is_pair:
                # Append
                self._gene_vectors_raw = np.append(self._gene_vectors_raw,
                                                   gene_val,
                                                   axis=-1)
                gn = gene_val.shape[-1]
                g_vecs_weights += [g.weight/np.sqrt(gn)]*gn
                self._gene_legend[0].append((g.name, gn))
            else:
                # Append
                self._gene_matrices_raw = np.append(self._gene_matrices_raw,
                                                    gene_val,
                                                    axis=-1)
                gn = gene_val.shape[-1]
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
                vspan = vmax-vmin
                # Fix the risk of division by zero
                vspan = np.where(np.isclose(vspan, 0), np.inf, vspan)
                vnorm = (vnorm-vmin)/vspan
                vnorm *= self._normrng[1]-self._normrng[0]
                vnorm += self._normrng[0]
        self._gene_vectors_norm = vnorm*g_vecs_weights

        mnorm = self._gene_matrices_raw.copy()
        if self._normdist is not None:
            mmax = np.amax(mnorm, axis=(0, 1))
            if not np.isclose(mmax, 0):
                mnorm *= self._normdist/mmax
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

    def get_linkage(self, method='single'):
        """Get the linkage matrix between structures in the collection,
        based on the genes currently in use. Only used in hierarchical
        clustering.

        Calls scipy.cluster.hierarchy.linkage.

        | Args:
        |   method (str): clustering method to employ. Valid entries are
        |                 'single', 'complete', 'weighted' and 'average'.
        |                 Refer to Scipy documentation for further details.

        | Returns:
        |   Z (np.ndarray): linkage matrix for the structures in the
        |                   collection. Refer to Scipy documentation for
        |                   details about the method

        """

        if self._needs_recalc:
            self._recalc()

        cdist = spdist.squareform(self._distmat)

        return hierarchy.linkage(cdist, method=method)

    def get_hier_clusters(self, t, method='single'):
        """Get multiple clusters (in the form of a list of collections) based
        on the hierarchical clustering methods and the currently set genes.

        Calls scipy.cluster.hierarchy.fcluster

        | Args:
        |   t (float): minimum distance of separation required to consider
        |              two clusters separate. This controls the number of
        |              clusters: a smaller value will produce more fine
        |              grained clustering. At the limit, a value smaller than
        |              the distance between the two closest structures will
        |              return a cluster for each structure. Remember that the
        |              'distances' in this case refer to distances between the
        |              'gene' values attributed to each structure. In other
        |              words they are a function of the chosen genes,
        |              normalization conditions and weights employed.
        |              In addition, the way they are calculated depends on the
        |              choice of method.
        |   method (str): clustering method to employ. Valid entries are
        |                 'single', 'complete', 'weighted' and 'average'.
        |                 Refer to Scipy documentation for further details.

        | Returns:
        |   clusters (tuple(list[int],
        |                   list[slices])): list of cluster index for each
        |                                   structure (counting from 1) and
        |                                   list of slices defining the
        |                                   clusters as formed by hierarchical
        |                                   algorithm.

        """

        if self._needs_recalc:
            self._recalc()

        Z = self.get_linkage(method=method)
        clusts = hierarchy.fcluster(Z, t, criterion='distance')
        clust_n = np.amax(clusts)
        clust_slices = [np.where(clusts == i)[0] for i in range(1, clust_n+1)]

        return clusts, clust_slices

    def get_hier_tree(self, method='single'):
        """Get a tree data structure describing the clustering order of based
        on the hierarchical clustering methods and the currently set genes.

        Calls scipy.cluster.hierarchy.to_tree

        | Args:
        |   method (str): clustering method to employ. Valid entries are
        |                 'single', 'complete', 'weighted' and 'average'.
        |                 Refer to Scipy documentation for further details.

        | Returns:
        |   root_node (ClusterNode): the root node of the tree. Access child
        |                            members with .left and .right, while .id
        |                            holds the number of the corresponding
        |                            cluster. Refer to Scipy documentation for
        |                            further details.

        """

        if self._needs_recalc:
            self._recalc()

        Z = self.get_linkage(method=method)
        return hierarchy.to_tree(Z)

    def get_max_cluster_dist(self):
        """Return the maximum possible distance between two clusters"""

        return np.linalg.norm([g.weight for g in self._genes])

    def get_kmeans_clusters(self, n):
        """Get a given number of clusters (in the form of a list of
        collections) based on the k-means clustering methods
        and the currently set genes.
        Warning: this method only works if there are no genes that work only
        with pairs of structures - as specific points, and not just distances
        between them, are required for this algorithm.

        Calls scipy.cluster.vq.kmeans

        | Args:
        |   n (int):    the desired number of clusters.

        | Returns:
        |   clusters (tuple(list[int],
        |                   list[slices])): list of cluster index for each
        |                                   structure (counting from 1) and
        |                                   list of slices defining the
        |                                   clusters as formed by k-means
        |                                   algorithm.


        """

        # Sanity check
        if self._has_pairgenes:
            # Then this method can't work!
            raise RuntimeError('k-means clustering can not be used with'
                               'presence of pair distance genes')

        if self._needs_recalc:
            self._recalc()

        centroids, dist = vq.kmeans(self._gene_vectors_norm, n)
        clusts, cdists = vq.vq(self._gene_vectors_norm, centroids)
        clust_n = np.amax(clusts)+1
        clust_slices = [np.where(clusts == i)[0] for i in range(clust_n)]
        clusts += 1

        return clusts, clust_slices

    def get_sklearn_clusters(self, method, params={}):
        """Get clusters applying any of the methods provided by the library
        scikit-learn (requires a separate installation).
        Warning: this method only works if there are no genes that work only
        with pairs of structures - as use of pairwise clustering methods is
        not implemented yet.

        Uses the sklearn.cluster.<method> class

        | Args:
        |   method (str): name of the clustering class from sklearn.clusters
        |                 to use. For reference check the documentation at
        |                 http://scikit-learn.org/stable/modules/clustering.html
        |   params (dict): parameters to be passed to the class when
        |                  initialising it. Change depending on the desired
        |                  method. Check the documentation for the specific
        |                  class.

        | Returns:
        |   clusters (tuple(list[int],
        |                   list[slices])): list of cluster index for each
        |                                   structure (counting from 1) and
        |                                   list of slices defining the
        |                                   clusters as formed by the
        |                                   requested algorithm.

        """

        # Sanity check
        if self._has_pairgenes:
            # Then this method can't work!
            raise RuntimeError('k-means clustering can not be used with'
                               'presence of pair distance genes')

        if self._needs_recalc:
            self._recalc()

        clusts = get_sklearn_clusters(self._gene_vectors_norm, method, params)
        clust_n = np.amax(clusts)+1
        clust_slices = [np.where(clusts == i)[0] for i in range(clust_n)]
        clusts += 1

        return clusts, clust_slices

    def get_clusters(self, method, params={}):
        """Wrapper method to get clusters by any available method. Depending
        on the value passed as 'method' it calls either ger_hier_clusters,
        get_kmeans_clusters, or get_sklearn_clusters. Check their respective
        docstrings for more detailed info.

        | Args:
        |   method (str): name of the clustering method to use. Can be 'hier',
        |                 'kmeans', or one of the methods in sklearn.clusters.
        |   params (dict): parameters to be passed to the class when
        |                  initialising it. Change depending on the desired
        |                  method. Check the documentation for the specific
        |                  class.

        | Returns:
        |   clusters (tuple(list[int],
        |                   list[slices])): list of cluster index for each
        |                                   structure (counting from 1) and
        |                                   list of slices defining the
        |                                   clusters as formed by the
        |                                   requested algorithm.

        """

        if method == 'hier':
            try:
                return self.get_hier_clusters(params['t'])
            except KeyError:
                raise ValueError('Parameter t required for hierarchical'
                                 ' clustering')
        elif method == 'kmeans':
            try:
                return self.get_kmeans_clusters(params['n'])
            except KeyError:
                raise ValueError('Parameter n required for k-means'
                                 ' clustering')
        else:
            return self.get_sklearn_clusters(method, params)

    def get_cluster_stats(self, clusters, raw=False):
        """Compute average values and standard deviation for each gene within
        a given clustering.

        | Args:
        |   clusters (tuple): the clustering in tuple form, as returned by one
        |                     of the get_clusters methods.
        |   raw (bool): if True, return average and standard deviation of raw
        |               instead of normalised gene values. Default is False.

        | Returns:
        |   avgs (np.ndarray): 2D array of average values of each gene for 
        |                      each cluster.
        |   stds (np.ndarray): 2D array of standard deviations of each gene 
        |                      for each cluster.
        |   genome_legend (list[tuple]): a list of tuples containing (name,
        |                                length) of the gene fragments in the
        |                                arrays
        """

        # Sanity check
        if self._has_pairgenes:
            warnings.warn('Pair distance gene stats can not be calculated')

        if raw:
            gv = self._gene_vectors_raw.copy()
        else:
            gv = self._gene_vectors_norm.copy()

        inds, slices = clusters

        avgs = np.zeros((len(slices), gv.shape[1]))
        stds = np.zeros((len(slices), gv.shape[1]))

        for i, sl in enumerate(slices):
            avgs[i] = np.average(gv[sl], axis=0)
            stds[i] = np.std(gv[sl], axis=0)

        return avgs, stds, self._gene_legend[0][:]

    def get_elbow_plot(self, method='kmeans', param_name='n',
                       param_range=range(1, 11)):
        """Returns data for an elbow plot by scanning the outcome of a given
        clustering method within a range of values for a chosen parameter.
        Used to determine optimal parameter values.

        | Args:
        |   method (str): name of the clustering method to use. Can be 'hier',
        |                 'kmeans', or one of the methods in sklearn.clusters.
        |                 Default is kmeans.
        |   param_name (str): parameter to be scanned over. Change depending
        |                     on the desired method. Check the documentation
        |                     for the specific class. Default is n, number of
        |                     clusters for k-means method.
        |   param_range (list): values of param_name to scan over. Default is
        |                       the integers from 1 to 10.

        | Returns:
        |   wss (np.ndarray): values of the "Within cluster Sum of Squares"
        |                     (WSS) to be used on the elbow plot y axis.
        |   param_range (list): range used for parameter scan, to be used on
        |                       the x axis (same as passed by the user).
        """

        wss = []

        for pval in param_range:
            clusts = self.get_clusters(method, params={param_name: pval})
            stats = self.get_cluster_stats(clusts)
            wss.append(np.sum(stats[1]**2))

        return np.array(wss), param_range

    def create_mapping(self, method="total-principal"):
        """Return an array of 2-dimensional points representing a reduced
        dimensionality mapping of the given genes using the algorithm of
        choice. All algorithms are described in [W. Siedlecki et al., Patt.
        Recog. vol. 21, num. 5, pp. 411 429 (1988)].

        | Args:
        |   method (str): can be one of the following algorithms:
        |                     - total_principal (default)
        |                     - clafic
        |                     - fukunaga-koontz
        |                     - optimal-discriminant

        """

        # Sanity check
        if self._has_pairgenes:
            # Then this method can't work!
            raise RuntimeError('No mapping can be performed in presence of'
                               'pair distance genes')

        if self._needs_recalc:
            self._recalc()

        algos = {
            'total-principal': mapping.total_principal_component,
            'clafic': mapping.classcond_principal_component,
            'fukunaga-koontz': mapping.standard_classcond_component,
            'optimal-discriminant': mapping.optimal_discriminant_plane
        }

        try:
            return algos[method](self._gene_vectors_norm)
        except KeyError:
            raise ValueError(('Invalid method passed to create_mapping.\n'
                              'Valid methods are: \n-') +
                             ('\n-'.join(algos.keys())))

    def save_collection(self, filename):
        """Save as pickle the collection bound to this PhylogenCluster.
        The calculated genes are also stored in it as arrays for future use.

        """

        saveC = self._collection[:]

        for g in self._gene_storage:
            saveC.set_array(g, self._gene_storage[g]['val'])

        saveC.save(filename)

    def save(self, filename):
        """Simply save a pickled copy to a given file path"""

        f = open(filename, 'w')
        pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Load a pickled copy from a given file path"""

        f = open(filename)
        f = pickle.load(f)
        if not isinstance(f, PhylogenCluster):
            raise ValueError('File does not contain a PhylogenCluster object')
        return f
