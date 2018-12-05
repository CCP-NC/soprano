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


""" A script that allows to run a phylogenetic analysis as a command-line
tool"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import numpy as np
import argparse as ap
from soprano.collection import AtomsCollection
from soprano.properties.basic import CalcEnergy
from soprano.analyse.phylogen import PhylogenCluster, load_genefile

try:
    import matplotlib.pyplot as plt
    _has_matplotlib = True
except:
    _has_matplotlib = False


def genomes2png(aC, pC, outf, dpi=90):
    """Produce a PNG plot of a series of genomes vectors"""

    genomes, legend = pC.get_genome_vectors_norm()
    h, w = genomes.shape
    # Now build the labels
    xlabels = ['{0}_{1}'.format(l[0], i+1)
               for l in legend
               for i in range(l[1])]
    ylabels = aC.all.map(lambda s: s.info['name'])
    # Calculate the size
    figw = (w*90.0+200.0)
    figh = (h*50.0+100.0)
    plt.figure(figsize=(figw/dpi, figh/dpi), dpi=dpi)
    plt.subplots_adjust(left=150.0/figw, right=1.0-50.0/figw, bottom=50.0/figh,
                        top=1.0-50.0/figh)
    plt.pcolor(genomes, edgecolors='black', cmap='jet')
    plt.tick_params(labelsize=10)
    plt.xticks(np.linspace(0, w-1, w)+0.5, xlabels)
    plt.yticks(np.linspace(0, h-1, h)+0.5, ylabels)
    plt.savefig(outf, format='png', dpi=dpi)


def tree2json(aC, pC, w=500, h=500):
    """Produce a dictionary representation of a Scipy cluster tree object
    suitable to be saved as JSON"""

    max_dist = pC.get_max_cluster_dist()

    # First find absolute minimum and maximum energy (for hue)

    struct_Es = CalcEnergy.get(aC)

    Emin = min(struct_Es)
    Emax = max(struct_Es)

    # Create the master node by using the tree root, then visit recursively

    def visit_tree(node):

        datastr = {'value': node.get_count(),
                   'id': node.get_id(),
                   'children': []}

        # Add label text if possible

        for br in (node.left, node.right):
            if br is None:
                continue
            datastr['children'].append(visit_tree(br))

        # Now the hue and the struct name. If this is a leaf, just use the
        # structure value
        if len(datastr['children']) == 0:
            if Emax > Emin:
                hue = (Emax - CalcEnergy.get(aC.structures[node.get_id()])) \
                    / (Emax - Emin)*240.0
            else:
                hue = 120.0
            struct_name = aC.structures[node.get_id()].info['name']
        else:
            # Otherwise, an average
            hue = 0.0
            struct_name = ""
            for c in datastr['children']:
                hue += c['hue']
                struct_name += c['struct_name'] + ',\n'
            hue /= len(datastr['children'])
            struct_name = struct_name[:-2]  # Remove last newline and comma

        datastr['struct_name'] = struct_name
        datastr['hue'] = hue

        return datastr

    return visit_tree(pC.get_hier_tree())

# Entry point
def __main__():
    # So, first parse the command line arguments
    parser = ap.ArgumentParser()
    # Main argument
    parser.add_argument('seedname', type=str, default=None,
                        help="Seedname of the job")
    parser.add_argument('input_files', type=str, nargs='+', default=None,
                        help="Files to load and analyse")
    # Optional arguments
    # Boolean args
    parser.add_argument('-pkl',   action='store_true', default=False,
                        help="Input is one or more AtomsCollections"
                        " in PICKLE format")
    parser.add_argument('-tree',  action='store_true', default=False,
                        help="Create JSON tree file")
    parser.add_argument('-distmat',  action='store_true', default=False,
                        help="Save a distance matrix file")
    parser.add_argument('-forcemat',   action='store_true', default=False,
                        help="Create JSON distance matrix file for force "
                        "layout")
    parser.add_argument('-genomes',   action='store_true', default=False,
                        help="Save genomes text file")
    parser.add_argument('-gene_png',   action='store_true', default=False,
                        help="Save genomes png file")
    parser.add_argument('-gene_clamp',   action='store_true', default=True,
                        help="Clamp genes between 0 and 1")
    # Value args
    parser.add_argument('-n', type=int, default=None,
                        help="Number of lowest energy structures to keep")
    # parser.add_argument('-Z', type=int, default=None,
    # help="Expected numbers of molecules in cell (discard structures "
    # "that don't agree)")
    parser.add_argument('-corrmat_minforce',     type=float, default=0.01,
                        help="Correlation matrix minimum force")
    parser.add_argument('-corrmat_maxforce',     type=float, default=1.0,
                        help="Correlation matrix maximum force")
    parser.add_argument('-savepkl',  type=str, default=None,
                        help="Directory to save collection as pickle file")

    args = parser.parse_args()

    # First, parse the gene list
    gene_list = load_genefile(args.seedname + '.gene')

    # Then create an AtomsCollection
    if not args.pkl:
        aC = AtomsCollection(args.input_files,
                             info={'name': args.seedname},
                             cell_reduce=True,
                             progress=True)
    else:
        aC = AtomsCollection([])
        for f in args.input_files:
            aC += AtomsCollection.load(f)

    # Take only n lowest energies if required
    if (args.n is not None):
        # Sort by energy
        CalcEnergy.get(aC, store_array=True)
        aC = aC.sorted_byarray('calc_energy')[:args.n]

    #discard_Z = args.Z is not None

    # Calculate genomes
    nrange = (0, 1) if args.gene_clamp else (None, None)
    ndist = 1.0 if args.gene_clamp else None
    pC = PhylogenCluster(aC, genes=[], norm_range=nrange, norm_dist=ndist)
    pC.set_genes(gene_list, load_arrays=True)

    if args.savepkl is not None:
        # Save the structure
        pC.save_collection(os.path.join(args.savepkl, args.seedname + '.pkl'))

    # And now for the output
    if args.genomes:
        with open(args.seedname + "_genomes.dat", 'w') as ofile:
            genvec, legend = pC.get_genome_vectors()
            # Create a header
            header = ('Structure\t'
                      '{0}\n').format('\t'.join(['\t'.join(
                          ['{0}_{1}'.format(l[0],
                                            i+1)
                           for i in range(l[1])])
                          for l in legend]))
            ofile.write(header)
            try:
                for i, gen in enumerate(genvec):
                    ofile.write(aC.structures[i].info['name'] +
                                '\t' + ' '.join([str(x) for x in gen]) + '\n')
            except KeyError:
                # Someone has no name!
                raise RuntimeError('Not all structures have a name, '
                                   'can\'t print out genomes as table')

    if args.gene_png and _has_matplotlib:
        with open(args.seedname + "_genomes.png", 'w') as ofile:
            genomes2png(aC, pC, ofile)

    if args.tree:
        tree = pC.get_hier_tree()
        with open(args.seedname + "_tree.json", 'w') as ofile:
            json.dump(tree2json(aC, pC), ofile)
