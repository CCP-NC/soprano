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

'''
CLI to plot the extract and summarise dipolar couplings.

TODO
* implement symmetry/label-based averaging.
* implement averaging over functional groups.
* implement rotational averaging. 
'''

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "August 10, 2022"


import click
import numpy as np
from ase import io
from soprano.properties.labeling import MagresViewLabels
from soprano.properties.nmr import *
from soprano.data.nmr import  _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
import pandas as pd
import logging
import click_log
from soprano.scripts.cli_utils import \
                                    add_options,\
                                    DIPOLAR_OPTIONS, \
                                    NO_CIF_LABEL_WARNING, \
                                    get_missing_cols,\
                                    get_matching_cols,\
                                    print_results,\
                                    find_XHn_groups,\
                                    sortdf,\
                                    viewimages
HEADER = '''
@click_log.simple_verbosity_option(logger)
##################################################
#  Extracting Dipolar couplingsfrom magres file  #
'''
FOOTER = '''
#  End of dipolar coupling extraction            #
##################################################

'''


# logging
logging.captureWarnings(True)
logger = logging.getLogger('cli')
click_log.basic_config(logger)
@click.command()

# one of more magres files
@click.argument('files',
                nargs=-1,
                type=click.Path(exists=True),
                required=True)

@add_options(DIPOLAR_OPTIONS)
def dipolar(
        files,
        selection_i,
        selection_j,
        output,
        output_format,
        merge,
        isotopes,
        precision,
        sortby,
        sort_order,
        include,
        exclude,
        query,
        view,
        quiet,
        **kwargs
        ):
    """
    Extract and summarise dipolar couplings from structure files.
    
    Usage:
    soprano dipolar seedname.{magres|cif|POSCAR|etc}
    """

    if quiet:
        logging.basicConfig(level=logging.WARNING)
        verbose = False
    else:
        verbose = True
        logging.basicConfig(level=logging.INFO)

    # set pandas print precision
    pd.set_option('display.precision', precision)
    # make sure we output all rows, even if there are lots!
    pd.set_option('display.max_rows', None)
    
    dfs = []
    images = []
    # loop over files
    for fname in files:

        logger.info(HEADER)
        logger.info(fname)

        # try to read in the file:
        try:
            atoms = io.read(fname)
        except IOError:
            logger.error(f"Could not read file {fname}, skipping.")
            continue
        
        # Inform user of best practice RE CIF labels
        if not has_cif_labels(atoms):
            logger.info(NO_CIF_LABEL_WARNING)

        # Selections -- if None, does all combinations
        sel_i = AtomSelection.all(atoms)
        sel_j = AtomSelection.all(atoms)
        
        # select subset of atoms based on selection string
        if selection_i:
            logger.info(f'\nSelecting atoms based on selection string: {selection_i}')
            sel_i = AtomSelection.from_selection_string(atoms, selection_i)
        if selection_j:
            logger.info(f'\nSelecting atoms based on selection string: {selection_j}')
            sel_j = AtomSelection.from_selection_string(atoms, selection_j)


        df = extract_dipolar_couplings(atoms,
                                       sel_i=sel_i,
                                       sel_j = sel_j,
                                       isotopes = isotopes)
        # add file info
        df['file'] = fname

        if query:
        # use pandas query to filter the dataframe
            logger.info(f'\nFiltering dataframe using query: {query}')
            df.query(query, inplace=True)
            logger.info(f'-----> Filtered to {len(df)} sites.')
        
        # reformat the 'v' column
        df['v'] = df['v'].apply(lambda x: np.round(x, precision))

        # what columns should we include/exclude?
        essential_columns = ['pair',
                            'label_i',
                            'label_j',
                            'isotope_i',
                            'isotope_i',
                            'file',
                            'D',
                            'alpha',
                            'beta']
        if include:
            # what columns should we include/exclude?
            specified_columns = [c for c in include if c not in essential_columns]
            logger.info(f'\nIncluding only columns containing: {specified_columns}')
            columns_to_include =essential_columns + specified_columns
            missing_columns = get_missing_cols(df, columns_to_include)
            if len(missing_columns) > 0:
                logger.warn(f'These columns specified {missing_columns}'
                            f' do not match any in the dataframe ({df.columns})')
            columns_to_include = get_matching_cols(df, columns_to_include)
            df = df[columns_to_include].copy()
        if exclude:
            logger.info(f'\nExcluding columns: {exclude}')
            # remove those that are already not in df
            specified_columns = get_matching_cols(df, exclude)
            df = df.drop(specified_columns, axis=1)
        # drop any that have only NaN values
        df = df.dropna(axis=1, how='all')


        if len(df) > 0:
            # done -- save to lists
            dfs.append(df)
            images.append(atoms)
            logger.info(FOOTER)
 
    if view:
        # TODO: make sure indices match the original/df indices (currently don't if organic!)
        viewimages(images)
        
    if merge:
        # merge all dataframes into one
        dfs = [pd.concat(dfs, axis=0)]
    for i, df in enumerate(dfs):
        dfs[i] = sortdf(df, sortby, sort_order)
    # write to file(s)
    print_results(dfs, output, output_format, verbose)






def extract_dipolar_couplings(atoms, **kwargs):
    """
    Extracts the dipolar couplings from the atoms object.
    :param atoms : ASE atoms object
    :return: a pandas dataframe with the couplings
    """
    # isotopes
    if 'isotopes' in kwargs:
        isotopes = kwargs['isotopes']
    else:
        isotopes = None
    elements = atoms.get_chemical_symbols()
    isotopelist = _get_isotope_list(elements, isotopes=isotopes, use_q_isotopes=False)
    species = [f'{iso}{el}' for el, iso in zip(elements, isotopelist)]


    dip = DipolarCoupling.get(atoms, **kwargs)
    if has_cif_labels(atoms):
        labels = atoms.get_array('labels')
    else:
        labels = MagresViewLabels.get(atoms)

    # transform dip into columns 
    pairs, values = zip(*sorted(dip.items()))
    idx_i, idx_j = zip(*pairs)
    d, v = zip(*values)
    # convert to np array
    d = np.array(d)
    # d = d * 2 * np.pi # convert to rad/s? <- not conventional in SSNMR!
    d = d * 1e-3 # convert to kHz
    dip_df = pd.DataFrame({
        'pair': pairs,
        'label_i':   [labels[i] for i in idx_i],
        'label_j':   [labels[j] for j in idx_j],
        'isotope_i': [species[i] for i in idx_i],
        'isotope_j': [species[j] for j in idx_j],
        'D': d,
        'v': v})

    # get angles in degrees
    dip_df['alpha'] = dip_df.v.apply(lambda x: np.arctan2(-x[1], -x[0]) *180/np.pi)
    dip_df['beta']  = dip_df.v.apply(lambda x: np.arccos(-x[2])         *180/np.pi)
    # wrap back to -180 to 180
    dip_df['alpha'] = dip_df['alpha'].apply(lambda x: (x + 180) % 360 - 180)
    dip_df['beta']  = dip_df['beta'].apply( lambda x: (x + 180) % 360 - 180)
    return dip_df
