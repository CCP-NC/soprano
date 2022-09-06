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

'''CLI to extract and process NMR-related properties from .magres files.

TODO: add support for different shift {Haeberlen,NQR,IUPAC}and quadrupole {Haeberlen,NQR} conventions.
TODO: check if df is too wide to fit in window -- if so, split into multiple plots.
TODO: spinsys output is not yet implemented.
TODO: document config file setup
'''

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "July 08, 2022"


import click
import numpy as np
from ase import io
from ase.units import Ha, Bohr
from soprano.properties.labeling import UniqueSites, MagresViewLabels
from soprano.properties.nmr import *
from soprano.data.nmr import _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
import pandas as pd
import warnings
import logging
import click_log
from soprano.scripts.cli_utils import \
                                    add_options,\
                                    NMREXTRACT_OPTIONS, \
                                    NO_CIF_LABEL_WARNING, \
                                    average_quaternions_by_tags,\
                                    get_missing_cols,\
                                    get_matching_cols,\
                                    print_results,\
                                    find_XHn_groups,\
                                    sortdf,\
                                    viewimages
# logging
logging.captureWarnings(True)
logger = logging.getLogger('cli')
click_log.basic_config(logger)
HEADER = '''
##########################################
#  Extracting NMR info from magres file  #
'''
FOOTER = '''
# End of NMR info extraction            #
##########################################
'''


@click_log.simple_verbosity_option(logger)

@click.command()

# one of more files
@click.argument('files',
                nargs=-1,
                type=click.Path(exists=True),
                required=True)

@add_options(NMREXTRACT_OPTIONS)

def nmr(files,
        selection,
        output,
        output_format,
        merge,
        isotopes,
        references,
        gradients,
        reduce,
        average_group,
        combine_rule,
        symprec,
        properties,
        precision,
        euler_convention,
        sortby,
        sort_order,
        include,
        exclude,
        query,
        view,
        quiet):
    """
    Extract and analyse NMR data from magres file(s).
    
    Usage:
    soprano nmr seedname.magres

    Processes .magres file(s) containing NMR-related properties
    and prints a summary. It defaults to printing all NMR properties
    present in the file for all the atoms. 
    
    See the below arguments for how to extract specific information.
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
    
    nfiles = len(files)
    dfs = []
    images = []
    # loop over files
    for fname in files:

        logger.info(HEADER)
        logger.info(fname)
        logger.info(f"\nExtracting properties: {properties}")

            


        # try to read in the file:
        try:
            atoms = io.read(fname)
        except IOError:
            logger.error(f"Could not read file {fname}, skipping.")
            continue
        
        # Do they actually have any magres data?
        if not any([atoms.has(k) for k in properties]):
            logger.error(f"File {fname} has no {' '.join(properties)} data to extract. Skipping.")
            continue

        # Inform user of best practice RE CIF labels
        if not has_cif_labels(atoms):
            logger.info(NO_CIF_LABEL_WARNING)
            

        all_selections = AtomSelection.all(atoms)
        # create new array for multiplicity
        multiplicity = np.ones(len(atoms))
        atoms.set_array('multiplicity', multiplicity)

        # note we must change datatype to allow more space!
        labels = atoms.get_array('labels').astype('U25')
        # reduce by symmetry?
        tags = np.arange(len(atoms))

        if reduce:

            logger.info('\nTagging equivalent sites')
            # tag equivalent sites
            tags = UniqueSites.get(atoms, symprec=symprec)

            # log the number of unique sites
            unique_sites, unique_site_idx = np.unique(tags, return_index=True)
            logger.info(f'    This leaves {len(unique_sites)} unique sites')
            logger.info(f'    The unique site labels are: {labels[unique_site_idx]}')
                


        if average_group:
            XHn_groups = find_XHn_groups(atoms, average_group, tags= tags, vdw_scale=1.0)
            for ipat, pattern in enumerate(XHn_groups):
                # check if we found any that matched this pattern
                if len(pattern) == 0:
                    logging.warn(f"No XHn groups found for pattern {average_group.split(',')[ipat]}")
                    continue
                
                logger.info(f"Found {len(pattern)} {average_group.split(',')[ipat]} groups")
                # get the indices of the atoms that matched this pattern
                # update the tags and labels accordingly
                for ig, group in enumerate(pattern):
                    logger.info(f"    Group {ig} contains: {np.unique(labels[group])}")
                    # fix labels here as aggregate of those in group
                    combined_label = '--'.join(np.unique(labels[group]))
                    # labels[group] = f'{ig}'#combined_label
                    labels[group] = combined_label

                    tags[group] = -(ipat+1)*1e5-ig
        # update atoms object with new labels
        atoms.set_array('labels', labels)
        # update atoms tags
        atoms.set_tags(tags)
        
        # select subset of atoms based on selection string
        if selection:
            logger.info(f'\nSelecting atoms based on selection string: {selection}')
            sel_selectionstring = AtomSelection.from_selection_string(atoms, selection)
            all_selections *= sel_selectionstring
        elements = atoms.get_chemical_symbols()
        isotopelist = _get_isotope_list(elements, isotopes=isotopes, use_q_isotopes=False)
        species = [f'{iso}{el}' for el, iso in zip(elements, isotopelist)]
        
        df = pd.DataFrame({
                'indices': atoms.get_array('indices'),
                'labels': labels,
                'species':species,
                'multiplicity': atoms.get_array('multiplicity'),
                'tags': tags,
                })

        # If there are no cif labels, generate and save MagresView-style labels
        if not has_cif_labels(atoms):
            # generate MagresView-type Labels
            magresview_labels = MagresViewLabels.get(atoms)
            df.insert(2, 'MagresView_labels', magresview_labels)

        # Let's add a column for the file name -- useful to keep track of 
        # which file the data came from if merging multiple files.
        df['file'] = fname
        if 'ms' in properties:
            try:
                ms_summary = pd.DataFrame(get_ms_summary(atoms, euler_convention, references, gradients))
                if not references:
                    # drop shift column if no references are given
                    ms_summary.drop(columns=['MS_shift'], inplace=True)

                df = pd.concat([df, ms_summary], axis=1)
            except RuntimeError:
                warnings.warn(f'No MS data found in {fname}\n'
                'Set argument `-p efg` if the file(s) only contains EFG data ')
                pass
            except:
                warnings.warn('Failed to load MS data from .magres')
                raise
        if 'efg' in properties:
            try:
                efg_summary = pd.DataFrame(get_efg_summary(atoms, isotopes, euler_convention))
                df = df = pd.concat([df, efg_summary], axis=1)
            except RuntimeError:
                warnings.warn(f'No EFG data found in {fname}\n'
                'Set argument `-p ms` if the file(s) only contains MS data ')
                pass
            except:
                warnings.warn('Failed to load EFG data from .magres')
                raise

        # Apply selections 
        selection_indices = all_selections.indices
        # sort
        selection_indices.sort()
        # extract from df
        df = df.iloc[selection_indices]

        # apply group averaging
        if average_group or reduce:
            # These are the rules for aggregating groups
            # Default rule: take the mean
            aggrules = dict.fromkeys(df, combine_rule)
            # note we could add more things here! e.g.
            # aggrules = dict.fromkeys(df, ['mean', 'std'])
            # for most of the columns that have objects, we just take the first one
            aggrules.update(dict.fromkeys(df.columns[df.dtypes.eq(object)], 'first'))
            
            # we no longer need these two columns
            del aggrules['indices']
            del aggrules['tags']

            aggrules['labels'] = set
            if 'MagresViewLabels' in df.columns:
                aggrules['MagresView_labels'] = set
            aggrules['multiplicity'] = 'count'

            logger.info('\nAveraging over sites with the same tag')
            logger.info(f'   We apply the following rules to each column:\n {aggrules}')
            # apply group averaging
            grouped = df.groupby('tags')
            df = grouped.agg(aggrules).reset_index()
            # fix the labels print formatting            
            df['labels'] = df['labels'].apply(lambda x: ','.join(x))
            if 'MagresView_labels' in df.columns:
                df['MagresView_labels'] = df['MagresView_labels'].apply(lambda x: ','.join(sorted(list(x))))
            
        
        

        
        total_explicit_sites = df['multiplicity'].sum()
        logger.info(f'\nFound {total_explicit_sites} total sites.')
        if average_group or reduce:
            logger.info(f'    -> reduced to {len(df)} sites after averaging equivalent ones')


        if query:
            # use pandas query to filter the dataframe
            logger.info(f'\nFiltering dataframe using query: {query}')
            df.query(query, inplace=True)
            logger.info(f'-----> Filtered to {len(df)} sites.')


        # what columns should we include/exclude?
        essential_columns = ['labels', 'species', 'multiplicity', 'tags', 'file']
        if include:
            # what columns should we include/exclude?
            essential_columns = ['labels', 'species', 'multiplicity', 'tags', 'file']
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
            dfs.append(df)
            images.append(atoms)
            logger.info(FOOTER)
        # if the df is empty, raise warning and don't append
        else:
            logger.warn(f"No results found for {fname}.\n "
            "Try removing filters/checking the file contents.")
            
    if view:
        viewimages(images)
        
    if merge:
        # merge all dataframes into one
        dfs = [pd.concat(dfs, axis=0)]
    for i, df in enumerate(dfs):
        dfs[i] = sortdf(df, sortby, sort_order)
    
    # write to file(s)
    print_results(dfs, output, output_format, verbose)
        


def get_ms_summary(atoms, euler_convention, references, gradients):
    '''
    For an Atoms object with ms tensor arrays, return a summary of the tensors.
    '''
    # Isotropy, Anisotropy and Asymmetry (Haeberlen convention)
    iso   = MSIsotropy.get(atoms)
    shift  = MSIsotropy.get(atoms, ref=references, grad=gradients)
    aniso = MSAnisotropy.get(atoms)
    red_aniso = MSReducedAnisotropy.get(atoms)
    asymm = MSAsymmetry.get(atoms)
    # Span and skew
    span = MSSpan.get(atoms)
    skew = MSSkew.get(atoms)
    # quaternion
    quat = MSQuaternion.get(atoms)
    # We need to be carefull with the angle averaging
    quat = average_quaternions_by_tags(quat, atoms.get_tags())
    # Euler angles
    alpha, beta, gamma = np.array([q.euler_angles(mode=euler_convention)*180/np.pi for q in quat]).T
    ms_summary = {
            'MS_shielding': iso,
            'MS_shift': shift,
            'MS_anisotropy': aniso,
            'MS_reduced_anisotropy': red_aniso,
            'MS_asymmetry': asymm,
            'MS_span': span,
            'MS_skew': skew,
            'MS_alpha': alpha,
            'MS_beta': beta,
            'MS_gamma': gamma
            }
    return ms_summary
    

def get_efg_summary(atoms, isotopes, euler_convention):
    '''
    For an Atoms object with EFG tensor arrays, return a summary of the tensors.
    '''
    Vzz   = EFGVzz.get(atoms)
    # convert Vzz from au to V/m^2
    Vzz = Vzz * (Ha / Bohr) * 1e-1

    # For quadrupolar constants, isotopes become relevant. This means we need to create custom Property instances to
    # specify them. There are multiple ways to do so - check the docstrings for more details - but here we set them
    # by element. When nothing is specified it defaults to the most common NMR active isotope.
    qP = EFGQuadrupolarConstant(isotopes=isotopes) # Deuterated; for the others use the default
    qC = qP(atoms)/1e6 # To MHz
    
    # asymmetry
    eta = EFGAsymmetry.get(atoms)

    # quaternion
    quat = EFGQuaternion.get(atoms)
    # We need to be carefull with the angle averaging
    quat = average_quaternions_by_tags(quat, atoms.get_tags())
    # Euler angles
    alpha, beta, gamma = np.array([q.euler_angles(mode=euler_convention)*180/np.pi for q in quat]).T

    # NQR transitions
    nqrs = EFGNQR.get(atoms, isotopes=isotopes)
    # unique transitions
    transition_keys = sorted(set([k for nqr in nqrs for k in nqr.keys()]))
    nqr_dict = {}
    for k in transition_keys:
        header = f'EFG_NQR {k}'
        values = np.zeros(len(nqrs))
        for inqr, nqr in enumerate(nqrs):
            if k in nqr:
                values[inqr] = nqr[k] * 1e-6
            else:
                values[inqr] = np.nan
        nqr_dict[header] = values



    efg_summary = {
                'EFG_Vzz': Vzz,
                'EFG_quadrupolar_constant': qC,
                'EFG_asymmetry': eta,
                'EFG_alpha': alpha,
                'EFG_beta': beta,
                'EFG_gamma': gamma,
                **nqr_dict
                }


    return efg_summary










