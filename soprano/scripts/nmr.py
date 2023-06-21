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
from ase import Atoms
from ase.units import Ha, Bohr
from soprano.properties.labeling import UniqueSites, MagresViewLabels
from soprano.properties.nmr import *
from soprano.data.nmr import _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels, merge_sites
from soprano.properties.linkage import Molecules
import pandas as pd
import warnings
import logging
import click_log
from typing import List, Tuple, Dict, Union, Optional
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
                                    viewimages, \
                                    units_rename, \
                                    apply_df_filtering
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



@click.command()
# one of more files
@click.argument('files',
                nargs=-1,
                type=click.Path(exists=True),
                required=True)

@add_options(NMREXTRACT_OPTIONS)

def nmr(
        files,
        subset=None,
        output=None,
        output_format=None,
        merge=False,
        isotopes={},
        references={},
        gradients={},
        reduce=True,
        average_group=None,
        combine_rule='mean',
        symprec=1e-4,
        properties=['efg', 'ms'],
        precision = 3,
        euler_convention='zyz',
        sortby=None,
        sort_order='ascending',
        include=None,
        exclude=None,
        query=None,
        view=False,
        verbosity=0):
    """
    Extract and analyse NMR data from magres file(s).
    
    Usage:
    soprano nmr seedname.magres

    Processes .magres file(s) containing NMR-related properties
    and prints a summary. It defaults to printing all NMR properties
    present in the file for all the atoms. 
    
    See the below arguments for how to extract specific information.
    """
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    dfs, images = nmr_extract(
                    files,
                    subset = subset,
                    merge = merge,
                    isotopes = isotopes,
                    references = references,
                    gradients = gradients,
                    reduce = reduce,
                    average_group = average_group,
                    symprec = symprec,
                    properties = properties,
                    euler_convention = euler_convention,
                    sortby = sortby,
                    sort_order = sort_order,
                    include = include,
                    exclude = exclude,
                    query = query,
                    logger = logger,
                    )
    if view:
        viewimages(images)
        
    # write to file(s)
    print_results(dfs, output, output_format, precision, verbosity > 0)

def nmr_extract(
        files,
        subset=None,
        merge=False,
        isotopes={},
        references={},
        gradients={},
        reduce=True,
        average_group=None,
        merging_strategies= {
            # for the positions, just take the first one 
            'positions': lambda x: x[0],
            # for the labels, just take the first one
            'labels': lambda x: x[0],
        },
        symprec=1e-4,
        properties=['efg', 'ms'],
        euler_convention='zyz',
        sortby=None,
        sort_order='ascending',
        include=None,
        exclude=None,
        query=None,
        logger = None,
        ):
    '''
    Extract NMR data from magres file(s). See CLI help for more details on the arguments. (`soprano nmr --help`)

    Args:
        files (list): list of magres files to extract data from.
        subset (str): subset of atoms to extract data from. e.g. "H1,H2,C" for all H1, H2 and C atoms.
        merge (bool): whether to merge the pandas dataframes from more than one .magres file.
        isotopes (dict): dictionary of isotope labels to use for each element. e.g. {"H": "1H", "C": "13C"}.
        references (dict): dictionary of shielding reference values to use for each element. e.g. {"H": 30.0, "C": 100.0}.
        gradients (dict): dictionary of gradient reference values to use for each element. e.g. {"H": -1.0, "C": -0.95}. 
                          If not provided, the gradient is assumed to be -1 for all elements.
        reduce (bool): whether to reduce to symmetry equivalent sites (using either the CIF labels or symmetry operations found using SPGLIB).
        average_group (str): comma-separated list of functional groups to average over e.g. methyl groups. e.g. "CH3" averages over H atoms in methyl groups.
        merging_strategies (dict): dictionary of merging strategies to use for each property. e.g. {"positions": lambda x: x[0]}.
        symprec (float): tolerance for symmetry operations. Default is 1e-4.
        properties (list): list of properties to extract. Options are 'efg', 'ms'.
        euler_convention (str): convention to use for Euler angles. Options are 'zyz' or 'zxz'.
        sortby (str): column to sort the dataframe by.
        sort_order (str): order to sort the dataframe by. Options are 'ascending' or 'descending'.
        include (str): comma-separated list of columns to include in the output.
        exclude (str): comma-separated list of columns to exclude from the output.
        query (str): query string to filter the dataframe.
        logger (logging.Logger): logger to use for logging. If None, a new logger is created.


    
    Returns:
        dfs (list): list of pandas DataFrames containing the extracted data.
        images (list): list of ASE Atoms objects containing the crystal structures.
    '''

    if logger is None:
        logger = logging.getLogger(__name__)

    
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
            # immediately try to reload as molecular crystal
            atoms = reload_as_molecular_crystal(atoms)
            # label atoms 
            atoms = label_atoms(atoms)
        except IOError:
            logger.error(f"Could not read file {fname}, skipping.")
            continue
        
        # Do they actually have any magres data?
        if not any([atoms.has(k) for k in properties]):
            logger.error(f"File {fname} has no {' '.join(properties)} data to extract. Skipping.")
            continue


        all_selections = AtomSelection.all(atoms)
        # select subset of atoms based on selection string
        if subset:
            logger.info(f'\nSelecting atoms based on selection subset string: {subset}')
            sel_selectionstring = AtomSelection.from_selection_string(atoms, subset)
            all_selections *= sel_selectionstring
            logger.debug(f'    Selected atoms: {all_selections.indices}')
            ## apply selection string to atoms object
            atoms = all_selections.subset(atoms)
        
        # create new array for multiplicity
        multiplicity = np.ones(len(atoms))
        atoms.set_array('multiplicity', multiplicity)

        
        
        
        # reduce by symmetry?
        tags = np.arange(len(atoms))
        labels = np.asarray(atoms.get_array('labels'), dtype='U25')

        if reduce:
            logger.info('\nTagging equivalent sites')
            # tag equivalent sites
            tags = UniqueSites.get(atoms, symprec=symprec)
            

            # log the number of unique sites
            unique_sites, unique_site_idx = np.unique(tags, return_index=True)
            logger.debug(f'    This leaves {len(unique_sites)} unique sites')
            logger.debug(f'    The unique site labels are: {labels[unique_site_idx]}')

            # check to make sure that all sites with the same tag have the same MSIsotropy
            # if not, throw a warning, suggest to turn on debug logging and --no-reduce flag
            # and then continue
            if not check_equivalent_sites_ms(atoms, tags):
                logger.warning('    Some sites with the same symmetry tag/CIF label have different MS isotropy values.')
                logger.warning('    You can turn off symmetry reduction with the --no-reduce flag.')
                logger.warning('    You can also turn on debug logging with the -vv flag.')
                logger.warning('    If you find that the (symmetry) reduction algorithm is working incorrectly,')
                logger.warning('    please report this to the developers.')

        # set tags to atoms object
        atoms.set_tags(tags)

        if average_group:
            atoms = tag_functional_groups(average_group, atoms, vdw_scale=1.0)
        atoms = merge_tagged_sites(atoms, merging_strategies=merging_strategies)
        
        
        if isotopes:
            logger.info(f'\nCustom isotopes for: {isotopes}')
        # build the dataframe
        df = build_nmr_df(
                atoms,
                fname,
                isotopes = isotopes,
                references = references,
                gradients = gradients,
                properties = properties,
                euler_convention = euler_convention,
                logger = logger,
        )
        
        essential_columns = ['labels', 'species', 'multiplicity', 'tags', 'file']
        # apply filters
        df = apply_df_filtering(
            df,
            include,
            exclude,
            query,
            essential_columns=essential_columns,
            logger=logger)

        # ----- atoms object manipulation -----
        


        # only keep the atoms that are in the dataframe (based on tag)
        atoms = atoms[np.isin(atoms.get_tags(), df['tags'].values)]

        # if the df is not empty, append it to the list
        if len(df) > 0:
            dfs.append(df)
            images.append(atoms)
            logger.info(FOOTER)
        # if the df is empty, raise warning and don't append
        else:
            logger.warning(f"No results found for {fname}.\n "
                "Try removing filters/checking the file contents.")
            
        
    if merge:
        # merge all dataframes into one
        dfs = [pd.concat(dfs, axis=0)]
    for i, df in enumerate(dfs):
        dfs[i] = sortdf(df, sortby, sort_order)
    # rename columns to include units for those that have units
    for df in dfs:
        df.rename(columns=units_rename, inplace=True)
    return dfs, images


def reload_as_molecular_crystal(atoms):
        '''
        If the atoms object is a molecular crystal, reload it with the correct
        connectivity.

        Args:
            atoms (ASE Atoms object): the atoms object to be reloaded.
        
        Returns:
            atoms (ASE Atoms object): the atoms object with the correct connectivity.
        '''
        # save initial order
        atoms.set_array('order_tag', np.arange(len(atoms)))
        # check if it's a molecular crystal
        elements = set(atoms.get_chemical_symbols())
        # Rough very basic check if it's organic:
        if 'C' in elements and 'H' in elements:
            # temporarily translate the atoms to the COM
            com = atoms.cell.T.dot([0.5,0.5,0.5]) - atoms.get_center_of_mass()
            atoms.translate(com)
            # let's assume this is an organic molecule/crystal
            # and try to reload the atoms object with the correct
            # connectivity:
            mols = Molecules.get(atoms)
            if len(mols) > 1:
                logger.debug('Found {} molecules'.format(len(mols)))
                temp = mols[0].subset(atoms, use_cell_indices=True)
                for mol in mols[1:]:
                    temp.extend(mol.subset(atoms, use_cell_indices=True))
                # restore original order
                temp = temp[temp.get_array('order_tag').argsort()]
                # restore original centering 
                temp.translate(-com)
                atoms =temp

        return atoms

def label_atoms(atoms: Atoms)->Atoms:
    if has_cif_labels(atoms):
        return atoms
    
    # Inform user of best practice RE CIF labels
    logger.debug(NO_CIF_LABEL_WARNING)
    
    if atoms.has('magresview_labels'):
        labels = atoms.get_array('magresview_labels')
    else:
        labels = MagresViewLabels.get(atoms, store_array=True)
    
    # note we must change datatype to allow more space!
    labels = np.array(labels, dtype='U25')

    # remove current labels:
    if atoms.has('labels'):
        atoms.set_array('labels', None)
    # add labels to atoms object
    atoms.set_array('labels', labels)
    return atoms


def tag_functional_groups(
        average_group:str,
        atoms:Atoms,
        vdw_scale:float=1.0,
        )-> Atoms:
    '''
    Average over groups of atoms based on the average_group string.
    See find_XHn_groups for more details.

    Args:
        average_group (str): string of comma-separated patterns to average over. e.g. 'CH3,CH2'
        atoms (Atoms): Atoms object
        vdw_scale (float): scaling factor for the van der Waals radii. Default is 1.0.

    Returns:
        Atoms
    '''
    if atoms.has('tags'):
        tags = atoms.get_tags()
    else:
        tags = np.arange(len(atoms))

    labels = atoms.get_array('labels')
    # make sure dtype of labels allows for enough characters
    labels = labels.astype('U25')
    XHn_groups = find_XHn_groups(atoms, average_group, tags= tags, vdw_scale=vdw_scale)
    logger.info(f"\nAveraging over functional groups: {average_group}")
    for ipat, pattern in enumerate(XHn_groups):
        # check if we found any that matched this pattern
        if len(pattern) == 0:
            logging.warn(f"No XHn groups found for pattern {average_group.split(',')[ipat]}")
            continue
        logger.debug(f"Found {len(pattern)} {average_group.split(',')[ipat]} groups")
        # get the indices of the atoms that matched this pattern
        # update the tags and labels accordingly
        for ig, group in enumerate(pattern):
            logger.debug(f"    Group {ig} contains: {np.unique(labels[group])}")
            # fix labels here as aggregate of those in group
            combined_label = ','.join(np.unique(labels[group]))
            # labels[group] = f'{ig}'#combined_label
            labels[group] = combined_label

            tags[group] = -(ipat+1)*1e5-ig
    # update atoms object with new labels
    # note we must change datatype to allow more space!
    atoms.set_array('labels', None)
    atoms.set_array('labels', labels, dtype='U25')
    # update atoms tags
    atoms.set_tags(tags)
    return atoms


def merge_tagged_sites(atoms_in:Atoms, merging_strategies:dict = {})->Atoms:
    '''
    Merge sites that are tagged with the same tag.

    Args:
        atoms (Atoms): Atoms object. Must have tags.
        merging_strategies (dict): dictionary of merging strategies. See merge_sites for more details.
    '''
    atoms = atoms_in.copy()
    # if there are no tags present, return the atoms object
    if not atoms.has('tags'):
        return atoms
    
    # now we need to apply the filters etc to the atoms object
    # first we need to merge sites with the same tag
    unique_tags, unique_counts = np.unique(atoms.get_tags(),
                                            return_counts=True)
    # groups with more than one atom
    multi_group_tags = unique_tags[unique_counts > 1]
    for tag in multi_group_tags:
        # where are these tags in the current tags?
        tag_idx = np.where(atoms.get_tags() == tag)[0]
        # merge the sites
        atoms = merge_sites(
            atoms,
            tag_idx,
            merging_strategies=merging_strategies,
            keep_all = False)
    
    # sort by tag
    atoms = atoms[np.argsort(atoms.get_tags())]

    return atoms


def build_nmr_df(
        atoms: Atoms,
        fname: str,
        isotopes: dict   = {},
        references: dict = {},
        gradients: dict  ={},
        properties: List[str] = ['efg', 'ms'],
        euler_convention: str ='zyz',
        logger: logging.Logger = logging.getLogger('cli'),
        ):
    '''
    Build the dataframe containing the NMR properties.

    Args:
        atoms (ASE Atoms object): the atoms object to be reloaded.
        fname (str): the filename of the file being processed.
        all_selections (AtomSelection): the AtomSelection object containing all selections.
        isotopes (dict): dictionary of isotopes to use for each element. e.g. {'H': 2, 'C': 13}
        references (dict): dictionary of shielding references for each element. e.g. {'H': 20.0, 'C': 100.0}
        gradients (dict): dictionary of gradients for each element. e.g. {'H': -1.0, 'C': -0.95} defaults to {} == -1 for all elements.
        average_group (str): string of comma-separated patterns to average over. e.g. 'CH3,CH2'
        properties (list): list of properties to extract. e.g. ['efg', 'ms']
        euler_convention (str): the euler convention to use for the EFG tensor. Options are 'zyz' or 'zxz'

    Returns:
        df (pandas DataFrame): the dataframe containing the NMR properties.
    '''
    elements = atoms.get_chemical_symbols()
    isotopelist = _get_isotope_list(elements, isotopes=isotopes, use_q_isotopes=False)
    species = [f'{iso}{el}' for el, iso in zip(elements, isotopelist)]
    labels = np.asarray(atoms.get_array('labels'), dtype='U25')
    tags = atoms.get_tags()

    df = pd.DataFrame({
                'indices': atoms.get_array('indices'),
                'original_index': np.arange(len(atoms)),
                'labels': labels,
                'species':species,
                'multiplicity': atoms.get_array('multiplicity'),
                'tags': tags,
                })

    # If there are MagresView-style labels, add them in
    if atoms.has('magresview_labels'):
        df.insert(2, 'MagresView_labels', atoms.get_array('magresview_labels'))

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
            logger.warning(f'No MS data found in {fname}\n'
                'Set argument `-p efg` if the file(s) only contains EFG data ')
            pass
        except:
            logger.warning('Failed to load MS data from .magres')
            raise
    if 'efg' in properties:
        try:
            efg_summary = pd.DataFrame(get_efg_summary(atoms, isotopes, euler_convention))
            df = df = pd.concat([df, efg_summary], axis=1)
        except RuntimeError:
            logger.warning(f'No EFG data found in {fname}\n'
                'Set argument `-p ms` if the file(s) only contains MS data ')
            pass
        except:
            logger.warning('Failed to load EFG data from .magres')
            raise

        
    ## how many sites do we have now?
    total_explicit_sites = df['multiplicity'].sum()
    logger.info(f'\nFound {int(total_explicit_sites)} total sites.')
    logger.info(f'Reduced to {len(df)} sites.')


    return df





        


def get_ms_summary(
        atoms: Atoms,
        euler_convention: str,
        references: Union[dict, None] = None,
        gradients: Union[dict, None] = None,
        ) -> pd.DataFrame:
    '''
    For an Atoms object with ms tensor arrays, return a summary of the tensors.

    Args:
        atoms (Atoms): the Atoms object
        euler_convention (str): the euler convention to use
        references (dict, optional): the reference tensors. Defaults to None. e.g. {'C': 100}
        gradients (dict, optional): the gradient tensors. Defaults to None. e.g. {'C': -1}

    Returns:
        pd.DataFrame: a dataframe with the summary of the ms tensors
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
    

def get_efg_summary(
        atoms: Atoms,
        isotopes: dict,
        euler_convention: str,
        ) -> pd.DataFrame:
    '''
    For an Atoms object with EFG tensor arrays, return a summary of the tensors.

    Args:
        atoms (Atoms): the Atoms object
        isotopes (dict): the isotopes to use for the quadrupolar constants
        euler_convention (str): the euler convention to use

    Returns:
        pd.DataFrame: a dataframe with the summary of the EFG tensors

    '''
    Vzz   = EFGVzz.get(atoms)
    # convert Vzz from au to V/m^2
    Vzz = Vzz * (Ha / Bohr) * 1e-1

    # For quadrupolar constants, isotopes become relevant. This means we need to create custom Property instances to
    # specify them. There are multiple ways to do so - check the docstrings for more details - but here we set them
    # by element. When nothing is specified it defaults to the most common NMR active isotope.
    qP = EFGQuadrupolarConstant(isotopes=isotopes)
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



def check_equivalent_sites_ms(atoms, tags, tolerance=1e-3):
    """
    Check if the sites with the same tags have the same MS isotropy to within a tolerance.
    
    Args:
        atoms (Atoms): the Atoms object
        tags (list): the tags to check
        tolerance (float, optional): the tolerance. Defaults to 1e-3.
        
    Returns:
        bool: True if the sites are equivalent, False otherwise
    
    """
    unique_sites, counts = np.unique(tags, return_counts=True)
    ms = MSIsotropy.get(atoms)
    # loop over unique sites that have more than equivalent site
    for i in unique_sites[counts > 1]:
        # get the indices of the equivalent sites
        idx = np.where(tags == i)[0]
        # check if the ms isotropy is the same to within the tolerance
        if not np.allclose(ms[idx], ms[idx[0]], atol=tolerance):
            return False
    return True










