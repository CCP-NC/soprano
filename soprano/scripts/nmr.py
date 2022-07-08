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
'''

__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "July 08, 2022"


import click
import numpy as np
import re
import os
import sys
import re
from ase import io
from ase.visualize import view as aseview
from soprano.properties.labeling import UniqueSites
from soprano.properties.nmr import *
from soprano.data.nmr import _el_iso, _get_isotope_list
from soprano.selection import AtomSelection
from soprano.utils import has_cif_labels
import pandas as pd
import warnings
HEADER = '''
##########################################
#  Extracting NMR info from magres file  #
'''
FOOTER = '''
# End of NMR info extraction            #
##########################################
'''
# TODO: write guide for this on website...
NO_CIF_LABEL_WARNING = '''
## Protip: ##
This magres file doesn't seem to have CIF-stlye labels.
Using these is considered a good idea, but it's not required.
You can export these automatically from a cif file using 
cif2cell. e.g. for CASTEP:

cif2cell mystructure.cif --export-cif-labels -p castep

'''
implemented_properties = ['ms', 'efg']
def isotope_selection(ctx, parameter, isotope_string):
    """Parse the isotope string.
    Args:
        ctx: click context
        parameter: click parameter
        isotope_string (str): The isotopes specification, in the form ``"2H,15N" for deuterium and 15N``.
    Returns:
        dict: The isotope for each element specified. Formatted as::
            {Element: Isotope}.
    
    """
    if isotope_string == '':
        return {}
    isotope_dict = {}
    for sym in isotope_string.split(","):
        try:
            el, isotope = _el_iso(sym)
        except Exception as e:
            raise e
        isotope_dict[el] = isotope
    return isotope_dict
def keyvalue_parser(ctx, parameter, value):
    """Parse strings in the form 'C:1,H:2' into a dictionary.
        Also accept = as the separator between key and value.
        e.g. the MS shift reference and gradient strings.
        If the value is a single float, that will returned instead of a dict.
    Args:
        ctx: click context
        parameter: click parameter
        value (str): The references specification, in the form ``"C:100,H:123"``.
                     If value is a single float, that will returned instead of a dict.
    Returns:
        dict: The values for each key specified. Formatted as::
            {key: value}.
    """
    try:
        return float(value)
    except ValueError:
        pass

    keyvaluedict = {}
    if value != '':
        for sym in re.split(',', value):
            try:
                el, reference = re.split(":|=", sym)
                keyvaluedict[el] = float(reference)
            except Exception as e:
                raise e
    return keyvaluedict

@click.command()

# one of more files
@click.argument('files',
                nargs=-1,
                type=click.Path(exists=True),
                required=True)
# option to select a subset of atoms
@click.option('--select',
                '-s',
                'selection',
                type=str,
                default=None,
                help='Selection string of sites include. e.g. \n'
                '``-s C`` for only and all carbon atoms,\n'
                '``-s C.1-3,H.1.2`` for carbons 1,2,3 and hydrogens 1 and 2,\n'
                '``-s C1,H1a,H1b`` for any sites with the labels C1, H1a and H1b.'
                )
@click.option('--output',
            '-o',
            type=str,
            default=None,
            help='Output file name. If not specified, output is printed to stdout.')
@click.option('--output-format',
            '-f',
            default=None,
            type=click.Choice(['csv', 'json']),
            help='Output file format. '
            'If not specified, the format is guessed from output filename extension.')
# merge output files
@click.option('--merge',
            '-m',
            is_flag=True,
            default=False,
            help="If present, merge all files into a single output file.")

# Option to specify the isotopes to use
@click.option('--isotopes',
            '-i',
            callback = isotope_selection,
            # type=isotope_selection,
            default='',
            metavar = 'ISOTOPES',
            help='Isotopes specification (e.g. ``-i 13C`` for carbon 13 '
        '``-i 2H,15N`` for deuterium and 15N). '
        'When nothing is specified it defaults to the most common NMR active isotope.')         
# flag option to reduce by symmetry
@click.option('--reduce',
            '-r',
            is_flag=True,
            default=False,
            help="Reduce the output by symmetry-equivalent sites. "
        "The merged equivalent sites are averaged. "
        "If there are CIF-style labels present, then these override the symmetry-grouping in "
        "case of a clash. "
        "Note that this doesn't take into account magnetic symmetry!")
# symprec flag
@click.option('--symprec',
            type=click.FLOAT,
            default=1e-4,
            help="Symmetry precision for symmetry reduction. "
        "Defaults to 1e-4.")
# option to specify group_pattern for averaging
@click.option('--average-group',
            '-g',
            type=str,
            default=None,
            help="Group pattern for averaging. "
            "Currently only works for XHn groups such as CH3, CH2, NH2 etc. "
            "You can specify several, comma separated as in ``-g CH3,CH2,NH2``. "
            "If not specified, no averaging is performed.")
# combine rule
@click.option('--combine-rule',
            default='mean',
            type=click.Choice(['mean', 'first']),
            help="How to combine the data from equivalent sites. "
        "``mean`` is the default, which averages the data. "
        "``first`` Takes the first item from each group of equivalent sites. "
        "Special handling for labels, indices, tags and multiplicity. "
        "Set verbose to True to see what rules have been used.")
#  what to extract/analyse
@click.option('--properties',
            '-p',
            type=click.Choice(implemented_properties),
            default=['efg', 'ms'],
            multiple=True,
            help="Properties for which to extract and summarise e.g. ``-p ms.`` "
            "They can be combined by using the flag multiple times: ``-p ms -p efg.`` "
            "Defaults to both ms and efg.")
# optional argument for the precision of the output
@click.option('--precision',
            type=click.INT,
            default=3,
            help="Precision of the output (number of decimal places). Defaults to 3.")
# choose between Euler angle conventions 'zyz' or 'zxz'
@click.option('--euler',
            'euler_convention',
            type=click.Choice(['zyz', 'zxz']),
            default='zyz',
            help="Convention for Euler angles. Defaults to ``zyz``.")
# sort by df column
@click.option('--sortby',
            type=str,
            default=None,
            help="Sort by column. Defaults to sorting by site number. "
            "It can be any column in the output. "
            "For example ``--sortby EFG_Vzz``")
@click.option('--sort-order',
            type=click.Choice(['ascending', 'descending']),
            default='ascending',
            help="Sort order. Defaults to ascending.")
# dictionary option to specify reference for each element
@click.option('--references',
            callback = keyvalue_parser,
            default='',
            help="Reference shielding for each element (in ppm). "
            "The format is ``--references C:170,H:123``. "
            "If the value is a single float, that reference will be used for all sites (not recommended!). ")
@click.option('--gradients',
            callback = keyvalue_parser,
            default='',
            help="Reference shielding gradients for each element. "
            "Defaults to -1 for all elements. Set it like this: "
            "``--gradients H:-1,C:-0.97``. "
            "If the value is a single float, that gradient will be used for all sites (not recommended!). "
            )
# todo: have an option to set a file/env variable for the references... 

# flag to view
@click.option('--view',
            is_flag=True,
            default=False,
            help="If present, view the structure(s) with the ASE GUI."
            "Note that the ASE GUI can color the sites according to their tags. "
            "This can be used to see what sites were tagged as equivalent.")


# verbose flag
@click.option('--verbose',
            '-v',
            is_flag=True,
            default=False,
            help="If present, print more information.")

      


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
        view,
        verbose):
    """
    e.g. soprano nmr seedname.magres

    Processes .magres file(s) containing NMR-related properties
    and prints a summary. It defaults to printing all NMR properties
    present in the file for all the atoms. 
    
    See the below arguments for how to extract specific information.
    """
    
    # set pandas print precision
    pd.set_option('precision', precision)
    
    nfiles = len(files)
    dfs = []
    images = []
    # loop over files
    for fname in files:
        if verbose:
            click.echo(HEADER.format(fname))
            click.secho(f'{fname}', fg='blue')
            click.echo('\nExtracting properties:')
            click.secho(f'{properties}', fg='blue')
            


        # try to read in the file:
        try:
            atoms = io.read(fname)
        except IOError:
            click.secho("Error: file {0} not read, skipping".format(fname), fg='red')
            continue
        
        # Do they actually have any magres data?
        if not any([atoms.has(k) for k in properties]):
            click.secho("Warning, file {0} has no {1} data extract, skipping {1}".format(fname, ' '.join(properties)), fg='red')
            continue
        if verbose:
            if not has_cif_labels(atoms):
                click.secho(NO_CIF_LABEL_WARNING, fg='blue')
        all_selections = AtomSelection.all(atoms)
        # create new array for multiplicity
        multiplicity = np.ones(len(atoms))
        atoms.set_array('multiplicity', multiplicity)

        # note we must change datatype to allow more space!
        labels = atoms.get_array('labels').astype('U25')
        # reduce by symmetry?
        tags = np.arange(len(atoms))


        if reduce:
            if verbose:
                click.echo('\nTagging equivalent sites')

            tags = UniqueSites.get(atoms, symprec=symprec)
            if verbose:
                unique_sites, unique_site_idx = np.unique(tags, return_index=True)
                click.echo('    This leaves {0} unique sites'.format(len(unique_sites)))
                click.echo(f'    The unique site labels are: {labels[unique_site_idx]}')
                


        if average_group:
            XHn_groups = find_XHn_groups(atoms, average_group, tags= tags, vdw_scale=1.0)
            for ipat, pattern in enumerate(XHn_groups):
                # check if we found any that matched this pattern
                if len(pattern) == 0:
                    click.echo(f"No XHn groups found for pattern {average_group.split(',')[ipat]}")
                    continue
                
                if verbose:
                    click.echo(f"Found {len(pattern)} {average_group.split(',')[ipat]} groups")
                # get the indices of the atoms that matched this pattern
                # update the tags and labels accordingly
                for ig, group in enumerate(pattern):
                    if verbose:
                        click.echo(f"    Group {ig} contains: {np.unique(labels[group])}")
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
            if verbose:
                click.echo(f'\nSelecting atoms based on selection string: {selection}')
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
        # Let's add a column for the file name -- useful to keep track of 
        # which file the data came from if merging multiple files.
        df['file'] = fname
        if 'ms' in properties:
            try:
                ms_summary = pd.DataFrame(get_ms_summary(atoms, euler_convention, references, gradients))
                if not references:
                    # drop shift column if no references are given
                    ms_summary.drop(columns=['MS_shift/ppm'], inplace=True)

                df = pd.concat([df, ms_summary], axis=1)
            except KeyError:
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
            except KeyError:
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

            # Special rule for labels:
            # if there are multiple labels, combine them with a comma
            # if there is only one unique label, keep it as is
            def agg_labels(df):
                labels = df['labels']
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    return ','.join(unique_labels)
                else:
                    return unique_labels[0]
            aggrules['labels'] = set
            aggrules['multiplicity'] = 'count'

            if verbose:
                click.echo('\nAveraging over sites with the same tag')
                click.echo(f'   We apply the following rules to each column:\n {aggrules}')
            # apply group averaging
            grouped = df.groupby('tags')
            df = grouped.agg(aggrules).reset_index()
            # fix the labels print formatting            
            df['labels'] = df['labels'].apply(lambda x: ','.join(x))
            
        
        

        if verbose:
            total_explicit_sites = df['multiplicity'].sum()
            click.echo(f'\nFound {total_explicit_sites} total sites.')
            if average_group or reduce:
                click.echo(f'    -> reduced to {len(df)} sites after averaging equivalent ones')


        dfs.append(df)
        images.append(atoms)
        if verbose:
            click.echo(FOOTER)
        
    if view:
        # If it's organic molecule/structure
        # we usaully want to reload with molecular units intact
        from soprano.properties.linkage import Molecules
        for i, atoms in enumerate(images):
            elements = set(atoms.get_chemical_symbols())
            # Rough very basic check if it's organic:
            if 'C' in elements and 'H' in elements:
                # let's assume this is an organic molecule/crystal
                # and try to reload the atoms object with the correct
                # connectivity:
                mols = Molecules.get(atoms)
                temp = mols[0].subset(atoms, use_cell_indices=True)
                for mol in mols[1:]:
                    temp.extend(mol.subset(atoms, use_cell_indices=True))
                images[i] =temp

        aseview(images)
    if merge:
        # merge all dataframes into one
        dfs = [pd.concat(dfs, axis=0)]
    for i, df in enumerate(dfs):
        dfs[i] = sortdf(df, sortby, sort_order)
    
    # write to file(s)
    print_results(dfs, output, output_format, verbose)
        
def print_results(dfs, output, output_format, verbose):
    nframes = len(dfs)
    if output:
        for i, df in enumerate(dfs):
            if nframes > 1:
                # then we want to write out 
                # each dataframe to a separate file
                # so let's prefix the filename
                magrespath = df['file'].iloc[0]
                prefix='_'.join(os.path.splitext(magrespath)[0].split('/')) + '-'
            else:
                prefix = ''
            fname = prefix + output
            if verbose:
                click.echo(f'Writing output to {fname}')
            if not output_format:
                # try to guess format from extension
                output_format = os.path.splitext(fname)[1][1:]

            if output_format == 'csv':
                df.to_csv(fname, index=True)
            elif output_format == 'json':
                df.to_json(fname)
            else:
                raise ValueError(f'Unknown output format: {output_format}')
    else:
        # We write to stdout

        # if there's only one dataframe
        # but it contains output from mutliple magres files
        # then we need that file column
        if nframes ==1 and dfs[0]['file'].nunique() > 1:
            # then there's only one dataframe
            # and we want the filename for each row
            click.echo(dfs[0])
        else:
            # we can drop the file column
            for df in dfs:
                fname = df['file'].iloc[0]
                click.echo(f"\n\nExtracted data from: {fname}")
                df.drop('file', axis=1, inplace=True)
                click.echo(df)


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
    alpha, beta, gamma = np.array([q.euler_angles(mode=euler_convention)*180/np.pi for q in quat]).T
    ms_summary = {
            'MS_shielding/ppm': iso,
            'MS_shift/ppm': shift,
            'MS_anisotropy/ppm': aniso,
            'MS_reduced_anisotropy/ppm': red_aniso,
            'MS_asymmetry': asymm,
            'MS_span/ppm': span,
            'MS_skew': skew,
            'MS_alpha/deg': alpha,
            'MS_beta/deg': beta,
            'MS_gamma/deg': gamma
            }
    return ms_summary
    

def get_efg_summary(atoms, isotopes, euler_convention):
    '''
    For an Atoms object with EFG tensor arrays, return a summary of the tensors.
    '''
    Vzz   = EFGVzz.get(atoms)

    # For quadrupolar constants, isotopes become relevant. This means we need to create custom Property instances to
    # specify them. There are multiple ways to do so - check the docstrings for more details - but here we set them
    # by element. When nothing is specified it defaults to the most common NMR active isotope.
    qP = EFGQuadrupolarConstant(isotopes=isotopes) # Deuterated; for the others use the default
    qC = qP(atoms)/1e6 # To MHz
    
    # asymmetry
    eta = EFGAsymmetry.get(atoms)

    # quaternion
    quat = EFGQuaternion.get(atoms)
    alpha, beta, gamma = np.array([q.euler_angles(mode=euler_convention)*180/np.pi for q in quat]).T
    efg_summary = {
                'EFG_Vzz/au': Vzz,
                'EFG_quadrupolar_constant/MHz': qC,
                'EFG_asymmetry': eta,
                'EFG_alpha/deg': alpha,
                'EFG_beta/deg': beta,
                'EFG_gamma/deg': gamma
                }
    return efg_summary

def sortdf(df, sortby, sort_order):
    ''' sort df by column, return new df'''

    if sortby:
        if sortby in df.columns: 
            ascending = sort_order == 'ascending'
            if sortby == 'labels':
                isalpha = df[sortby].str.isalpha()
                if all(isalpha):
                    # no cif style labels, so sort by alphabetical order
                    df = df.sort_values(by=sortby, ascending=ascending)
                elif any(isalpha):
                    # tricky -- we have a mix of cif style labels and non-cif style labels
                    # so we need to sort by the cif style labels first, then sort by the non-cif style labels
                    # this is a bit of a hack, but it works
                    raise NotImplementedError('Sorting by labels is not implemented'
                            'for a mix of cif and non-cif -style labels')
                else:
                    # all cif stlye
                    # some care is needed to sort by labels in a natural way
                    df[['_str', '_int']] = df[sortby].str.extract(r'([a-zA-Z]*)(\d*)')
                    df['_int'] = df['_int'].astype(int)

                    df = df.sort_values(by=['_str', '_int'], ascending=ascending).drop(['_int', '_str'], axis=1)    
            else:
                df = df.sort_values(by=sortby, ascending=ascending)
        else:
            raise ValueError(f'{sortby} not found in summary columns names')
    return df




def find_XHn_groups(atoms, pattern_string, tags=None, vdw_scale=1.0):
    """Find groups of atoms based on a functional group pattern. 
    The pattern is a string such as CH3 or CH2. 
    It must contain an element symbol, H and the number of H atoms


    | Args:
    |   atoms (ase.Atoms): Atoms object on which to perform selection
    |   pattern_string (str): functional group pattern e.g. 'CH3'
    |                        for a methyl group. Assumes the group is
    |                        the thing(s) connected to the first atom.
    |                        They can be combined, comma separated.
    |                        TODO: add SMILES/SMARTS support?
    |   vdw_scale (float): scale factor for vdw radius (used for bond searching)
    """
    from soprano.properties.linkage import Bonds
    
    if tags is None:
        tags = np.arange(len(atoms))

    bcalc = Bonds(vdw_scale=vdw_scale, return_matrix=True)
    bonds, bmat = bcalc(atoms)
    all_groups = []
    for group_pattern in pattern_string.split(','):
        # split into central element and number of H atoms
        if 'H' not in group_pattern:
            raise ValueError(f'{group_pattern} is not a valid group pattern '
                             '(must contain an element symbol, H, and the number of H atoms. e.g. CH3)')
        X, n = group_pattern.split('H')
        n = int(n)
        # Find XHn groups
        symbs = np.array(atoms.get_chemical_symbols())
        hinds = np.where(symbs == "H")[0]
        groups = []
        xinds = np.where(symbs == X)[0]
        xinds = xinds[np.where(np.sum(bmat[xinds][:, hinds], axis=1) == n)[0]]
        # group_tags = np.ones((len(xinds), n), dtype=int)
        seen_tags = []
        for ix, xind in enumerate(xinds):
            group = list(np.where(bmat[xind][hinds] == 1)[0])
            assert len(group) == n
            match = []
            if len(seen_tags)>0:
                match = np.where((seen_tags == tags[group]).all(axis=1))[0]
            
            if len(match) == 1:
                # how to handle this?
                groups[match[0]] += group
            elif len(match) == 0:
                seen_tags.append(tags[group])
                groups.append(group)
            else:
                raise ValueError(f'Found multiple matches for {group_pattern}')
            
        all_groups.append(groups)

    return all_groups
