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

'''A collection of functions and options for the command line interface for soprano.'''


__author__ = "J. Kane Shenton"
__maintainer__ = "J. Kane Shenton"
__email__ = "kane.shenton@stfc.ac.uk"
__date__ = "August 10, 2022"



import click
from collections import OrderedDict, defaultdict
import re
from soprano.data.nmr import _el_iso
from soprano.utils import average_quaternions
import os
import numpy as np
from configparser import ConfigParser
from ase.visualize import view as aseview
from soprano.properties.linkage import Molecules



# join home and config file
home = os.path.expanduser('~')
# get default soprano config file:
DEFAULT_CFG = os.environ.get('SOPRANO_CONFIG', f'{home}/.soprano/config.ini')

#callback to load config file
def configure(ctx, param, filename):
    cfg = ConfigParser()
    cfg.read(filename)
    ctx.default_map = {}
    for sect in cfg.sections():
        command_path = sect.split('.')
        if command_path[0] != 'soprano':
            continue
        defaults = ctx.default_map
        for cmdname in command_path[1:]:
            defaults = defaults.setdefault(cmdname, {})
        defaults.update(cfg[sect])


# We'll need to rename the columns to include units before printing
UNITS = {
    "MS_shielding": "ppm",
    "MS_shift": "ppm",
    "MS_anisotropy": "ppm",
    "MS_reduced_anisotropy": "ppm",
    "MS_span": "ppm",
    "MS_alpha": "deg",
    "MS_beta": "deg",
    "MS_gamma": "deg",
    "EFG_Vzz": 'Vm^-2',
    "EFG_quadrupolar_constant": 'MHz',
    "EFG_alpha": "deg",
    "EFG_beta": "deg",
    "EFG_gamma": "deg",
    "EFG_NQR": "MHz",
    "D": "kHz",
    "alpha": "deg",
    "beta": "deg",
    "gamma": "deg",
}


# TODO: write guide for this on website...
NO_CIF_LABEL_WARNING = '''
## Protip: ##
This .magres file doesn't seem to have CIF-style labels.
Using these is considered a good idea, but is not required.
You can export these automatically from a cif file using
cif2cell. e.g. for CASTEP:

cif2cell mystructure.cif --export-cif-labels -p castep

'''

### PARSER HELPERS ###
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

def get_column_list(ctx, parameter, value):
    """Parse the column names string.
    TODO: Document this a bit better.
    Args:
        ctx: click context
        parameter: click parameter
        value (str): The column names, comma-separated.
                    Some shortcuts defined for MS_angles and EFG_angles.
    Returns:
        list: The column names specified.
    """
    if value == '' or value is None:
        return None
    # shortcuts for some column groups
    special_names = {
                    'minimal':
                        ['MS_shielding',
                        'MS_anisotropy',
                        'EFG_quadrupolar_constant',
                        'EFG_asymmetry'
                        ],
                    'MS_defaults':
                        ['MS_shielding',
                        'MS_anisotropy',
                        'MS_reduced_anisotropy',
                        'MS_asymmetry'],
                    'EFG_defaults':
                        ['EFG_Vzz',
                        'EFG_quadrupolar_constant',
                        'EFG_asymmetry'],
                    'MS_angles':
                        ['MS_alpha',
                        'MS_beta',
                        'MS_gamma'],
                    'EFG_angles':
                        ['EFG_alpha',
                        'EFG_beta',
                        'EFG_gamma',
                            ],
                    }
    special_names['default'] = special_names['MS_defaults'] + special_names['EFG_defaults']

    specified_columns = [c.strip() for c in value.split(',')]
    # replace special names
    for special_name, special_cols in special_names.items():
        if special_name in specified_columns:
            specified_columns.remove(special_name)
            specified_columns.extend(special_cols)
    # make sure no duplicates, preserving order
    specified_columns = list(OrderedDict.fromkeys(specified_columns))
    return specified_columns


### CLI OPTIONS ###
config = click.option(
    '-c', '--config',
    type         = click.Path(dir_okay=False),
    default      = DEFAULT_CFG,
    callback     = configure,
    is_eager     = True,
    expose_value = False,
    show_default = True,
    help         = 'Read option defaults from the specified INI file'
                    'If not set, first checks environment variable: '
                    '``SOPRANO_CONFIG`` and then ``~/.soprano/config.ini``',
)
selection_help = '''Selection string of sites include. e.g.
                ``-s C`` for only and all carbon atoms,
                ``-s C.1-3,H.1.2`` for carbons 1,2,3 and hydrogens 1 and 2,
                ``-s C1,H1a,H1b`` for any sites with the labels C1, H1a and H1b.'''
# option to select a subset of atoms
select = click.option('--select',
                '-s',
                'selection',
                type=str,
                default=None,
                help=selection_help,
                )
#  what to extract/analyse
nmrproperties = click.option('--properties',
                    '-p',
                    type=click.Choice(['efg', 'ms']),
                    default=['efg', 'ms'],
                    multiple=True,
                    help="Properties for which to extract and summarise e.g. ``-p ms.`` "
                    "They can be combined by using the flag multiple times: ``-p ms -p efg.`` "
                    "Defaults to both ms and efg.")
df_output = click.option('--output',
            '-o',
            type=click.Path(exists=False),
            default=None,
            help='Output file name. If not specified, output is printed to stdout.')
df_output_format = click.option('--output-format',
            '-f',
            default=None,
            type=click.Choice(['csv', 'json']),
            help='Output file format. '
            'If not specified, the format is guessed from output filename extension.')
# merge output files
df_merge = click.option('--merge',
            '-m',
            is_flag=True,
            default=False,
            help="If present, merge all files into a single output file.")

# Option to specify the isotopes to use
isotopes = click.option('--isotopes',
            '-i',
            callback = isotope_selection,
            default='',
            metavar = 'ISOTOPES',
            help='Isotopes specification (e.g. ``-i 13C`` for carbon 13 '
        '``-i 2H,15N`` for deuterium and 15N). '
        'When nothing is specified it defaults to the most common NMR active isotope.')
# flag option to reduce by symmetry
df_reduce = click.option('--reduce',
            '-r',
            is_flag=True,
            default=False,
            help="Reduce the output by symmetry-equivalent sites. "
        "The merged equivalent sites are combined according to ``--combine-rule``. "
        "To see the rules used for each column, set ``--verbose``. "
        "If there are CIF-style labels present, then these override the symmetry-grouping in "
        "case of a clash. "
        "Note that this doesn't take into account magnetic symmetry!")
# symprec flag
symprec = click.option('--symprec',
            type=click.FLOAT,
            default=1e-4,
            help="Symmetry precision for symmetry reduction. "
        "Defaults to 1e-4.")
# option to specify group_pattern for averaging
average_group = click.option('--average-group',
            '-g',
            type=str,
            default=None,
            help="Group pattern for averaging. "
            "Currently only works for XHn groups such as CH3, CH2, NH2 etc. "
            "You can specify several, comma separated as in ``-g CH3,CH2,NH2``. "
            "If not specified, no averaging is performed.")
# combine rule
df_combine_rule = click.option('--combine-rule',
            default='mean',
            type=click.Choice(['mean', 'first']),
            help="How to combine the data from equivalent sites. "
        "``mean`` is the default, which averages the data. "
        "``first`` Takes the first item from each group of equivalent sites. "
        "Special handling for labels, indices, tags and multiplicity. "
        "Set verbose to True to see what rules have been used.")

# optional argument for the precision of the output
precision = click.option('--precision',
            type=click.INT,
            default=3,
            help="Precision of the output (number of decimal places). Defaults to 3.")
# choose between Euler angle conventions 'zyz' or 'zxz'
euler = click.option('--euler',
            'euler_convention',
            type=click.Choice(['zyz', 'zxz']),
            default='zyz',
            help="Convention for Euler angles. Defaults to ``zyz``.")
# sort by df column
df_sortby = click.option('--sortby',
            type=str,
            default=None,
            help="Sort by column. Defaults to sorting by site number. "
            "It can be any column in the output. "
            "For example ``--sortby EFG_Vzz``")
df_sort_order = click.option('--sort-order',
            type=click.Choice(['ascending', 'descending']),
            default='ascending',
            help="Sort order. Defaults to ascending.")
# dictionary option to specify reference for each element
references = click.option('--references',
            callback = keyvalue_parser,
            default='',
            help="Reference shielding for each element (in ppm). "
            "The format is ``--references C:170,H:123``. "
            "If the value is a single float, that reference will be used for all sites (not recommended!). ")
gradients = click.option('--gradients',
            callback = keyvalue_parser,
            default='',
            help="Reference shielding gradients for each element. "
            "Defaults to -1 for all elements. Set it like this: "
            "``--gradients H:-1,C:-0.97``. "
            "If the value is a single float, that gradient will be used for all sites (not recommended!). "
            )
# todo: have an option to set a file/env variable for the references...
# flag to include certain columns only
df_include = click.option('--include',
            callback=get_column_list,
            default=None,
            help="Include only certain columns. "
            "The columns are specified as a comma-separated list. "
            "For example ``--include MS_shielding,EFG_Vzz``. "
            "Defaults to all columns.")
# flag to exclude certain columns
df_exclude = click.option('--exclude',
            callback=get_column_list,
            default=None,
            help="Exclude certain columns. "
            "The columns are specified as a comma-separated list. "
            "For example ``--exclude MS_alpha,MS_beta,MS_gamma``. "
            "Defaults to None.")
# flag to filter results
df_query = click.option('--query',
            type=str,
            default=None,
            help="Filter results based on query. "
            "The filter is specified as a pandas query. "
            "Note that you must enclose the query in quotes! "
            "Refer to the column names without the units.  "
            "For example ``--query 'MS_shielding > 100'``. "
            "You can combine queries with ``and`` and ``or`` etc. "
            "e.g. ``--query 'MS_shielding > 100 and MS_shielding < 180'``. "
            "Defaults to #nofilter :).")

# flag to view
view = click.option('--view',
            is_flag=True,
            default=False,
            help="If present, view the structure(s) with the ASE GUI."
            "Note that the ASE GUI can color the sites according to their tags. "
            "This can be used to see what sites were tagged as equivalent.")


# quiet flag
quiet = click.option('--quiet',
            '-q',
            is_flag=True,
            default=False,
            help="If present, log less information.")


# option to select a subset of atoms
dip_selection_i = click.option('--select_i',
                '-s_i',
                'selection_i',
                type=str,
                default=None,
                help=selection_help
                )
dip_selection_j = click.option('--select_j',
                '-s_j',
                'selection_j',
                type=str,
                default=None,
                help=selection_help
                )
#### Groups of CLI options
# options that apply to pandas dataframes
DF_OPTIONS = [
    df_output,
    df_output_format,
    df_merge,
    df_reduce,
    df_combine_rule,
    df_sortby,
    df_sort_order,
    df_include,
    df_exclude,
    df_query,
    ]

NMR_OPTIONS = [
    nmrproperties,
    isotopes,
    average_group,
    euler,
    references,
    gradients,
    select,
    ]

COMMON_OPTIONS = [
    config,
    quiet,
    view,
    symprec,
    precision,
    ]

DIP_OPTIONS = [
    isotopes,
    average_group,
    dip_selection_i,
    dip_selection_j,
    ]

NMREXTRACT_OPTIONS = COMMON_OPTIONS + NMR_OPTIONS + DF_OPTIONS
DIPOLAR_OPTIONS = COMMON_OPTIONS + DIP_OPTIONS + DF_OPTIONS
# function to add options to a subcommand
def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options


### FUNCTIONS ###
def viewimages(images):
    '''
    Use ASE GUI to view the images.

    If they contain C and H, we'll assume it's a molecular
    crystal and reload it as such.

    We must be careful to keep the same order of atoms.
    '''
    for i, atoms in enumerate(images):
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
            print('Found {} molecules'.format(len(mols)))
            temp = mols[0].subset(atoms, use_cell_indices=True)
            for mol in mols[1:]:
                temp.extend(mol.subset(atoms, use_cell_indices=True))
            # restore original order
            temp = temp[temp.get_array('order_tag').argsort()]
            # restore original centering
            temp.translate(-com)
            images[i] = temp

    aseview(images)


def print_results(dfs, output, output_format, verbose):
    nframes = len(dfs)
    # rename columns to include units for those that have units
    for df in dfs:
        df.rename(columns=units_rename, inplace=True)
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
        if nframes == 1 and dfs[0]['file'].nunique() > 1:
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


def units_rename(colname, units_dict=UNITS):
    for key, unit in units_dict.items():
        if key in colname:
            return f'{colname}/{unit}'
    # if no matches found, return original name
    return colname


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


def get_matching_cols(df, lst):
    """
    Get the columns of a dataframe that roughly match a list of strings.
    """
    return [col for col in df.columns if any(x in col for x in lst)]
def get_missing_cols(df, lst):
    """
    Get the items in list that don't match any of the columns of a dataframe
    """
    return [x for x in lst if all(x not in col for col in df.columns)]
def get_duplicates(seq):
    '''
    Returns dict {duplicate_value: [indices]} for duplicates in a list
    '''
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return dict([(key,locs) for key,locs in tally.items()
                            if len(locs)>1])
def average_quaternions_by_tags(quaternions, tags):
    '''
    For repeated tags, average the quaternions.
    Return the modified list of quaternions.
    '''
    if len(set(tags)) != len(tags):
        dupl_dict = get_duplicates(tags)
        for tag, idx in dupl_dict.items():
            quat_group = [quaternions[ig] for ig in idx]
            # compute group average quaternion
            quat_av = average_quaternions(quat_group)
            # update list of quaternions
            for ig in idx:
                quaternions[ig] = quat_av
    return quaternions


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