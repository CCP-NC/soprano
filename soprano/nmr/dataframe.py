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

"""Pandas and column helpers for NMR dataframes.

These are the non-Click helpers used by :mod:`soprano.nmr.extract` and the
command-line tools.  They are kept free of Click so that library code can
import them without pulling in the CLI.
"""

import logging
from collections import defaultdict
from typing import List, Optional

import pandas as pd

from soprano.utils import average_quaternions

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
    "EFG_Vzz": "Vm^-2",
    "EFG_quadrupolar_constant": "MHz",
    "EFG_alpha": "deg",
    "EFG_beta": "deg",
    "EFG_gamma": "deg",
    "EFG_NQR": "MHz",
    "D": "kHz",
    "alpha": "deg",
    "beta": "deg",
    "gamma": "deg",
}

# Function to expand aliases
def expand_aliases(input_list, alias_dict):
    if input_list is None:
        return None
    output_list = []
    for item in input_list:
        if item in alias_dict:
            output_list.extend(alias_dict[item])
        else:
            output_list.append(item)
    return output_list

def units_rename(colname, units_dict=UNITS):
    for key, unit in units_dict.items():
        if key in colname:
            return f"{colname}/{unit}"
    # if no matches found, return original name
    return colname

def apply_df_filtering(
    df: pd.DataFrame,
    include: List,
    exclude: List,
    query: str,
    essential_columns: List = [],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Inlcude/exclude columns and filter the dataframe using a pandas query.

    Args:
        df (pd.DataFrame): the dataframe to filter
        include (list): list of columns to include
        exclude (list): list of columns to exclude
        query (str): pandas query string to filter the dataframe
        essential_columns (list): list of columns that must be included
        logger (logging.Logger): logger to use

    Returns:
        pd.DataFrame: the filtered dataframe

    """

    # if no logger specified, use the default
    if not logger:
        logger = logging.getLogger(__name__)

    if query:
        # use pandas query to filter the dataframe
        logger.info(f"\nFiltering dataframe using query: {query}")
        df.query(query, inplace=True)
        logger.info(f"-----> Filtered to {len(df)} sites.")

    # what columns should we include/exclude?

    if include:
        # what columns should we include/exclude?
        specified_columns = [c for c in include if c not in essential_columns]
        logger.debug(f"\nIncluding only columns containing: {specified_columns}")
        columns_to_include = essential_columns + specified_columns
        missing_columns = get_missing_cols(df, columns_to_include)
        if len(missing_columns) > 0:
            logger.warning(
                f"These columns specified {missing_columns}"
                f" do not match any in the dataframe ({df.columns})"
            )
        columns_to_include = get_matching_cols(df, columns_to_include)
        df = df[columns_to_include].copy()
    if exclude:
        logger.debug(f"\nExcluding columns: {exclude}")
        # remove those that are already not in df
        specified_columns = get_matching_cols(df, exclude)
        df = df.drop(specified_columns, axis=1)
    # drop any that have only NaN values
    df = df.dropna(axis=1, how="all")
    return df

def sortdf(df, sortby, sort_order):
    """sort df by column, return new df"""

    if sortby:
        if sortby in df.columns:
            ascending = sort_order == "ascending"
            if sortby == "labels":
                isalpha = df[sortby].str.isalpha()
                if all(isalpha):
                    # no cif style labels, so sort by alphabetical order
                    df = df.sort_values(by=sortby, ascending=ascending)
                elif any(isalpha):
                    # tricky -- we have a mix of cif style labels and non-cif style labels
                    # so we need to sort by the cif style labels first, then sort by the non-cif style labels
                    # this is a bit of a hack, but it works
                    raise NotImplementedError(
                        "Sorting by labels is not implemented"
                        "for a mix of cif and non-cif -style labels"
                    )
                else:
                    # all cif stlye
                    # some care is needed to sort by labels in a natural way
                    df[["_str", "_int"]] = df[sortby].str.extract(r"([a-zA-Z]*)(\d*)")
                    df["_int"] = df["_int"].astype(int)

                    df = df.sort_values(by=["_str", "_int"], ascending=ascending).drop(
                        ["_int", "_str"], axis=1
                    )
            else:
                df = df.sort_values(by=sortby, ascending=ascending)
        else:
            raise ValueError(f"{sortby} not found in summary columns names")
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
    """
    Returns dict {duplicate_value: [indices]} for duplicates in a list
    """
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return dict([(key, locs) for key, locs in tally.items() if len(locs) > 1])

def average_quaternions_by_tags(quaternions, tags):
    """
    For repeated tags, average the quaternions.
    Return the modified list of quaternions.
    """
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
