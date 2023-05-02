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

"""Implementation of AtomsProperties that relate to labeling of systems"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from soprano.properties import AtomsProperty
from soprano.selection import AtomSelection
from soprano.utils import recursive_mol_label, has_cif_labels
from soprano.properties.linkage import Molecules, HydrogenBonds, Bonds
from ase.spacegroup import get_spacegroup
import warnings


class SiteLabels(AtomsProperty):

    """
    SiteLabels

    Compute a unique label for an atom belonging to a molecule by
    exploiting network topology. Atoms can have the same label, but only if
    they're fundamentally indistinguishable in the molecule's chemical context
    (for example, three hydrogen atoms on a CH3 group). The label describes
    the molecular network as traversed starting from the given site, with each
    pair of square brackets indicating the traversal of a further bond, and
    all paths kept to the shortest possible.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.
    |   sites (list[int]): the sites for which the label have to be calculated.
    |                      If not given, all of them are calculated.
    |   custom_symbol (str): if present, replace the usual chemical symbol
    |                        with a custom name for the site on which the
    |                        label is being calculated, to make it
    |                        distinguishable from equivalent elements.

    | Returns:
    |   site_labels (list[str]): A list of the computed site labels
    """

    default_name = "site_labels"
    default_params = {"force_recalc": False, "sites": None, "custom_symbol": None}

    @staticmethod
    def extract(s, force_recalc, sites, custom_symbol):

        # First, we need the molecules
        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)
        mols = s.info[Molecules.default_name]

        elems = s.get_chemical_symbols()

        if sites is None:
            sites = range(len(s))

        labels = []

        for s_i in sites:
            # Find the molecule that it belongs to
            s_mol = [m for m in mols if s_i in m.indices][0]
            s_elems = np.array(elems, dtype="S2")
            if custom_symbol is not None:
                s_elems[s_i] = custom_symbol
            # Grab the bonds
            bonds = s_mol.get_array(Bonds.default_name)
            # This is a necessary step since the bonds are not classified
            # by original structure index yet
            bonds = {a: bonds[i] for i, a in enumerate(s_mol.indices)}
            labels.append(recursive_mol_label(s_i, s_mol.indices, bonds, s_elems))

        return labels


class MoleculeSites(AtomsProperty):

    """
    MoleculeSites

    Assigns unique labels to atoms belonging to molecules by exploiting
    network topology. Atoms can have the same label, but only if they're
    fundamentally indistinguishable in the molecule's chemical context
    (for example, three hydrogen atoms on a CH3 group). The molecule will be
    described by a characteristic string and by a series of labels in the
    format [element]_[number]. These sites will be saved by default and can
    be used for better insight when carrying out other analysis.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.
    |   save_info (bool): if True, save the found molecular sites as part of
    |                     the Atoms object info. By default True.
    |   save_asarray (bool): if True the molecular site names are also saved
    |                        as an array of the molecule selection.

    | Returns:
    |   molecular_sites (dict): A dictionary containing info characterising
    |                           the molecule's chemical sites unequivocally.
    |                           These are a string representation of the
    |                           molecule itself and a dictionary linking
    |                           atomic indices (as found in the molecule in
    |                           AtomSelection form) to site labels.
    """

    default_name = "molecule_sites"
    default_params = {"force_recalc": False, "save_info": True, "save_asarray": False}

    @staticmethod
    def extract(s, force_recalc, save_info, save_asarray):

        # First, we need the molecules
        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        elems = s.get_chemical_symbols()

        mol_sites = []

        for mol_i, mol in enumerate(s.info[Molecules.default_name]):

            # For each atom we do a depth-first traversal of the network
            sites = {}
            bonds = mol.get_array(Bonds.default_name)
            # This is a necessary step since the bonds are not classified
            # by original structure index yet
            bonds = {a: bonds[i] for i, a in enumerate(mol.indices)}
            for a in mol.indices:
                sites[a] = recursive_mol_label(a, mol.indices, bonds, elems)

            # Now grab the unique sites and pick the name of the molecule
            site_names = sorted(list(set(sites.values())))
            site_dict = {"name": site_names[0]}
            # Now rename the sites
            elem_sites = {}
            for a in sites:
                s_i = site_names.index(sites[a])
                if elems[a] not in elem_sites:
                    elem_sites[elems[a]] = [s_i]
                elif s_i not in elem_sites[elems[a]]:
                    elem_sites[elems[a]].append(s_i)
                sites[a] = "{0}_{1}".format(
                    elems[a], elem_sites[elems[a]].index(s_i) + 1
                )

            site_dict["sites"] = sites
            mol_sites.append(site_dict)

            if save_asarray:
                arr = [sites[a] for a in mol.indices]
                mol.set_array(MoleculeSites.default_name, arr)

        if save_info:
            s.info[MoleculeSites.default_name] = mol_sites

        return mol_sites


class HydrogenBondTypes(AtomsProperty):

    """
    HydrogenBondTypes

    Assign MoleculeSites labels to atoms, then characterise existing hydrogen
    bonds based on them, and return a list of such bonds detected in a system.
    The bonds come in the form '{0}<{1},{2}>..{3}<{4}>', where {0} is the name
    of the molecule containing the hydrogen, {2} is the hydrogen, {1} the atom
    to which the hydrogen is bonded, {3} the name of the other molecule and
    {4} the atom to which the hydrogen is hydrogen bonded.

    | Parameters:
    |   force_recalc (bool): if True, always recalculate the molecules even if
    |                        already present.
    |   save_info (bool): if True, save the found hydrogen bond types as part
    |                     of the Atoms object info. By default True.

    | Returns:
    |   hydrogen_bond_types (list): A list containing info characterising the
    |                               hydrogen bonds present in the system in a
    |                               detailed way.
    """

    default_name = "hydrogen_bond_types"
    default_params = {
        "force_recalc": False,
        "save_info": True,
    }

    @staticmethod
    def extract(s, force_recalc, save_info):

        # First, we need the molecules
        if Molecules.default_name not in s.info or force_recalc:
            Molecules.get(s)

        # Then the hydrogen bonds
        if HydrogenBonds.default_name not in s.info or force_recalc:
            HydrogenBonds.get(s)

        # Finally the sites
        if MoleculeSites.default_name not in s.info or force_recalc:
            MoleculeSites.get(s)

        mols = s.info[Molecules.default_name]
        hbonds = s.info[HydrogenBonds.default_name]
        all_sites = s.info[MoleculeSites.default_name]

        hblabels = []
        for hbtype in hbonds:
            for hb in hbonds[hbtype]:
                A_i = hb["A"][0]
                H_i = hb["H"]
                B_i = hb["B"][0]

                # Check in which molecule they are
                AH_sites = None
                B_sites = None
                for m_i, m in enumerate(mols):
                    if A_i in m:
                        AH_sites = all_sites[m_i]
                    if B_i in m:
                        B_sites = all_sites[m_i]

                if AH_sites is None or B_sites is None:
                    raise RuntimeError("Invalid hydrogen bond detected")

                # Now build the proper definition
                hblabel = ("{0}<{1},{2}>" "..{3}<{4}>").format(
                    AH_sites["name"],
                    AH_sites["sites"][A_i],
                    AH_sites["sites"][H_i],
                    B_sites["name"],
                    B_sites["sites"][B_i],
                )
                hblabels.append(hblabel)

        return sorted(hblabels)


class CarbonHybridationState(AtomsProperty):

    """
    CarbonHybridationState

    Returns an integer indicating the hybridation state of any given carbon
    atom in a system, 1, 2, or 3 for sp1, sp2 or sp3, obtained by counting its
    bonds. For non-carbon atoms or carbon atoms that can't be identified
    returns 0.

    | Parameters:
    |   bonds ([tuple]): if present, use this list of bonds instead of
    |                    recalculating them (must be formatted like the output
    |                    of the linkage.Bonds property).
    |   save_info (bool): if True, save the found hybridation states as an
    |                     array in the Atoms object. By default True.

    | Returns:
    |   hybrid ([int]): list of integers encoding the hybridation states

    """

    default_name = "carbon_hybridation_state"
    default_params = {"bonds": None, "save_info": True}

    @staticmethod
    def extract(s, bonds, save_info):

        elems = np.array(s.get_chemical_symbols())
        hybrid = np.zeros(len(elems)).astype(int)

        if (elems == "C").any():
            # Only do this if there is any carbon...

            if bonds is None:
                # Recalculate bonds
                bonds = Bonds.get(s)

            C_i = set(list(np.where(elems == "C")[0]))

            for b in bonds:
                C_b = C_i.intersection(b[:2])
                hybrid[list(C_b)] += 1

            # Now go from number of bonds to hybridation state
            hybrid[list(C_i)] = np.where(
                (hybrid[list(C_i)] > 1) * (hybrid[list(C_i)] <= 4),
                hybrid[list(C_i)] - 1,
                0,
            )

        if save_info:
            s.set_array(CarbonHybridationState.default_name, hybrid)

        return hybrid

class UniqueSites(AtomsProperty):

    """
    UniqueSites

    Assigns unique labels to atoms based on their position in the system and the
    allowed symmetry operations.

    Basically a wrapper around the ASE functionality.
    
    Atoms can have the same label, but only if they're
    fundamentally indistinguishable. These sites will be saved by default and can
    be used for better insight when carrying out other analysis.

    | Parameters:
    |   save_info (bool): if True, save the found unique sites as part of
    |                     the Atoms object info. By default True.
    |   symprec (float): the tolerance for symmetry operations. Default: 1e-4

    | Returns:
    |   unique ([int]): list of integers 
    """

    default_name = "unique_sites"
    default_params = {"force_recalc": False, "save_info": True, "symprec": 1e-4}

    @staticmethod
    def extract(s, force_recalc, save_info, symprec, override_cif = False):
        #-------------------------------------------------------------------#
        # symmetry reduction:
        # first we need the spacegroup
        sg = get_spacegroup(s)
        # tag each symmetry group of equivaluent sites with a unique integer label
        tags = sg.tag_sites(s.get_scaled_positions(), symprec=symprec)

        unique_tags, uniqueinds = np.unique(tags, return_index=True)


        # -- check to make sure that no two elements have the same tag --#
        # unique elements 
        elems = set(s.get_chemical_symbols())
        # indices for each element
        element_indices = [AtomSelection.from_element(s, el).indices for el in elems]
        # tags for each element
        element_tags = [tags[inds] for inds in element_indices]
        # unique tags for each element
        element_unique_tags = [set(el_tags) for el_tags in element_tags]
        # check that no two different elements have the same tag
        if len(elems) > 1 and set.intersection(*element_unique_tags) != set():
            raise ValueError("Some elements have the same 'unique-by-symmetry' tag -- this is not allowed.")
        #-------------------------------------------------------------------#
        # CIF label-based reduction
        #-------------------------------------------------------------------#
        if has_cif_labels(s):
            from collections import OrderedDict
            ciflabels = s.get_array('labels')
            _, unique_cif_inds, unique_cif_tags = np.unique(ciflabels, return_index=True, return_inverse=True)
            # sort
            unique_cif_inds = np.sort(unique_cif_inds)
            unique_cif_labels = ciflabels[unique_cif_inds]
            unique_cif_tags = unique_cif_tags[unique_cif_inds]
            cif_tags = np.array([np.where(lab == unique_cif_labels)[0][0] for lab in ciflabels])
            
            offset = max(tags) - max(cif_tags)
            if len(uniqueinds) != len(unique_cif_tags) or set(tags) != set(cif_tags):
                all_matched = False
            else:
                all_matched = (tags == (cif_tags+offset)).all()
            if not all_matched:
                mismatch_locations = np.where(tags != (cif_tags+offset))[0]
                warnings.warn("The symmetry-reduced sites don't match the CIF labels!"
                "Manually check that the symmetry reduction is working as expected."
                f"\nThe mismatched sites are: {ciflabels[mismatch_locations]} at {mismatch_locations}"
                )
                
                if not override_cif:
                    warnings.warn(
                        "\nProceeding with the CIF label reduction rather than the symmetry reduction.")
                    tags = cif_tags
                else:
                    warnings.warn(
                        "\nProceeding with the symmetry reduction rather than the CIF label reduction."
                        "\nThis is because the override_cif flag was set to True.")
                    tags = cif_tags


        if save_info:
            s.info[UniqueSites.default_name] = tags

        return tags

class MagresViewLabels(AtomsProperty):

    """
    MagresViewLabels

    Compute labels for a structure following the MagresView convention.

    | Parameters:
    |   save_asarray (bool): if True the magresview site labels are also saved
    |                        as an array. By default False.
    
    | Returns:
    |   magresview_labels (list[str]): A list of the computed site labels
    """

    default_name = "magresview_labels"
    default_params = {"save_asarray": False}

    @staticmethod
    def extract(s, save_asarray):
        # Magres labels
        symbs = np.array(s.get_chemical_symbols())
        elems = set(symbs)
        mlabs = [""] * len(s)
        for e in elems:
            e_i = np.where(symbs == e)[0]
            for i, j in enumerate(e_i):
                mlabs[j] = "{0}_{1}".format(e, i + 1)
        if save_asarray:
            # convert to numpy array with correct dtype
            mlabs = np.array(mlabs, dtype="U25")

            s.set_array(MagresViewLabels.default_name, mlabs)

        return mlabs