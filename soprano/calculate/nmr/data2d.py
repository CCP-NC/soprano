"""2D NMR data extraction and peak generation."""

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from soprano.calculate.nmr.config import MARKER_INFO
from soprano.calculate.nmr.utils import (
    ContourData,
    Peak2D,
    calculate_distances,
    extract_indices,
    filter_pairs_by_distance,
    generate_contour_map,
    generate_peaks,
    get_atom_labels,
    get_pair_dipolar_couplings,
    get_pair_j_couplings,
    merge_peaks,
    prepare_species_labels,
    process_pairs,
    sort_peaks,
    validate_elements,
)
from soprano.data.nmr import _get_isotope_list
from soprano.nmr.utils import _dip_constant
from soprano.properties.nmr import DipolarRSSByAtom, MSIsotropy

class NMRData2D:
    '''
    Class to hold and extract 2D NMR data.
    '''
    def __init__(self,
                atoms: Optional[Atoms] = None,
                xelement: Optional[str] = None,
                yelement: Optional[str] = None,
                references: Optional[dict[str, float]] = None,
                gradients: Optional[dict[str, float]] = None,
                peaks: Optional[List[Peak2D]] = None,
                pairs: Optional[List[Tuple[int, int]]] = None,
                correlation_strengths: Optional[List[float]] = None,
                correlation_strength_metric: Optional[str] = None, # 'fixed','distance', 'dipolar', 'dipolar2', 'jcoupling', 'inversedistance', 'custom', 'dipolar_rss'
                rcut: Optional[float] = None,
                rss_cutoff: float = 5.0,
                rss_expand_j: str = 'periodic_images',
                reduce: bool = False,
                symprec: float = 1e-4,
                atoms_full: Optional[Atoms] = None,
                isotopes: Optional[dict[str, int]] = None,
                is_shift: Optional[bool] = None,
                include_quadrupolar: bool = False,
                yaxis_order: str = '1Q',
                x_axis_label: Optional[str] = None,
                y_axis_label: Optional[str] = None,
                average_group: str = "",
                ):

        self.atoms = atoms
        self.xelement = xelement
        # if yelement is not provided, set it to xelement
        self.yelement = yelement if yelement is not None else xelement
        self.references = references
        self.gradients = gradients
        self.peaks = peaks
        self.pairs = pairs

        # if neither atoms nor peaks are provided, raise an error
        if self.atoms is None and self.peaks is None:
            raise ValueError("Either atoms or peaks must be given.")

        # Reduce to unique sites if requested — mirrors the CLI --reduce flag.
        # For dipolar_rss the full-cell atoms are needed so the expand_j
        # expansion can find all equivalent neighbours; store them before
        # merging duplicate sites away.
        if reduce and self.atoms is not None:
            from soprano.nmr.extract import label_atoms, nmr_extract_atoms  # lazy import
            self.atoms = label_atoms(self.atoms)
            if atoms_full is None:
                atoms_full = self.atoms   # keep labeled full atoms for RSS
            _reduced, _reduce_map = nmr_extract_atoms(
                self.atoms.copy(), reduce=True, symprec=symprec,
                return_index_map=True,
            )
            if _reduced is not None:
                if self.pairs is not None:
                    remapped = [
                        (int(_reduce_map[p[0]]), int(_reduce_map[p[1]]))
                        for p in self.pairs
                    ]
                    if remapped != list(self.pairs):
                        warnings.warn(
                            "reduce=True: user-supplied pairs have been "
                            "remapped from full-cell to reduced "
                            f"(asymmetric-unit) indices: {list(self.pairs)}"
                            f" → {remapped}. Pass pairs in reduced-cell "
                            "indices to suppress this warning.",
                            stacklevel=2,
                        )
                    self.pairs = remapped
                self.atoms = _reduced


        # Either provide correlation strengths or calculate them based on the metric
        # If both are provided, the provided values will be used
        if correlation_strengths:
            if correlation_strength_metric:
                self.logger.warning("Both correlation_strengths and correlation_strength_metric are provided. Using correlation_strengths.")
            correlation_strength_metric = 'custom'


        # If neither are provided, set the metric to 'fixed'
        if correlation_strengths is None and correlation_strength_metric is None:
            correlation_strength_metric = 'fixed'

        self.correlation_strengths = correlation_strengths
        self.correlation_strength_metric = correlation_strength_metric

        self.rcut = rcut
        self.rss_cutoff = rss_cutoff
        self.rss_expand_j = rss_expand_j
        self.symprec = symprec
        self.atoms_full = atoms_full
        self.isotopes = isotopes if isotopes is not None else {}
        # is_shift is a boolean.  If undefined, it will be set to True if references are provided, False otherwise
        #  If defined, it will be used as is
        self.is_shift = is_shift if is_shift is not None else (self.references is not None)
        self.include_quadrupolar = include_quadrupolar
        self.yaxis_order = yaxis_order
        self.logger = logging.getLogger(__name__)

        self.correlation_unit = MARKER_INFO[self.correlation_strength_metric]['unit']
        self.correlation_label = MARKER_INFO[self.correlation_strength_metric]['label']
        self.correlation_fmt = MARKER_INFO[self.correlation_strength_metric]['fmt']

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.average_group = average_group

        # run the main method to extract the data
        self.get_peaks()

    def _average_group_peaks(self, peaks: list) -> list:
        """Merge peaks from the same functional group (e.g. CH₃) post-hoc.

        This method operates on
        peaks generated from the *original*, unmodified atoms object so that
        every coupling is evaluated at the true inter-atomic distance.

        Peaks that belong to the same functional group and are paired with the
        same counter-atom are replaced by a single representative peak:

        * **position** (x, y) — mean of the group members' positions,
        * **correlation_strength** — mean of the members' coupling values
          (the multiplicity weight accounts for group size separately),
        * **multiplicity** — sum of the members' multiplicities
          (equals the group size when starting from the default of 1),
        * **label** — comma-separated sorted concatenation of member labels,
          matching the CLI ``--average-group`` convention.

        Args:
            peaks: Peak list from :func:`generate_peaks` (before
                :func:`merge_peaks`).

        Returns:
            New peak list with functional-group members merged.
        """
        import re  # noqa: PLC0415
        from collections import defaultdict  # noqa: PLC0415
        from dataclasses import replace as _replace  # noqa: PLC0415

        from soprano.nmr.extract import find_XHn_groups, label_atoms  # noqa: PLC0415

        # --- build atom → group-id mapping ---------------------------------
        atoms = label_atoms(self.atoms.copy(), logger=self.logger)
        all_groups = find_XHn_groups(atoms, self.average_group)

        atom_to_group: dict = {}
        for ipat, pattern_groups in enumerate(all_groups):
            for igrp, group in enumerate(pattern_groups):
                gid = (ipat, igrp)
                for aidx in group:
                    atom_to_group[int(aidx)] = gid

        if not atom_to_group:
            self.logger.warning(
                f"average_group='{self.average_group}' matched no groups; "
                "returning peaks unchanged."
            )
            return peaks

        def gkey(idx: int):
            """Group-id if atom is in a group, else its own index as sentinel."""
            return atom_to_group.get(int(idx), int(idx))

        # --- bucket peaks by merged-pair key --------------------------------
        peer_map: dict = defaultdict(list)
        for peak in peaks:
            key = (gkey(peak.idx_x), gkey(peak.idx_y))
            peer_map[key].append(peak)

        # --- combine each bucket --------------------------------------------
        def _sort_key(label: str):
            m = re.search(r'\d+', label)
            return int(m.group()) if m else label

        result = []
        for group_peaks in peer_map.values():
            if len(group_peaks) == 1:
                result.append(group_peaks[0])
                continue

            avg_x = float(np.mean([p.x for p in group_peaks]))
            avg_y = float(np.mean([p.y for p in group_peaks]))
            total_mult = sum(p.multiplicity for p in group_peaks)
            # Use multiplicity-weighted mean so that the invariant
            # merged_strength × merged_multiplicity = Σ(sᵢ × mᵢ)
            # is preserved — consistent with merge_peaks.
            total_weight = sum(p.correlation_strength * p.multiplicity for p in group_peaks)
            avg_strength = total_weight / total_mult if total_mult != 0 else 0.0

            xlabels = sorted({p.xlabel for p in group_peaks}, key=_sort_key)
            ylabels = sorted({p.ylabel for p in group_peaks}, key=_sort_key)
            result.append(_replace(
                group_peaks[0],
                x=avg_x,
                y=avg_y,
                correlation_strength=avg_strength,
                xlabel=','.join(xlabels),
                ylabel=','.join(ylabels),
                multiplicity=total_mult,
            ))

        n_before, n_after = len(peaks), len(result)
        if n_before != n_after:
            self.logger.debug(
                f"average_group='{self.average_group}': "
                f"merged {n_before} → {n_after} peaks."
            )
        return result

    def get_peaks(self, merge_identical=True, should_sort_peaks=False, force_recompute=False):
        '''
        Get the correlation peaks.

        If self.peaks already exists, then we return them as is.
        If they don't exist, we make sure the required data is available 
        and then generate the peaks and merge if desired.

        Set force_recompute=True to discard any cached peaks and regenerate
        from the underlying atoms/pairs data.
        '''

        if force_recompute:
            self.peaks = None

        if self.peaks is not None:
            self.logger.debug("Cached peaks found. Returning without recomputing. Use force_recompute=True to regenerate.")
            return self.peaks

        if self.atoms is None:
            raise ValueError("Either atoms or peaks must be given.")

        # make sure all the data is there
        self.extract_data()
        labels = get_atom_labels(self.atoms, self.logger)

        multiplicities = (
            self.atoms.get_array('multiplicity')
            if self.atoms is not None and self.atoms.has('multiplicity')
            else None
        )
        self.peaks = generate_peaks(self.data, self.pairs, labels, self.correlation_strengths, self.yaxis_order, self.xelement, self.yelement, multiplicities=multiplicities)

        if self.average_group:
            self.peaks = self._average_group_peaks(self.peaks)

        if merge_identical:
            self.peaks = merge_peaks(self.peaks, corr_rel_tol=0.05)

        if should_sort_peaks:
            self.peaks = sort_peaks(self.peaks)

        return self.peaks

    def extract_data(self):
        validate_elements(self.atoms, self.xelement, self.yelement)
        self.idx_x, self.idx_y = extract_indices(self.atoms, self.xelement, self.yelement)
        isotopes = _get_isotope_list(self.atoms.get_chemical_symbols(), isotopes=self.isotopes, use_q_isotopes=False)
        self.xisotope = isotopes[self.idx_x[0]]
        self.yisotope = isotopes[self.idx_y[0]]
        self.xspecies = prepare_species_labels(self.xisotope, self.xelement)
        self.yspecies = prepare_species_labels(self.yisotope, self.yelement)

        self.get_axis_labels()
        self.get_pairs()
        if self.correlation_strength_metric is None:
            raise ValueError("correlation_strength_metric is not defined. Please provide a value for this parameter.")


        self.correlation_strengths = self.get_correlation_strengths()

        self.data = MSIsotropy.get(self.atoms, ref=self.references, grad=self.gradients)
        self.logger.debug(f'Indices of xelement in the atoms object: {self.idx_x}')
        self.logger.debug(f'Indices of yelement in the atoms object: {self.idx_y}')
        self.logger.debug(f'X species: {self.xspecies}')
        self.logger.debug(f'Y species: {self.yspecies}')
        self.logger.debug(f'X values: {self.data[self.idx_x]}')
        self.logger.debug(f'Y values: {self.data[self.idx_y]}')


    def get_correlation_strengths(self):

        # Check that self.atoms is not None unless the metric is 'custom' or 'fixed', which don't require atoms
        if self.atoms is None and self.correlation_strength_metric not in ('custom', 'fixed'):
            raise ValueError(f"atoms must be provided to calculate correlation strengths for metric '{self.correlation_strength_metric}'.")
        
        if self.pairs is None:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        if self.correlation_strength_metric == 'custom':
            self.logger.info("Using custom correlation strengths.")

            # make sure correlation_strengths is a list or an array
            if not isinstance(self.correlation_strengths, (list, np.ndarray)):
                raise TypeError("correlation_strengths must be a list or an array.")

            # if user provides a list of these, use it!
            # just check that it's the right length
            if len(self.correlation_strengths) != len(self.pairs):
                raise ValueError(f"Length of correlation_strengths ({len(self.correlation_strengths)}) does not match the number of pairs ({len(self.pairs)}).")
            correlation_strengths = self.correlation_strengths

        elif self.correlation_strength_metric == 'fixed':
            self.logger.info("Using fixed correlation strength.")
            # set the correlation strength to be the same for all pairs
            correlation_strengths = np.ones(len(self.pairs))

        elif self.correlation_strength_metric in ('dipolar', 'dipolar2'):
            self.logger.info(f"Using {self.correlation_label.lower()} as correlation strength.")
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")
            correlation_strengths = get_pair_dipolar_couplings(self.atoms, self.pairs, self.isotopes)
            if self.correlation_strength_metric == 'dipolar2':
                correlation_strengths = np.square(correlation_strengths)
        elif self.correlation_strength_metric == 'distance' or self.correlation_strength_metric == 'inversedistance':
            log_message = "Using minimum image convention {isinverse}distance as correlation strength."
            isinverse = ''

            # now we can use ASE get_distance to get the distances for each pair
            correlation_strengths = self.pair_distances
            if self.correlation_strength_metric == 'inversedistance':
                correlation_strengths = 1 / correlation_strengths
                isinverse = 'inverse '
            self.logger.info(log_message.format(isinverse=isinverse))


        elif self.correlation_strength_metric == 'jcoupling':
            self.logger.info("Using J-coupling as correlation strength.")
            correlation_strengths = get_pair_j_couplings(self.atoms, self.pairs, self.isotopes)
        elif self.correlation_strength_metric == 'dipolar_rss':
            self.logger.info(
                f"Using dipolar RSS (cutoff={self.rss_cutoff} Å, "
                f"expand_j='{self.rss_expand_j}') as correlation strength."
            )
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")

            if self.rss_expand_j != 'periodic_images' and self.atoms_full is not None:
                # When the structure has been reduced to the asymmetric unit
                # (reduce=True), equivalent sites have been merged away, so
                # expand_j='cif_labels'/'symmetry' would find nothing to expand
                # in self.atoms.  Instead, map pair indices to the full (unmerged)
                # atoms via CIF labels, then let DipolarRSSByAtom expand there.
                reduced_labels = get_atom_labels(self.atoms, self.logger)
                full_labels = get_atom_labels(self.atoms_full, self.logger)

                def _first_full_idx(label):
                    matches = np.where(full_labels == label)[0]
                    if len(matches) == 0:
                        raise ValueError(
                            f"Label '{label}' from reduced atoms not found in "
                            "atoms_full. Ensure atoms_full is labeled consistently."
                        )
                    return int(matches[0])

                correlation_strengths = np.array([
                    DipolarRSSByAtom.get(
                        self.atoms_full,
                        sel_i=[_first_full_idx(reduced_labels[i])],
                        sel_j=[_first_full_idx(reduced_labels[j])],
                        cutoff=self.rss_cutoff,
                        isotopes=self.isotopes,
                        expand_j=self.rss_expand_j,
                        symprec=self.symprec,
                    )[0]
                    for i, j in self.pairs
                ]) * 1e-3  # convert Hz → kHz to match MARKER_INFO and dipolar metric
            else:
                if self.rss_expand_j != 'periodic_images':
                    self.logger.warning(
                        f"rss_expand_j='{self.rss_expand_j}' requested but atoms_full "
                        "was not provided. If the structure has been reduced to the "
                        "asymmetric unit, the expansion will find no additional sites. "
                        "Pass atoms_full (the full unit-cell atoms) to NMRData2D, or "
                        "use reduce=True to let NMRData2D handle this automatically."
                    )
                correlation_strengths = np.array([
                    DipolarRSSByAtom.get(
                        self.atoms,
                        sel_i=[i],
                        sel_j=[j],
                        cutoff=self.rss_cutoff,
                        isotopes=self.isotopes,
                        expand_j=self.rss_expand_j,
                        symprec=self.symprec,
                    )[0]
                    for i, j in self.pairs
                ]) * 1e-3  # convert Hz → kHz to match MARKER_INFO and dipolar metric
        else:
            raise ValueError(f"Unknown correlation_strength_metric option: {self.correlation_strength_metric}")

        # Make sure correlation_strengths is an array
        correlation_strengths = np.array(correlation_strengths)

        self.logger.debug(f"correlation_strengths : {correlation_strengths}")
        # Log pair with smallest and largest correlation strength
        min_idx = np.argmin(np.abs(correlation_strengths))
        max_idx = np.argmax(np.abs(correlation_strengths))
        smallest_pair = self.pairs[min_idx]
        largest_pair = self.pairs[max_idx]

        # Labels for the smallest and largest pairs
        labels = get_atom_labels(self.atoms, self.logger)
        smallest_pair_labels = [labels[smallest_pair[0]], labels[smallest_pair[1]]]
        largest_pair_labels = [labels[largest_pair[0]], labels[largest_pair[1]]]

        self.logger.info(
            f"Pair with smallest (abs) {self.correlation_label}: "
            f"{smallest_pair_labels} ({correlation_strengths[min_idx]:.2f})"
        )
        self.logger.info(
            f"Pair with largest (abs) {self.correlation_label}: "
            f"{largest_pair_labels} ({correlation_strengths[max_idx]:.2f})"
        )

        return correlation_strengths

    def get_axis_labels(self):
        if self.is_shift:
            axis_label = r"$\delta$"
        else:
            axis_label = r"$\sigma$"

        if self.x_axis_label is None:
            self.x_axis_label = f'{self.xspecies} ' + axis_label + ' /ppm'
        if self.y_axis_label is None:
            if self.yaxis_order == '2Q':
                self.y_axis_label = f'{self.yspecies} ' + axis_label + r'$_{\mathrm{%s}}$' % self.yaxis_order + ' /ppm'
            else:
                self.y_axis_label = f'{self.yspecies} ' + axis_label + ' /ppm'


    def get_pairs(self):
        '''
        Get the pairs of x and y indices to plot

        self.idx_x and self.idx_y are the indices of the x and y elements in the atoms object

        This method will set the following attributes:
        self.pairs_el_idx: a list of tuples of the form (xindex, yindex)
        self.pairs: a list of tuples of the form (xindex, yindex)
        '''
        # Process pairs; idx_x/idx_y (unique element indices) are not overwritten here —
        # process_pairs expands them into per-pair form internally.
        self.pairs, self.pairs_el_idx, _, _ = process_pairs(self.idx_x, self.idx_y, self.pairs)

        # In a DQ/SQ experiment the diagonal peaks arise from two *distinct*
        # atoms of the same chemical shift — not from a spin correlated with
        # itself.  The literal self-pair (i, i) is unphysical: the inter-nuclear
        # distance is zero so the dipolar coupling diverges, and it contributes
        # nothing to the real spectrum.  Remove all self-pairs unconditionally
        # in DQ mode so that diagonal peaks emerge naturally from the physics.
        if self.yaxis_order == '2Q':
            n_before = len(self.pairs)
            valid = [(i, p) for i, p in enumerate(self.pairs) if p[0] != p[1]]
            if not valid:
                raise ValueError(
                    "No valid (non-self) pairs found for DQ/SQ spectrum. "
                    "Diagonal peaks in a DQ experiment come from distinct atoms "
                    "with the same shift — check that the structure contains "
                    "coupled pairs of the requested elements."
                )
            idxs, self.pairs = zip(*valid)
            self.pairs = list(self.pairs)
            self.pairs_el_idx = [self.pairs_el_idx[i] for i in idxs]
            n_removed = n_before - len(self.pairs)
            if n_removed:
                self.logger.debug(
                    f"Removed {n_removed} self-pair(s) (i==i) — unphysical in DQ/SQ."
                )

        # For distance-based metrics, self-pairs would cause division by zero;
        # after the DQ filter above they are already gone for 2Q mode.
        # For SQ/homonuclear mode we still want to flag them rather than silently
        # produce NaN couplings.
        if self.correlation_strength_metric != 'fixed':
            self_pairs = [p for p in self.pairs if p[0] == p[1]]
            if self_pairs:
                self.logger.warning(
                    f"{len(self_pairs)} self-pair(s) remain after filtering with "
                    f"metric='{self.correlation_strength_metric}'. "
                    "Distance/coupling for a self-pair is undefined. "
                    "Consider using --rcut or passing explicit pairs."
                )

        if len(self.pairs) == 0:
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        # Calculate distances
        # if pair_distances is not already calculated, calculate it
        if not hasattr(self, 'pair_distances'):
            self.pair_distances = calculate_distances(self.pairs, self.atoms)


        # Filter pairs by distance if rcut is specified
        if self.rcut:
            self.logger.info(f"Filtering out pairs that are further than {self.rcut} Å apart.")
            self.logger.info(f"Number of pairs before filtering: {len(self.pairs_el_idx)}")

            self.pairs, self.pairs_el_idx, self.pair_distances, _, _ = filter_pairs_by_distance(
                self.pairs, self.pairs_el_idx, self.pair_distances, self.rcut)

            self.logger.info(f"Number of pairs remaining: {len(self.pairs_el_idx)}")
            self.logger.debug(f"Pairs remaining: {self.pairs}")
            self.logger.debug(f"Pairs el indices remaining: {self.pairs_el_idx}")

    def to_dataframe(self, include_metadata=True):
        """
        Convert the NMR data to a pandas DataFrame.
        
        This method exports all peak data and optionally includes metadata.
        The resulting DataFrame can be saved to CSV or other formats for
        later use or sharing.
        
        Parameters
        ----------
        include_metadata : bool, optional
            If True, includes columns with experimental metadata such as
            elements, isotopes, references, etc. Default is True.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing peak positions (x, y), labels (xlabel, ylabel),
            correlation strengths, and optionally metadata.
        
        Raises
        ------
        ImportError
            If pandas is not installed.
        
        Examples
        --------
        >>> df = nmr_data.to_dataframe()
        >>> df.to_csv('nmr_peaks.csv', index=False)
        
        >>> # Load back from CSV
        >>> import pandas as pd
        >>> df = pd.read_csv('nmr_peaks.csv')
        """
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required to export data to DataFrame. "
                "Install it with: pip install pandas"
            )
        
        # Get peaks (this will generate them if needed)
        peaks = self.get_peaks()
        
        if not peaks:
            self.logger.warning("No peaks to export.")
            return pd.DataFrame()
        
        # Extract data from Peak2D namedtuples
        data = {
            'x': [peak.x for peak in peaks],
            'y': [peak.y for peak in peaks],
            'xlabel': [peak.xlabel for peak in peaks],
            'ylabel': [peak.ylabel for peak in peaks],
            'correlation_strength': [peak.correlation_strength for peak in peaks],
        }
        
        if include_metadata:
            # Add metadata columns (same value for all rows)
            data.update({
                'xelement': self.xelement,
                'yelement': self.yelement,
                'xisotope': self.xisotope if hasattr(self, 'xisotope') else None,
                'yisotope': self.yisotope if hasattr(self, 'yisotope') else None,
                'correlation_metric': self.correlation_strength_metric,
                'correlation_unit': self.correlation_unit,
                'correlation_label': self.correlation_label,
                'yaxis_order': self.yaxis_order,
                'is_shift': self.is_shift,
            })
            
            # Add references if they exist
            if self.references:
                data['xreference'] = self.references.get(self.xelement, None)
                data['yreference'] = self.references.get(self.yelement, None)
            
            # Add gradients if they exist
            if self.gradients:
                data['xgradient'] = self.gradients.get(self.xelement, None)
                data['ygradient'] = self.gradients.get(self.yelement, None)
            
            # Add rcut if it was used
            if self.rcut:
                data['rcut'] = self.rcut
        
        df = pd.DataFrame(data)
        
        self.logger.info(f"Exported {len(df)} peaks to DataFrame.")
        return df

    def get_contour_data(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        grid_max: Optional[float] = None,
        broadening_type: str = 'lorentzian',
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
    ) -> 'ContourData':
        """
        Compute (and cache) the 2D contour grid for this spectrum.

        The result is stored on ``self._contour_data`` and is reused as long
        as the same parameters are requested.  Pass any argument explicitly to
        force recomputation with different settings.

        Parameters
        ----------
        x_broadening : float, optional
            FWHM linewidth in the direct (x) dimension.  Defaults to 5 % of
            the x peak range.  Internally converted to HWHM (Lorentzian) or
            σ (Gaussian) as appropriate.
        y_broadening : float, optional
            FWHM linewidth in the indirect (y) dimension.  Same default logic
            as *x_broadening*.
        grid_max : float, optional
            If provided, scale the computed contour grid so that ``Z.max()``
            equals this value. Useful when exporting to external tools that
            assume a particular intensity magnitude.
        broadening_type : str
            ``'gaussian'`` (default) or ``'lorentzian'``.
        grid_size : int
            Number of points along each axis of the grid.  Default 150.
        xlims : tuple of float, optional
            Used **only** to compute the default 5 % broadening when
            *x_broadening* is *None*.  The grid itself always spans the full
            peak range plus ``5 × x_broadening`` padding on each side;
            passing *xlims* does not clip the grid.  Use
            ``PlotSettings.xlim`` to control the display limits.
        ylims : tuple of float, optional
            Same as *xlims* but for the indirect (y) dimension.

        Returns
        -------
        ContourData
            Named-tuple with fields ``X``, ``Y``, ``Z`` (meshgrid arrays),
            ``x_broadening``, ``y_broadening``, ``broadening_type``,
            ``xlims``, ``ylims``.
        """
        peaks = self.get_peaks()
        if not peaks:
            raise ValueError("No peaks available – cannot compute contour data.")

        # Use absolute correlation strengths: the heatmap shows *magnitude*
        # of correlation.  Signed metrics (e.g. negative dipolar constants)
        # would otherwise produce a map with negative intensities.
        from dataclasses import replace as _dc_replace
        peaks_abs = [
            _dc_replace(p, correlation_strength=abs(p.correlation_strength))
            for p in peaks
        ]

        # Resolve default broadening from peak spread or supplied limits
        if xlims is None:
            x_min = min(p.x for p in peaks_abs)
            x_max = max(p.x for p in peaks_abs)
        else:
            x_min, x_max = min(xlims), max(xlims)

        if ylims is None:
            y_min = min(p.y for p in peaks_abs)
            y_max = max(p.y for p in peaks_abs)
        else:
            y_min, y_max = min(ylims), max(ylims)

        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0

        if x_broadening is None:
            x_broadening = 0.05 * x_range
        if y_broadening is None:
            y_broadening = 0.05 * y_range

        # Check cache
        cache_key = (x_broadening, y_broadening, grid_max, broadening_type, grid_size,
                     xlims, ylims)
        if getattr(self, '_contour_cache_key', None) == cache_key:
            return self._contour_data

        X, Y, Z = generate_contour_map(
            peaks_abs,
            grid_size=grid_size,
            broadening=broadening_type,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
        )

        if grid_max is not None:
            z_current_max = float(np.max(Z))
            if z_current_max > 0:
                Z = Z * (grid_max / z_current_max)
            else:
                self.logger.warning(
                    "grid_max requested but contour grid maximum is non-positive; "
                    "skipping grid scaling."
                )

        # Actual grid limits (may be wider than xlims due to broadening padding)
        actual_xlims = (float(X[0, 0]), float(X[0, -1]))
        actual_ylims = (float(Y[0, 0]), float(Y[-1, 0]))

        self._contour_data = ContourData(
            X=X,
            Y=Y,
            Z=Z,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
            broadening_type=broadening_type,
            xlims=actual_xlims,
            ylims=actual_ylims,
        )
        self._contour_cache_key = cache_key
        return self._contour_data

    def export_contour_data(
        self,
        path: str,
        fmt: str = 'simpson',
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        grid_max: Optional[float] = None,
        broadening_type: str = 'lorentzian',
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
        x_larmor_freq_mhz: Optional[float] = None,
        y_larmor_freq_mhz: Optional[float] = None,
    ) -> None:
        """
        Export the 2D NMR contour data to a file.

        The contour grid is computed via :meth:`get_contour_data` (and cached
        for subsequent calls).  The peak list is always included alongside the
        grid where the format supports it.

        Parameters
        ----------
        path : str
            Output file path.  For ``'simpson'`` format use a ``.spe``
            extension so that nmrglue / ssNake auto-detect the file type.
        fmt : {'simpson', 'npz', 'csv', 'json', 'ssnake'}
            Export format:

            ``'simpson'``
                SIMPSON TEXT format (``TYPE=SPE``).  Readable by nmrglue
                (``nmrglue.fileio.simpson.read``) and ssNake
                (*Open → SIMPSON*).  The grid is written as a 2D real
                spectrum; the imaginary channel is zero everywhere.  A
                companion ``<path>.peaks.csv`` file is written alongside.

            ``'npz'``
                NumPy compressed archive.  Arrays ``X``, ``Y``, ``Z`` plus
                scalar metadata are stored.  Reload with
                ``np.load(path, allow_pickle=True)``.

            ``'csv'``
                Flat table with columns ``x``, ``y``, ``intensity``.
                Useful for import into Origin, Excel, etc.

            ``'json'``
                ssNake native JSON format.  Stores Larmor frequency
                (``freq``) directly, so **ppm is available immediately**
                on load without any manual axis editing.  Requires
                *x_larmor_freq_mhz* (and *y_larmor_freq_mhz* for
                heteronuclear spectra).

        x_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **direct (x)** dimension nucleus.
            Only used by the ``'simpson'`` exporter.

        y_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **indirect (y)** dimension nucleus.
            For homonuclear experiments this equals *x_larmor_freq_mhz*.
            For heteronuclear experiments (e.g. 13C on x, 1H on y) the two
            frequencies differ and both must be provided.

            *Why these matter for ssNake:*  The SIMPSON TEXT format has no
            field for the spectrometer frequency.  ssNake therefore sets the
            carrier to 0 MHz on load and cannot offer ppm as a unit.  When
            Larmor frequencies are provided:

            * ``SW`` is written in Hz using *x_larmor_freq_mhz*.
            * ``SW1`` is written in Hz using *y_larmor_freq_mhz* (falls
              back to *x_larmor_freq_mhz* when *y_larmor_freq_mhz* is None).

            After loading in ssNake, go to *Axes → Edit axes* and enter the
            appropriate Larmor frequency for each dimension; ppm will then
            be available.

            When both are *None* sweep widths are written in ppm and a
            warning is emitted.

        x_broadening, y_broadening, broadening_type, grid_size, xlims, ylims
            Forwarded to :meth:`get_contour_data`.  If the grid has already
            been cached with identical parameters the cached result is reused.

        Raises
        ------
        ValueError
            If no peaks are available or an unknown format is requested.
        """
        from soprano.calculate.nmr.export import export_contour_data

        export_contour_data(
            nmr_data=self,
            path=path,
            fmt=fmt,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
            grid_max=grid_max,
            broadening_type=broadening_type,
            grid_size=grid_size,
            xlims=xlims,
            ylims=ylims,
            x_larmor_freq_mhz=x_larmor_freq_mhz,
            y_larmor_freq_mhz=y_larmor_freq_mhz,
        )

