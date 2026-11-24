"""2D NMR data extraction and peak generation."""

import logging
import warnings
from dataclasses import replace as _dc_replace
from typing import List, Optional, Tuple, Union

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
    generate_contour_map,
    generate_peaks,
    get_atom_labels,
    get_pair_dipolar_couplings,
    get_pair_j_couplings,
    merge_peaks,
    prepare_species_labels,
    process_pairs,
    select_atoms_by_elements,
    sort_peaks,
    validate_elements,
)
from soprano.data.nmr import _get_isotope_list
from soprano.nmr.utils import _dip_constant
from soprano.properties.nmr import DipolarRSSByAtom, MSIsotropy
from soprano.sitemap import SiteMap

class NMRData2D:
    '''
    Class to hold and extract 2D NMR data.

    ``pairs``, if given, is a list of ``(x_atom, y_atom)`` tuples of indices
    into the input structure ``atoms``, whatever ``reduce`` is set to.  Each
    x atom must be of ``xelement`` and each y atom of ``yelement``.
    '''
    def __init__(self,
                atoms: Optional[Atoms] = None,
                xelement: Optional[str] = None,
                yelement: Optional[str] = None,
                references: Optional[dict[str, float]] = None,
                gradients: Optional[Union[dict[str, float], float]] = None,
                peaks: Optional[List[Peak2D]] = None,
                pairs: Optional[List[Tuple[int, int]]] = None,
                correlation_strengths: Optional[List[float]] = None,
                correlation_strength_metric: Optional[str] = None, # 'fixed','distance', 'dipolar', 'dipolar2', 'jcoupling', 'inversedistance', 'custom', 'dipolar_rss'
                rcut: Optional[float] = None,
                rss_cutoff: float = 5.0,
                rss_expand_j: str = 'periodic_images',
                reduce: bool = False,
                symprec: float = 1e-4,
                isotopes: Optional[dict[str, int]] = None,
                is_shift: Optional[bool] = None,
                include_quadrupolar: bool = False,
                yaxis_order: str = '1Q',
                x_axis_label: Optional[str] = None,
                y_axis_label: Optional[str] = None,
                average_group: str = "",
                ):

        self.logger = logging.getLogger(__name__)
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

        # The structure the user handed us is the source: the only index space
        # every derived structure can be related back to.  self.site_map records
        # that relation, and is composed as the structure is derived below.
        self.source: Optional[Atoms] = None
        self.site_map: Optional[SiteMap] = None
        if self.atoms is not None:
            from soprano.nmr.extract import label_atoms  # lazy import
            self.atoms = label_atoms(self.atoms)
            self.source = self.atoms
            self.site_map = SiteMap.identity(len(self.atoms))

        # Reduce to unique sites if requested, mirroring the CLI --reduce flag.
        if reduce and self.atoms is not None:
            from soprano.nmr.extract import nmr_extract_atoms  # lazy import
            _reduced, _reduce_map = nmr_extract_atoms(
                self.atoms.copy(), reduce=True, symprec=symprec,
                return_index_map=True,
            )
            if _reduced is not None:
                self.site_map = self.site_map.compose(
                    SiteMap(_reduce_map, len(_reduced))
                )
                if self.pairs is not None:
                    remapped = [
                        (int(self.site_map.to_sites[p[0]]), int(self.site_map.to_sites[p[1]]))
                        for p in self.pairs
                    ]
                    if remapped != [tuple(p) for p in self.pairs]:
                        warnings.warn(
                            "reduce=True: user-supplied pairs have been "
                            "remapped from input-structure to reduced "
                            f"(asymmetric-unit) indices: {list(self.pairs)}"
                            f" → {remapped}.",
                            stacklevel=2,
                        )
                self.atoms = _reduced

        # if atoms are provided, let's use the subset of atoms that have the xelement and yelement
        if self.atoms is not None:
            _sel = select_atoms_by_elements(self.atoms, [self.xelement, self.yelement])
            self.site_map = self.site_map.compose(
                SiteMap.from_selection(_sel, self.atoms)
            )
            self.atoms = _sel.subset(self.atoms)

        # User pairs are indices into the input structure; map them to the sites
        # of the structure the peaks are built on, keeping the atoms themselves
        # so their couplings are evaluated on the real atom pair.
        self._user_source_pairs: Optional[List[Tuple[int, int]]] = None
        if self.pairs is not None and self.source is not None:
            self._user_source_pairs = [(int(i), int(j)) for i, j in self.pairs]
            self.pairs = self._pairs_to_sites(self.pairs)


        # Either provide correlation strengths or calculate them based on the metric
        # If both are provided, the provided values will be used
        if correlation_strengths is not None:
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
        self.isotopes = isotopes if isotopes is not None else {}
        # is_shift is a boolean.  If undefined, it will be set to True if references are provided, False otherwise
        #  If defined, it will be used as is
        self.is_shift = is_shift if is_shift is not None else (self.references is not None)
        self.include_quadrupolar = include_quadrupolar
        self.yaxis_order = yaxis_order
        # Contour grid cache, filled by get_contour_data()
        self._contour_data: Optional[ContourData] = None
        self._contour_cache_key: Optional[tuple] = None

        self.correlation_unit = MARKER_INFO[self.correlation_strength_metric]['unit']
        self.correlation_label = MARKER_INFO[self.correlation_strength_metric]['label']
        self.correlation_fmt = MARKER_INFO[self.correlation_strength_metric]['fmt']

        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label
        self.average_group = average_group
        self._groups = None
        if self.average_group and self.source is not None:
            self._groups = self._find_groups()

        # run the main method to extract the data
        self.get_peaks()

    def _find_groups(self) -> dict:
        """Find the ``average_group`` groups in the source structure.

        A pattern like CH3 or NH3 is anchored on its heavy atom, and both
        deriving steps can remove it: the element filter drops carbon from an
        H-H plot, and reduction merges the group members away. Only the source
        has the bonding topology intact.

        Returns ``{group_id: [source atom indices]}``.
        """
        from soprano.nmr.extract import find_XHn_groups, label_atoms  # noqa: PLC0415

        atoms = label_atoms(self.source.copy(), logger=self.logger)
        groups = {}
        for ipat, pattern_groups in enumerate(find_XHn_groups(atoms, self.average_group)):
            for igrp, group in enumerate(pattern_groups):
                groups[(ipat, igrp)] = [int(a) for a in group]
        return groups

    def _site_representatives(self) -> np.ndarray:
        """Source atom standing for each site on the x side of a pair.

        By default this is the first member of the site. With functional-group
        averaging the representatives of the sites one group spans must come
        from the *same* physical group, or the group's members would be read
        off different molecules. For every class of equivalent groups (groups
        whose atoms fall on the same sites) one group is chosen, and its atoms
        represent their sites.
        """
        reps = np.array([int(self.site_map.members(k)[0])
                         for k in range(self.site_map.n_sites)], dtype=int)
        if not self._groups:
            return reps
        to_sites = self.site_map.to_sites
        claimed: set = set()
        for gid in sorted(self._groups):
            atoms = [a for a in self._groups[gid] if to_sites[a] >= 0]
            sites = {int(to_sites[a]) for a in atoms}
            if not sites or sites & claimed:
                continue
            claimed |= sites
            for a in atoms:
                site = int(to_sites[a])
                if reps[site] not in self._groups[gid]:
                    reps[site] = a
        return reps

    def _average_group_peaks(self, peaks: list) -> list:
        """Merge peaks from the same functional group (e.g. CH₃) post-hoc.

        Each peak carries the source atom pair it was evaluated on
        (``self.src_pairs``), so groups are identified on real atoms and every
        coupling is evaluated at the true inter-atomic distance.

        Peaks whose atoms belong to the same functional group and are paired
        with the same counter-atom (or counter-group) are replaced by a single
        representative peak:

        * **position** (x, y) — multiplicity-weighted mean of the members'
          positions,
        * **correlation_strength** — for ``dipolar``, the coupling of the
          group-averaged dipolar tensor; otherwise the multiplicity-weighted
          mean of the members' values,
        * **multiplicity** — sum of the members' multiplicities,
        * **label** — comma-separated sorted concatenation of member labels,
          matching the CLI ``--average-group`` convention.

        Args:
            peaks: Peak list from :func:`generate_peaks` (before
                :func:`merge_peaks`), in the order of ``self.src_pairs``.

        Returns:
            New peak list with functional-group members merged.
        """
        import re  # noqa: PLC0415
        from collections import defaultdict  # noqa: PLC0415

        groups = self._groups or {}
        atom_to_group = {a: gid for gid, atoms in groups.items() for a in atoms}
        if not atom_to_group:
            self.logger.warning(
                f"average_group='{self.average_group}' matched no groups; "
                "returning peaks unchanged."
            )
            return peaks

        def gkey(atom: int):
            """Group id if the atom is in a group, else the atom itself."""
            return atom_to_group.get(int(atom), ('atom', int(atom)))

        def source_atoms(key) -> list:
            return groups[key] if key in groups else [key[1]]

        # --- bucket peaks by merged-pair key --------------------------------
        peer_map: dict = defaultdict(list)
        for peak, (a, b) in zip(peaks, self.src_pairs):
            peer_map[(gkey(a), gkey(b))].append(peak)

        # --- combine each bucket --------------------------------------------
        def _sort_key(label: str):
            m = re.search(r'\d+', label)
            return int(m.group()) if m else label

        use_tensor_avg = self.correlation_strength_metric == 'dipolar'

        result = []
        for key, group_peaks in peer_map.items():
            if len(group_peaks) == 1:
                result.append(group_peaks[0])
                continue

            weights = np.array([p.multiplicity for p in group_peaks], dtype=float)
            total_mult = int(weights.sum())
            avg_x = float(np.average([p.x for p in group_peaks], weights=weights))
            avg_y = float(np.average([p.y for p in group_peaks], weights=weights))

            avg_strength: Optional[float] = None
            if use_tensor_avg:
                # Fast exchange within the group: average the coupling
                # *tensors* over the member pairs, not the constants.  The
                # arithmetic mean of the constants ignores the orientational
                # cancellation between the pair tensors and overestimates
                # the residual coupling (see docs/dipolar-averaging.md).
                src_i = source_atoms(key[0])
                src_j = source_atoms(key[1])
                # A single identical atom on both sides has no valid pairs
                if len(src_i) > 1 or src_i != src_j:
                    from soprano.properties.nmr import (  # noqa: PLC0415
                        averaged_dipolar_coupling,
                    )
                    d_eff, _ = averaged_dipolar_coupling(
                        self.source, src_i, src_j, isotopes=self.isotopes
                    )
                    avg_strength = d_eff * 1e-3  # Hz -> kHz, matching MARKER_INFO

            if avg_strength is None:
                # Multiplicity-weighted mean preserves the invariant
                # merged_strength × merged_multiplicity = Σ(sᵢ × mᵢ),
                # consistent with merge_peaks.
                total_weight = sum(
                    p.correlation_strength * p.multiplicity for p in group_peaks
                )
                avg_strength = total_weight / total_mult if total_mult != 0 else 0.0

            xlabels = sorted({p.xlabel for p in group_peaks}, key=_sort_key)
            ylabels = sorted({p.ylabel for p in group_peaks}, key=_sort_key)
            result.append(_dc_replace(
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

    def get_peaks(self, merge_identical=True, should_sort_peaks=False, force_recompute=False) -> List['Peak2D']:
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

        self.peaks = generate_peaks(
            self.data, self.pairs, labels, self.correlation_strengths,
            self.yaxis_order, self.xelement, self.yelement,
            pair_multiplicities=self.pair_multiplicities,
        )

        if self.average_group:
            self.peaks = self._average_group_peaks(self.peaks)

        if merge_identical:
            self.peaks = merge_peaks(self.peaks, corr_rel_tol=0.05)

        if should_sort_peaks:
            self.peaks = sort_peaks(self.peaks)

        return self.peaks

    def extract_data(self) -> None:
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

        self.data = MSIsotropy.get(self.atoms, references=self.references, gradients=self.gradients)
        self.logger.debug(f'Indices of xelement in the atoms object: {self.idx_x}')
        self.logger.debug(f'Indices of yelement in the atoms object: {self.idx_y}')
        self.logger.debug(f'X species: {self.xspecies}')
        self.logger.debug(f'Y species: {self.yspecies}')
        self.logger.debug(f'X values: {self.data[self.idx_x]}')
        self.logger.debug(f'Y values: {self.data[self.idx_y]}')


    def get_correlation_strengths(self) -> np.ndarray:

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
            # One value per requested pair; atom pairs expanded from the same
            # requested pair share it.
            if len(self.correlation_strengths) != self._n_requested_pairs:
                raise ValueError(f"Length of correlation_strengths ({len(self.correlation_strengths)}) does not match the number of pairs ({self._n_requested_pairs}).")
            correlation_strengths = np.asarray(self.correlation_strengths)[self._pair_origin]

        elif self.correlation_strength_metric == 'fixed':
            self.logger.info("Using fixed correlation strength.")
            # set the correlation strength to be the same for all pairs
            correlation_strengths = np.ones(len(self.pairs))

        elif self.correlation_strength_metric in ('dipolar', 'dipolar2'):
            self.logger.info(f"Using {self.correlation_label.lower()} as correlation strength.")
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")
            correlation_strengths = get_pair_dipolar_couplings(self.source, self.src_pairs, self.isotopes)
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
            correlation_strengths = get_pair_j_couplings(self.source, self.src_pairs, self.isotopes)
        elif self.correlation_strength_metric == 'dipolar_rss':
            self.logger.info(
                f"Using dipolar RSS (cutoff={self.rss_cutoff} Å, "
                f"expand_j='{self.rss_expand_j}') as correlation strength."
            )
            if self.isotopes:
                self.logger.debug(f"Using custom isotopes: {self.isotopes}")

            # RSS is summed over the source structure, not over self.atoms.
            # Reduction merges symmetry-equivalent sites away, so expanding in
            # the derived structure would find nothing to expand; and a site
            # stands for every atom it was merged from, so all of them have to
            # contribute. site_map.members gives exactly those atoms.
            # RSS is a per-atom quantity (one x atom against the whole j set),
            # so each entry is evaluated for its x atom, and get_pairs weights
            # it by the multiplicity of the x site only.
            # Members of a symmetry-merged site agree, but a CIF-label merge
            # can group atoms that are not equivalent, so flag that case.
            rss_values = []
            cache: dict = {}
            for (i, j), (a, _) in zip(self.pairs, self.src_pairs):
                if (i, j) not in cache:
                    members_i = self.site_map.members(i).tolist()
                    members_rss = DipolarRSSByAtom.get(
                        self.source,
                        sel_i=members_i,
                        sel_j=self.site_map.members(j).tolist(),
                        cutoff=self.rss_cutoff,
                        isotopes=self.isotopes,
                        expand_j=self.rss_expand_j,
                        symprec=self.symprec,
                    )
                    if not np.allclose(members_rss, members_rss[0], rtol=1e-3, atol=0.0):
                        self.logger.warning(
                            f"Dipolar RSS differs between the atoms merged into site {i} "
                            f"(pair {i}-{j}): {members_rss.min():.1f} to {members_rss.max():.1f} Hz. "
                            "The merged atoms are not equivalent; using the representative atom's value."
                        )
                    cache[(i, j)] = dict(zip(members_i, members_rss))
                rss_values.append(cache[(i, j)][a])
            correlation_strengths = np.array(rss_values) * 1e-3  # convert Hz → kHz to match MARKER_INFO and dipolar metric
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

    def get_axis_labels(self) -> Tuple[str, str]:
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


    def _pairs_to_sites(self, pairs) -> List[Tuple[int, int]]:
        '''
        Map user pairs from input-structure indices to indices of self.atoms.

        Each pair is (x-element atom, y-element atom) in self.source.
        '''
        symbols = self.source.get_chemical_symbols()
        pairs_out = []
        for i, j in pairs:
            i, j = int(i), int(j)
            for idx, element, role in ((i, self.xelement, 'x'), (j, self.yelement, 'y')):
                if symbols[idx] != element:
                    raise ValueError(
                        f"Pair ({i}, {j}): atom {idx} is {symbols[idx]}, but the "
                        f"{role} element is {element}. Pairs are (x, y) indices "
                        "into the input structure."
                    )
            pairs_out.append((int(self.site_map.to_sites[i]), int(self.site_map.to_sites[j])))
        return pairs_out

    def get_pairs(self) -> None:
        '''
        Get the pairs of x and y indices to plot.

        Every pair metric is evaluated on atom pairs of the source structure,
        never on the derived (reduced) structure, which is not a physical
        crystal. A site pair ``(I, J)`` is expanded into the atom pairs
        ``(a, b)``, with ``a`` the representative atom of site ``I`` and ``b``
        every atom merged into site ``J``. Each atom pair stands for the
        ``m_I`` symmetry copies obtained by moving ``a`` over its site, so it
        is weighted by ``m_I``; summed over ``b`` this recovers the
        ``m_I × m_J`` atom pairs of the site pair, each with its own distance
        and coupling. Without reduction every site is one atom and this is
        the plain atom-pair list.

        User-supplied pairs are not expanded: the requested atom pair is used.
        ``dipolar_rss`` is a per-atom quantity that already sums over the j
        set, so it keeps one entry per site pair.

        This method sets, all aligned with each other:

        * ``self.pairs``: ``(I, J)`` site indices into ``self.atoms``,
        * ``self.src_pairs``: ``(a, b)`` atom indices into ``self.source``,
        * ``self.pair_multiplicities``: weight of each entry,
        * ``self.pair_distances``: minimum-image distance of ``(a, b)``,
        * ``self.pairs_el_idx``: ``(x, y)`` element indices of the site pair.
        '''
        if not hasattr(self, '_requested_site_pairs'):
            self._requested_site_pairs = self.pairs
        site_pairs, site_el_idx, _, _ = process_pairs(
            self.idx_x, self.idx_y, self._requested_site_pairs)
        self._n_requested_pairs = len(site_pairs)

        mult = (
            self.atoms.get_array('multiplicity')
            if self.atoms.has('multiplicity')
            else np.ones(len(self.atoms), dtype=int)
        )
        reps = self._site_representatives()
        dq = self.yaxis_order == '2Q'
        per_atom = self.correlation_strength_metric == 'dipolar_rss'

        pairs, src_pairs, mults, el_idx, origin = [], [], [], [], []
        rss_seen: set = set()
        rss_j_sets: dict = {}
        for k, (I, J) in enumerate(site_pairs):
            I, J = int(I), int(J)
            members_j = [int(b) for b in self.site_map.members(J)]
            if self._user_source_pairs is not None:
                candidates = [self._user_source_pairs[k]]
            elif per_atom:
                # One entry per (x atom, j set). Single-atom DQ self-pairs
                # are dropped like any other zero-separation pair.
                if dq and I == J and len(members_j) == 1:
                    continue
                j_set = self._rss_j_key(members_j)
                if (I, j_set) in rss_seen:
                    continue  # same atom against the same expanded j set
                rss_seen.add((I, j_set))
                rss_j_sets[len(pairs)] = j_set
                candidates = [(int(reps[I]), int(reps[J]))]
            else:
                candidates = [(int(reps[I]), b) for b in members_j]
            for a, b in candidates:
                # In a DQ experiment the diagonal peaks arise from two
                # *distinct* atoms of the same shift; an atom paired with
                # itself has zero separation and is unphysical.
                if dq and a == b and not per_atom:
                    continue
                pairs.append((I, J))
                src_pairs.append((int(a), int(b)))
                mults.append(int(mult[I]))
                el_idx.append(site_el_idx[k])
                origin.append(k)

        if not pairs:
            if dq:
                raise ValueError(
                    "No valid pairs found for DQ/SQ spectrum. "
                    "Diagonal peaks in a DQ experiment come from distinct atoms "
                    "with the same shift — check that the structure contains "
                    "coupled pairs of the requested elements."
                )
            raise ValueError("No pairs found after filtering. Please check the input file and/or the user-specified filters.")

        # Self-pairs that survive (SQ/homonuclear) take their distance and
        # coupling from the nearest equivalent site, so they are not undefined.
        if self.correlation_strength_metric != 'fixed':
            n_self = sum(1 for a, b in src_pairs if a == b)
            if n_self and not per_atom:
                self.logger.info(
                    f"{n_self} self-pair(s) remain with "
                    f"metric='{self.correlation_strength_metric}'. Each uses the "
                    "distance/coupling to its nearest equivalent site."
                )

        distances = calculate_distances(src_pairs, self.source)
        # An RSS entry stands for its whole expanded j set: use the nearest member.
        for n, j_set in rss_j_sets.items():
            a = src_pairs[n][0]
            distances[n] = calculate_distances([(a, b) for b in j_set], self.source).min()

        keep = np.arange(len(pairs))
        if self.rcut:
            self.logger.info(f"Filtering out pairs that are further than {self.rcut} Å apart.")
            self.logger.info(f"Number of pairs before filtering: {len(pairs)}")
            keep = np.flatnonzero(distances <= self.rcut)
            if len(keep) == 0:
                raise ValueError('No pairs found after filtering by distance. Try increasing the cutoff distance (rcut).')
            self.logger.info(f"Number of pairs remaining: {len(keep)}")

        self.pairs = [pairs[i] for i in keep]
        self.src_pairs = [src_pairs[i] for i in keep]
        self.pair_multiplicities = np.array([mults[i] for i in keep], dtype=int)
        self.pairs_el_idx = [el_idx[i] for i in keep]
        self.pair_distances = distances[keep]
        self._pair_origin = np.array([origin[i] for i in keep], dtype=int)
        self.logger.debug(f"Pairs remaining: {self.pairs}")

    def _rss_j_key(self, members_j: List[int]) -> tuple:
        '''
        The j set a dipolar RSS entry sums over, after ``rss_expand_j``.

        Two entries of the same x atom with the same key are the same RSS
        value; keeping both would count the j orbit twice.
        '''
        if self.rss_expand_j == 'periodic_images':
            return tuple(sorted(members_j))
        from soprano.properties.nmr.dipolar import _expand_sel_j_by_equiv  # noqa: PLC0415
        from soprano.selection import AtomSelection  # noqa: PLC0415
        cache = self.__dict__.setdefault('_rss_j_cache', {})
        key = tuple(sorted(members_j))
        if key not in cache:
            sel = _expand_sel_j_by_equiv(
                self.source, AtomSelection(self.source, list(key)),
                self.rss_expand_j, self.symprec)
            cache[key] = tuple(sorted(int(x) for x in sel.indices))
        return cache[key]

    def to_dataframe(self, include_metadata=True) -> 'pd.DataFrame':
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

    def resolve_broadening(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Return the FWHM broadening used for the contour grid in each dimension.

        A value given explicitly is returned unchanged.  An unspecified one
        defaults to 5 % of the peak range in that dimension (1.0 for a single
        peak, so the default is never zero).
        """
        peaks = self.get_peaks()
        if peaks:
            x_range = max(p.x for p in peaks) - min(p.x for p in peaks) or 1.0
            y_range = max(p.y for p in peaks) - min(p.y for p in peaks) or 1.0
        else:
            x_range = y_range = 1.0

        if x_broadening is None:
            x_broadening = 0.05 * x_range
        if y_broadening is None:
            y_broadening = 0.05 * y_range
        return x_broadening, y_broadening

    def get_contour_data(
        self,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        grid_max: Optional[float] = None,
        broadening_type: str = 'lorentzian',
        grid_size: int = 500,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
        use_signed: bool = False,
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
            ``'lorentzian'`` (default) or ``'gaussian'``.
        grid_size : int
            Number of points along each axis of the grid.  Default 500.
        xlims : tuple of float, optional
            Extends the grid, if needed, so that it covers these limits.  The
            grid otherwise spans the full peak range plus ``5 × x_broadening``
            padding on each side, and is never clipped.  The default broadening
            does not depend on *xlims*.  Use ``PlotSettings.xlim`` to control the
            display limits.
        ylims : tuple of float, optional
            Same as *xlims* but for the indirect (y) dimension.
        use_signed : bool, optional
            If ``True``, preserve the sign of correlation strengths when
            building the contour grid.  The default ``False`` takes the
            absolute value, which is appropriate for magnitude spectra.

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

        # By default take absolute correlation strengths so the heatmap shows
        # *magnitude* of correlation.  Signed metrics (e.g. negative dipolar
        # constants) would otherwise produce negative intensities.  When
        # use_signed=True the caller wants the signed map.
        if use_signed:
            peaks_for_grid = peaks
        else:
            peaks_for_grid = [
                _dc_replace(p, correlation_strength=abs(p.correlation_strength))
                for p in peaks
            ]

        x_broadening, y_broadening = self.resolve_broadening(x_broadening, y_broadening)

        # Check cache. The peak list is part of the key, so editing peaks
        # invalidates a grid computed from the old ones.
        peaks_key = tuple(
            (p.x, p.y, p.correlation_strength, p.multiplicity) for p in peaks_for_grid
        )
        cache_key = (peaks_key, x_broadening, y_broadening, grid_max, broadening_type,
                     grid_size, xlims, ylims, use_signed)
        if self._contour_cache_key == cache_key:
            return self._contour_data

        X, Y, Z = generate_contour_map(
            peaks_for_grid,
            grid_size=grid_size,
            broadening=broadening_type,
            x_broadening=x_broadening,
            y_broadening=y_broadening,
            xlims=xlims,
            ylims=ylims,
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
        fmt: Optional[str] = None,
        x_broadening: Optional[float] = None,
        y_broadening: Optional[float] = None,
        grid_max: Optional[float] = None,
        broadening_type: Optional[str] = None,
        grid_size: Optional[int] = None,
        xlims: Optional[Tuple[float, float]] = None,
        ylims: Optional[Tuple[float, float]] = None,
        x_larmor_freq_mhz: Optional[float] = None,
        y_larmor_freq_mhz: Optional[float] = None,
        b0_field_tesla: Optional[float] = None,
        spectrometer_freq_mhz: Optional[float] = None,
        use_signed: Optional[bool] = None,
        config: Optional["ExportConfig"] = None,
    ) -> None:
        """
        Export the 2D NMR contour data to a file.

        .. warning::
            Experimental. The formats, their parameters and the files written
            may change without a deprecation period.

        The contour grid is computed via :meth:`get_contour_data` (and cached
        for subsequent calls).  The peak list is always included alongside the
        grid where the format supports it.

        Parameters
        ----------
        path : str
            Output file path.  The format is inferred from the extension
            unless *fmt* is given explicitly.  Supported extensions:
            ``.spe``/``.sim`` → simpson, ``.npz`` → npz, ``.csv`` → csv,
            ``.json``/``.ssnake`` → json, ``.txt`` → plain.
        fmt : {'simpson', 'npz', 'csv', 'json', 'ssnake', 'plain', 'bruker'}, optional
            Export format.  If *None*, inferred from *path* extension.
            ``'bruker'`` must be specified explicitly (no extension mapping);
            *path* is the TopSpin dataset name (a directory, not a file).
            The data is written to ``<path>/1/pdata/1`` (experiment 1,
            processing 1), so open ``<path>/1/pdata/1`` in TopSpin.  The
            peak list is written as ``<path>/1/peaks.csv``, not as a
            ``<path>.peaks.csv`` file.

            ``'simpson'``
                SIMPSON TEXT format (``TYPE=SPE``).  Readable by nmrglue
                (``nmrglue.fileio.simpson.read``) and ssNake
                (*Open → SIMPSON*).  The grid is written as a 2D real
                spectrum; the imaginary channel is zero everywhere.  A
                companion ``<path>.peaks.csv`` file is written alongside.

            ``'npz'``
                NumPy compressed archive.  Arrays ``X``, ``Y``, ``Z`` plus
                scalar metadata are stored.  Reload with
                ``np.load(path, allow_pickle=True)``.  The peak list is stored
                in the archive (``peak_x``, ``peak_y``, ...); no companion
                file is written.

            ``'csv'``
                Flat table with columns ``x``, ``y``, ``intensity``.
                Useful for import into Origin, Excel, etc.  A companion
                ``<path>.peaks.csv`` file is written alongside.

            ``'json'``
                ssNake native JSON format.  Stores Larmor frequency
                (``freq``) directly, so **ppm is available immediately**
                on load without any manual axis editing.  Larmor
                frequencies are auto-computed from element gyromagnetic
                ratios when possible; provide *x_larmor_freq_mhz* to
                override.  The peak list is stored under ``peaks``; no
                companion file is written.

            ``'plain'``
                Simple space-separated text file (x y z columns) with a
                header comment block.  Useful for gnuplot or quick
                inspection.  A companion ``<path>.peaks.csv`` file is
                written alongside.

        x_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **direct (x)** dimension nucleus.
            Overrides auto-computation.  Only used by ``'simpson'`` and
            ``'json'`` exporters.

        y_larmor_freq_mhz : float, optional
            Larmor frequency in MHz for the **indirect (y)** dimension nucleus.
            For homonuclear experiments this equals *x_larmor_freq_mhz*.
            For heteronuclear experiments (e.g. 13C on x, 1H on y) the two
            frequencies differ and both must be provided.

            For DQ/SQ data (``yaxis_order='2Q'``) the indirect ppm values are
            δ_A + δ_B, so this is still the ordinary single-quantum Larmor
            frequency of the y nucleus, not twice it: doubling it halves the
            ppm scale.  The auto-computed value (from *b0_field_tesla* or
            *spectrometer_freq_mhz*) is already the single-quantum frequency,
            so leave it unset unless you need to override it.

            *Why these matter for ssNake:*  The SIMPSON TEXT format has no
            field for the spectrometer frequency.  ssNake therefore sets the
            carrier to 0 MHz on load and cannot offer ppm as a unit.  When
            Larmor frequencies are provided (or auto-computed):

            * ``SW`` is written in Hz using *x_larmor_freq_mhz*.
            * ``SW1`` is written in Hz using *y_larmor_freq_mhz* (falls
              back to *x_larmor_freq_mhz* when *y_larmor_freq_mhz* is None).

            After loading in ssNake, go to *Axes → Edit axes* and enter the
            appropriate Larmor frequency for each dimension; ppm will then
            be available.

            When both are *None* and auto-computation fails, sweep widths
            are written in ppm and a warning is emitted.

        b0_field_tesla : float, optional
            Magnetic field strength in Tesla.  Used to auto-compute Larmor
            frequencies from gyromagnetic ratios.  Overridden by explicit
            *x_larmor_freq_mhz* / *y_larmor_freq_mhz*.  If neither this nor
            *spectrometer_freq_mhz* is given, auto-computation is skipped and
            a warning is emitted.

        spectrometer_freq_mhz : float, optional
            Proton (¹H) spectrometer frequency in MHz (e.g. 600 for a 600 MHz
            instrument).  Converted to Tesla internally; alternative to
            *b0_field_tesla*.  *b0_field_tesla* takes precedence if both are
            given.

        config : ExportConfig, optional
            Full configuration object.  Keyword arguments take precedence
            over any values in *config*.

        x_broadening, y_broadening, broadening_type, grid_size, xlims, ylims, use_signed
            Forwarded to :meth:`get_contour_data`.

        Raises
        ------
        ValueError
            If no peaks are available, the format cannot be inferred, or an
            unknown format is requested.
        """
        from soprano.calculate.nmr.export import ExportConfig, export_contour_data

        # Start from provided config (or defaults), then apply explicit kwargs
        base = config if config is not None else ExportConfig()
        overrides = {
            "x_broadening": x_broadening,
            "y_broadening": y_broadening,
            "grid_max": grid_max,
            "broadening_type": broadening_type,
            "grid_size": grid_size,
            "xlims": xlims,
            "ylims": ylims,
            "x_larmor_freq_mhz": x_larmor_freq_mhz,
            "y_larmor_freq_mhz": y_larmor_freq_mhz,
            "b0_field_tesla": b0_field_tesla,
            "spectrometer_freq_mhz": spectrometer_freq_mhz,
            "use_signed": use_signed,
        }
        # replace() keeps every config field not overridden here (e.g. overwrite)
        effective = _dc_replace(base, **{k: v for k, v in overrides.items() if v is not None})

        # Reuse the cached grid when no rendering params were overridden.
        # This means plot-then-export (or get_contour_data-then-export) uses
        # the same grid the caller already computed, avoiding silent divergence.
        precomputed = (
            self._contour_data
            if not effective.has_rendering_overrides()
            else None
        )
        export_contour_data(nmr_data=self, path=path, fmt=fmt, config=effective, contour_data=precomputed)


