# ADR 0002: SiteMap for atom-to-site provenance

**Status:** Accepted
**Date:** 2026-09-18

## Context

Soprano routinely derives one `Atoms` object from another, either by selecting a subset or by merging equivalent positions into representative sites. The derived object has its own index space, and code downstream frequently needs to translate between the two.

Nothing in the package owns that translation. `AtomSelection` states outright that it "does not keep track of the original Atoms object it's been created from", and `subset()` discards the index map it has just used. As a result the package accumulated six partial, mutually incompatible provenance channels: `tags`, `labels`, an `indices` array written by ASE's magres reader, `multiplicity`, an `order_tag` invented inside `reload_as_molecular_crystal`, and a single-consumer `index_map` from `nmr_extract_atoms`. None is authoritative and none composes.

`NMRData2D` showed what that costs. It carried a reduced structure, a pre-reduction structure (`atoms_full`), a pre-filter structure, and two separate index arrays at once, and recovered the reduced-to-full relation by joining on CIF labels. Functional-group averaging searched for `CH3` in a structure the element filter had already stripped of carbon, so `average_group` silently did nothing on homonuclear plots. Fixing that by searching the pre-filter structure instead still failed under `reduce=True`, because reduction had already merged the group members away. Only the user's original input has intact bonding topology.

## Decision

Introduce `SiteMap` in `soprano/sitemap.py`, exported from `soprano`, as the single way to relate two index spaces.

A `SiteMap` is a pure index map: an integer array of length `n_atoms` giving the site index each source atom maps to (`-1` when dropped), plus `n_sites`. It holds no `Atoms` references. It is constructed by `from_selection`, `from_tags` or `identity`, composes with `compose`, and inverts per site with `members(k)`.

Vocabulary, recorded in `CONTEXT.md`: the user's input structure has **atoms**; any derived structure has **sites**; a site represents one or more atoms; **multiplicity** is that count; the input is the **source** and is the only thing a site map is ever expressed against.

Shape mismatches raise. `compose` rejects maps whose spaces do not meet, `members` rejects an out-of-range site, and an opt-in `validate(atoms)` rejects a structure of the wrong length.

`NMRData2D` composes one `SiteMap` from source to final structure in `__init__` and uses it for both functional-group detection and dipolar RSS expansion. `atoms_full`, the `_all_full_indices` label join, and the pre-filter bookkeeping are all removed; the canonical structure is exposed as `source`.

`nmr_extract_multi` gains `return_sources`, so the 2D plotting CLI can ask for the structures as they were read instead of reading the files a second time to recover them.

## Considered options

**Extend `AtomSelection` with a parent back-reference.** Reuses existing vocabulary and adds no new concept. Rejected because `AtomSelection` deliberately does not track its parent, and more importantly because it can only express a selection. Reduction and functional-group averaging are many-to-one merges, which a selection cannot represent, so `reduce` and `average_group` would still have needed a second mechanism.

**Have `SiteMap` hold the source and derived `Atoms`.** Self-describing and impossible to pair with the wrong structure. Rejected because it ties the map to object lifetimes, costs picklability, and invites passing a `SiteMap` where an `Atoms` is expected. Shape checks plus an opt-in `validate` catch the realistic mistakes at a fraction of the cost.

**Keep it private to the NMR subpackage.** Smaller commitment. Rejected because the same pattern appears in `linkage`, `properties`, `collection` and the CLI scripts, and a private helper gives those no place to land.

**Teach `AtomSelection.subset`, `merge_sites` and `merge_tagged_sites` to return maps.** Rejected as unnecessary: `sel.indices` already *is* the map for a selection and the `tags` array already is for a merge, so the constructors reproduce both exactly with no producer changes. The producers get docstring pointers instead.

## Consequence: RSS now covers every atom a site represents

A site stands for all the atoms merged into it, so dipolar RSS for a site pair sums over all of them. That was already true for `expand_j='cif_labels'` and `'symmetry'`, which went through the label join. It was not true for `'periodic_images'` under `reduce=True`, which computed on the reduced cell and saw one copy of each neighbour, undercounting the sum by the symmetry multiplicity. The reduced cell is not a physical cell, so that result was wrong rather than merely different.

Values are unchanged for every other combination of `expand_j` and `reduce`, checked against the previous implementation on EDIZUM.

## Non-scope

These are known index-correspondence defects that `SiteMap` is the intended fix for. They are deliberately excluded here so this change stays reviewable; each has its own blast radius and needs its own tests. Tracked as separate issues.

- `Molecules.get(atoms)[0].subset(atoms)` yields a structure whose `bonds` array still holds parent indices. Confirmed out of range.
- `soprano spinsys --select_i` resolves against the reduced structure, so the user gets the nth atom of the asymmetric unit rather than of the file.
- `scripts/dipolar.py` carries an acknowledged TODO that indices do not match the dataframe for organic systems.
- `scripts/extract_nmr.py` builds groups with `range(max(tags))`, dropping the last group.
- `AtomsProperty.__call__` silently ignores `selection=` while `.get()` honours it.
- `soprano/nmr/extract.py` and `soprano/scripts/nmr.py` hold near-verbatim duplicates of the extraction pipeline, only one of which can produce an index map.

Group detection runs against the source rather than the reduced structure on principle, not on evidence: no fixture in the test corpus distinguishes the two. The source is the only structure guaranteed to have intact bonding topology and real positions, since a reduced site's position is whichever member `merge_first` happened to keep.

## Consequences

The `NMRData2D` diff is net-negative: one composed map replaces two stored structures, two index arrays and a label join, and removes a branch, a warning and a `ValueError` from `get_correlation_strengths`.

`atoms_full` is removed outright rather than deprecated. It is unreleased (it does not exist in `NMRData2D` on `master`) and nothing in the package, tests, docs or tutorials ever passed it.

Label-based index translation is no longer used for this purpose. It agreed with the index map on every site of the test corpus, so the change is behaviour-preserving there, but it required consistent labelling across two structures and could over-match when a label repeats.
