# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.20.2",
#     "soprano==0.11.1",
#     "ase==3.27.0",
#     "weas-widget==0.2.6",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Interactive NMR Properties Explorer


    ```
          _
        /|_|\
       / / \ \
      /_/   \_\
      \ \   / /
       \ \_/ /
        \|_|/

    ```
    SOPRANO: a Python library for generation, manipulation and analysis of large batches of crystalline structures


    *Developed within the CCP-NC project.
    """)
    return


@app.cell
def _():
    # Basic imports
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # ===== INLINED MARIMO UTILS =====
    # Utility functions for marimo notebooks (originally from marimo_utils.py)

    import json
    import numpy as np
    import pandas as pd
    from weas_widget.atoms_viewer import AtomsViewer
    from weas_widget.base_widget import BaseWidget
    from weas_widget.utils import ASEAdapter
    from soprano.selection import AtomSelection
    import altair as alt

    guiConfig = {"controls": {"enabled": False}}

    def view_atoms(
        atoms,
        model_style=1,
        boundary=[[-0.1, 1.1], [-0.1, 1.1], [-0.1, 1.1]],
        show_bonded_atoms=True,
        highlight_indices: list[int] | None = None,
    ):
        """Function to visualise an ASE Atoms object (of list of them) using weas_widget."""
        v = AtomsViewer(BaseWidget(guiConfig=guiConfig))
        v.atoms = ASEAdapter.to_weas(atoms)
        v.model_style = model_style
        v.boundary = boundary
        v.show_bonded_atoms = show_bonded_atoms
        v.color_type = "VESTA"
        v.cell.settings["showAxes"] = True
        if highlight_indices is not None:
            v.highlight.settings['my_highlight'] = {"type": "sphere", "indices": highlight_indices, "color": "yellow", "scale": 1.3}
        return v._widget

    atoms_viewer_params = {
        "model_style": mo.ui.dropdown(
            options={"Ball": 0, "Ball and Stick": 1, "Polyhedral": 2, "Stick": 3},
            label="Model Style",
            value="Ball and Stick",
        ),
        "show_bonded_atoms": mo.ui.checkbox(
            label="Show atoms bonded beyond cell", value=True
        )
    }

    viewer_model_style = atoms_viewer_params['model_style']
    viewer_show_bonded = atoms_viewer_params['show_bonded_atoms']
    return (
        AtomSelection,
        alt,
        np,
        pd,
        view_atoms,
        viewer_model_style,
        viewer_show_bonded,
    )


@app.cell
def _(AtomSelection, mo):
    # ===== ATOM SELECTION UTILITIES =====

    def create_atom_selection_ui(atoms=None):
        """Create all individual UI elements with options populated from atoms object."""

        # Get options from atoms object
        elements = list(set(atoms.get_chemical_symbols())) if atoms else []
        array_names = list(atoms.arrays.keys()) if atoms else []
        labels = list(set(atoms.arrays['labels'])) if atoms and 'labels' in atoms.arrays else []
        max_index = len(atoms) - 1 if atoms else 0

        ui_elements = {
            # Method selector
            'method': mo.ui.dropdown(
                # label="Selection Method",
                options=[
                    "all", "index", "from_element", "from_selection_string", 
                    "from_index", "from_label", "from_box", "from_sphere", 
                    "from_bonds", "from_array",
                    #   "unique"
                ],
                value="all"
            ),

            # Individual UI elements with populated options
            'index': mo.ui.text(
                label="Enter atom indices (comma-separated)",
                value="0,1,2",
                placeholder=f"0-{max_index}" if atoms else "0,1,2"
            ),

            'element': mo.ui.dropdown(
                label="Select Element",
                options=elements,
                value=elements[0] if elements else None
            ),

            'selection_string': mo.ui.text(
                label="Selection String (e.g., 'Si', 'Si.1-3', 'C1,H2')",
                value="Si",
                placeholder="Enter selection string"
            ),

            'label': mo.ui.dropdown(
                label="Select Label",
                options=labels,
                value=labels[0] if labels else None
            ),

            'box_corner1': mo.ui.text(
                label="Box Corner 1 (x,y,z)",
                value="0,0,0",
                placeholder="0,0,0"
            ),

            'box_corner2': mo.ui.text(
                label="Box Corner 2 (x,y,z)",
                value="10,10,10", 
                placeholder="10,10,10"
            ),

            'box_periodic': mo.ui.checkbox(
                label="Include periodic copies"
            ),

            'box_scaled': mo.ui.checkbox(
                label="Use fractional coordinates"
            ),

            'sphere_center': mo.ui.text(
                label="Sphere Center (x,y,z)",
                value="0,0,0",
                placeholder="0,0,0"
            ),

            'sphere_radius': mo.ui.number(
                label="Sphere Radius",
                value=5.0,
                start=0.1,
                step=0.1
            ),

            'sphere_periodic': mo.ui.checkbox(
                label="Include periodic copies"
            ),

            'sphere_scaled': mo.ui.checkbox(
                label="Use fractional coordinates"
            ),

            'bonds_center': mo.ui.number(
                label="Center Atom Index",
                value=0,
                start=0,
                stop=max_index,
                step=1
            ),

            'bonds_distance': mo.ui.number(
                label="Bond Distance",
                value=1,
                start=0,
                step=1
            ),

            'bonds_operator': mo.ui.dropdown(
                label="Comparison Operator",
                options=["le", "lt", "eq", "ge", "gt"],
                value="le"
            ),

            'array_name': mo.ui.dropdown(
                label="Array Name",
                options=array_names,
                value=array_names[0] if array_names else None
            ),

            'array_value': mo.ui.text(
                label="Array Value",
                value="",
                placeholder="Enter value to compare"
            ),

            'array_operator': mo.ui.dropdown(
                label="Comparison Operator",
                options=["eq", "lt", "le", "ge", "gt"],
                value="eq"
            ),

            # 'unique_symprec': mo.ui.number(
            #     label="Symmetry Precision",
            #     value=1e-4,
            #     start=1e-6,
            #     step=1e-5
            # ),
        }

        return ui_elements

    def create_selection_forms(ui_elements):
        """Create form objects for each selection method using batch().form() pattern."""

        # All atoms form (no parameters needed)
        all_form = mo.md("""
        ### Select All Atoms
        No parameters required.
        """).batch().form(submit_button_label="Select All")

        # Index selection form
        index_form = mo.md("""
        ### Index Selection
        {index_ui}
        """).batch(
            index_ui=ui_elements['index']
        ).form(submit_button_label="Select by Index")

        # Element selection form
        element_form = mo.md("""
        ### Element Selection
        {element_ui}
        """).batch(
            element_ui=ui_elements['element']
        ).form(submit_button_label="Select by Element")

        # Selection string form
        selection_string_form = mo.md("""
        ### Selection String
        {selection_string_ui}
        """).batch(
            selection_string_ui=ui_elements['selection_string']
        ).form(submit_button_label="Select by String")

        # Label selection form
        label_form = mo.md("""
        ### Label Selection
        {label_ui}
        """).batch(
            label_ui=ui_elements['label']
        ).form(submit_button_label="Select by Label")

        # Box selection form
        box_form = mo.md("""
        ### Box Selection
        **Corner 1:** {box_corner1_ui}
        **Corner 2:** {box_corner2_ui}
        **Options:** {box_periodic_ui} {box_scaled_ui}
        """).batch(
            box_corner1_ui=ui_elements['box_corner1'],
            box_corner2_ui=ui_elements['box_corner2'],
            box_periodic_ui=ui_elements['box_periodic'],
            box_scaled_ui=ui_elements['box_scaled']
        ).form(submit_button_label="Select by Box")

        # Sphere selection form
        sphere_form = mo.md("""
        ### Sphere Selection
        **Center:** {sphere_center_ui}
        **Radius:** {sphere_radius_ui}
        **Options:** {sphere_periodic_ui} {sphere_scaled_ui}
        """).batch(
            sphere_center_ui=ui_elements['sphere_center'],
            sphere_radius_ui=ui_elements['sphere_radius'],
            sphere_periodic_ui=ui_elements['sphere_periodic'],
            sphere_scaled_ui=ui_elements['sphere_scaled']
        ).form(submit_button_label="Select by Sphere")

        # Bonds selection form
        bonds_form = mo.md("""
        ### Bond Distance Selection
        **Center Atom:** {bonds_center_ui}
        **Distance:** {bonds_distance_ui}
        **Operator:** {bonds_operator_ui}
        """).batch(
            bonds_center_ui=ui_elements['bonds_center'],
            bonds_distance_ui=ui_elements['bonds_distance'],
            bonds_operator_ui=ui_elements['bonds_operator']
        ).form(submit_button_label="Select by Bonds")

        # Array selection form
        array_form = mo.md("""
        ### Array Value Selection
        **Array Name:** {array_name_ui}
        **Value:** {array_value_ui}
        **Operator:** {array_operator_ui}
        """).batch(
            array_name_ui=ui_elements['array_name'],
            array_value_ui=ui_elements['array_value'],
            array_operator_ui=ui_elements['array_operator']
        ).form(submit_button_label="Select by Array")

        # # Unique atoms form
        # unique_form = mo.md("""
        # ### Symmetry Unique Selection
        # **Precision:** {unique_symprec_ui}
        # """).batch(
        #     unique_symprec_ui=ui_elements['unique_symprec']
        # ).form(submit_button_label="Select Unique")

        return {
            "all": all_form,
            "index": index_form,
            "from_element": element_form,
            "from_selection_string": selection_string_form,
            "from_index": index_form,  # Same as index
            "from_label": label_form,
            "from_box": box_form,
            "from_sphere": sphere_form,
            "from_bonds": bonds_form,
            "from_array": array_form,
            # "unique": unique_form
        }

    def get_current_form(method, selection_forms):
        """Get the current form based on selected method."""
        return selection_forms.get(method, selection_forms["all"])

    def create_selection_from_form(method, form, atoms_in):
        """Create AtomSelection from form values."""

        atoms = atoms_in.copy()
        if not form.value:
            return AtomSelection.all(atoms)
        form_data = form.value
        try:
            if method == "all":
                return AtomSelection.all(atoms)

            elif method == "index" or method == "from_index":
                indices = [int(i.strip()) for i in form_data['index_ui'].split(",")]
                return AtomSelection(atoms, indices)

            elif method == "from_element":
                return AtomSelection.from_element(atoms, form_data['element_ui'])

            elif method == "from_selection_string":
                return AtomSelection.from_selection_string(atoms, form_data['selection_string_ui'])

            elif method == "from_label":
                return AtomSelection.from_array(atoms, 'labels', form_data['label_ui'])

            elif method == "from_box":
                corner1 = [float(x.strip()) for x in form_data['box_corner1_ui'].split(",")]
                corner2 = [float(x.strip()) for x in form_data['box_corner2_ui'].split(",")]
                return AtomSelection.from_box(
                    atoms, corner1, corner2,
                    periodic=form_data['box_periodic_ui'],
                    scaled=form_data['box_scaled_ui']
                )

            elif method == "from_sphere":
                center = [float(x.strip()) for x in form_data['sphere_center_ui'].split(",")]
                return AtomSelection.from_sphere(
                    atoms, center, form_data['sphere_radius_ui'],
                    periodic=form_data['sphere_periodic_ui'],
                    scaled=form_data['sphere_scaled_ui']
                )

            elif method == "from_bonds":
                return AtomSelection.from_bonds(
                    atoms, 
                    form_data['bonds_center_ui'],
                    form_data['bonds_distance_ui'],
                    op=form_data['bonds_operator_ui']
                )

            elif method == "from_array":
                # Try to convert value to appropriate type
                value_str = form_data['array_value_ui']
                try:
                    if '.' in str(value_str):
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str

                return AtomSelection.from_array(
                    atoms,
                    form_data['array_name_ui'],
                    value,
                    op=form_data['array_operator_ui']
                )

            # elif method == "unique":
            #     return AtomSelection.unique(atoms, symprec=form_data['unique_symprec_ui'])

            else:
                return AtomSelection(atoms, [])

        except Exception as e:
            mo.output.append(mo.md(f"## Error creating selection: {e}"))
            return AtomSelection(atoms, [])
    return (
        create_atom_selection_ui,
        create_selection_forms,
        create_selection_from_form,
        get_current_form,
    )


@app.cell
def _(alt, np, pd):
    # ===== NMR SPECTRUM UTILITIES =====

    def create_nmr_spectrum(df, shielding_column, broadening=0.5, points=2048, 
                           chemical_shift_range=None, normalize=True,
                            reference_shielding = 0,
                           species=None, species_column='species', labels_column='labels'):
        """
        Create a simulated NMR spectrum from magnetic shielding data.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the magnetic shielding data
        shielding_column : str
            Name of the column containing magnetic shielding values
        broadening : float
            Gaussian broadening factor (FWHM in ppm), default 0.5
        points : int
            Number of points in the spectrum, default 2048
        chemical_shift_range : tuple or None
            Range for chemical shift axis (min, max) in ppm. If None, automatically 
            determined from data range with 10% padding, default None
        normalize : bool
            Whether to normalize the spectrum, default True
        reference_shielding : float
            Reference shielding value for converting to chemical shifts, default 0 ppm
        species : str or None
            Species to filter by (e.g., 'H', 'C', 'N'). If None, uses all data
        species_column : str
            Name of the column containing species information, default 'species'
        labels_column : str
            Name of the column containing atom labels, default 'labels'

        Returns:
        --------
        dict
            Dictionary with 'spectrum' DataFrame and 'peaks' DataFrame
        """

        # Filter by species if specified
        if species is not None:
            if species_column not in df.columns:
                raise ValueError(f"Species column '{species_column}' not found in DataFrame")

            # Filter for rows containing the specified species
            filtered_df = df[df[species_column].str.contains(species, case=False, na=False)]

            if filtered_df.empty:
                raise ValueError(f"No rows found containing species '{species}' in column '{species_column}'")

            print(f"Filtered to {len(filtered_df)} rows containing species '{species}'")
        else:
            filtered_df = df

        # Extract shielding values, labels, and remove any NaN values
        useful_columns = [shielding_column]
        if labels_column in filtered_df.columns:
            useful_columns.append(labels_column)

        clean_df = filtered_df[useful_columns].dropna(subset=[shielding_column])
        shielding_values = clean_df[shielding_column].values

        # Get labels if available
        if labels_column in filtered_df.columns and not clean_df.empty:
            labels = clean_df[labels_column].values
        else:
            labels = [f"Peak_{i+1}" for i in range(len(shielding_values))]
            if labels_column not in filtered_df.columns:
                print(f"Warning: Labels column '{labels_column}' not found, using generic labels")

        # Convert shielding to chemical shift (δ = σ_ref - σ_sample)
        chemical_shifts = reference_shielding - shielding_values

        # Determine chemical shift range
        if chemical_shift_range is None:
            if len(chemical_shifts) == 1:
                # If only one peak, set a default range around it
                cs_min = chemical_shifts[0] - 5
                cs_max = chemical_shifts[0] + 5
                print(f"Only one peak found. Setting chemical shift range to: {cs_min:.2f} to {cs_max:.2f} ppm")
            else:
                # Auto-determine range from data with 10% padding
                cs_min_data = np.min(chemical_shifts)
                cs_max_data = np.max(chemical_shifts)
                cs_range = cs_max_data - cs_min_data
                if cs_range < 1e-3:
                    cs_range = 10.0
                padding = cs_range * 0.1  # 10% padding on each side
                cs_min = cs_min_data - padding
                cs_max = cs_max_data + padding
                print(f"Auto-determined chemical shift range: {cs_min:.2f} to {cs_max:.2f} ppm")
        else:
            cs_min, cs_max = chemical_shift_range

        # Create the chemical shift axis
        cs_axis = np.linspace(cs_min, cs_max, points)

        # Initialize spectrum and peak information
        spectrum = np.zeros(points)
        peak_info = []  # Store peak information for tooltips

        # Add peaks for each chemical shift
        sigma = broadening / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
        for i, (cs, label) in enumerate(zip(chemical_shifts, labels)):
            if cs_min <= cs <= cs_max:
                # Create Gaussian peak
                peak = np.exp(-0.5 * ((cs_axis - cs) / sigma) ** 2)
                spectrum += peak

                # Store peak information
                peak_info.append({
                    'chemical_shift': cs,
                    'label': str(label),
                    'shielding': shielding_values[i]
                })

        # Normalize if requested
        if normalize and np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        # Create DataFrame for plotting
        spectrum_df = pd.DataFrame({
            'chemical_shift': cs_axis,
            'intensity': spectrum
        })

        # Add peak information for hover tooltips

        # Create a mapping of peak positions to labels
        peak_labels_map = {}
        for peak_data in peak_info:
            # Find closest point in spectrum to this chemical shift
            closest_idx = np.argmin(np.abs(cs_axis - peak_data['chemical_shift']))
            peak_labels_map[closest_idx] = peak_data

        # Add label information to spectrum dataframe
        spectrum_df['peak_label'] = ''
        spectrum_df['atom_shielding'] = np.nan

        for idx, peak_data in peak_labels_map.items():
            # Assign label to nearby points (within broadening width)
            sigma_points = int(sigma / (cs_axis[1] - cs_axis[0]))  # Convert sigma to points
            start_idx = max(0, idx - sigma_points)
            end_idx = min(len(spectrum_df), idx + sigma_points + 1)

            spectrum_df.loc[start_idx:end_idx, 'peak_label'] = peak_data['label']
            spectrum_df.loc[start_idx:end_idx, 'atom_shielding'] = peak_data['shielding']

        return {
        "spectrum": spectrum_df,
        "peaks": pd.DataFrame(peak_info)  # chemical_shift, label, shielding
    }


    def plot_nmr_spectrum(results, title="Simulated NMR Spectrum", width=700, height=400, y_headroom=0.1, show_labels=True):
        """
        Plot NMR spectrum with chemical shift axis on bottom and peak labels on top.
        Parameters
        ----------
        results : dict
            Output from create_nmr_spectrum containing:
            - 'spectrum' : DataFrame with chemical_shift, intensity
            - 'peaks'    : DataFrame with chemical_shift, label, shielding
        title : str
            Plot title
        width, height : int
            Chart dimensions
        y_headroom : float
            Fractional headroom to add above max intensity
        show_labels : bool
            Whether to show peak labels above ticks
        """
        spectrum_df = results["spectrum"]
        peaks_df = results["peaks"]

        if spectrum_df.empty:
            return alt.Chart().mark_text(text="No data to display")

        y_max = spectrum_df['intensity'].max() * (1 + y_headroom)

        # --- main spectrum line (defines bottom axis) ---
        line_chart = alt.Chart(spectrum_df).mark_line(
            color='#2E86AB', strokeWidth=2
        ).encode(
            x=alt.X('chemical_shift:Q',
                    title='Chemical Shift (\u03b4) / ppm',
                    scale=alt.Scale(reverse=True),
                    axis=alt.Axis(
                        orient='bottom',
                        titleFontSize=14,
                        labelFontSize=12,
                        grid=False,
                        domain=True,
                        ticks=True,
                        labels=True
                    )),
            y=alt.Y('intensity:Q',
                    title='Intensity',
                    scale=alt.Scale(domain=[0, y_max]),
                    axis=alt.Axis(
                        orient='left',
                        titleFontSize=14,
                        labelFontSize=12,
                        labels=False,
                        ticks=True,
                        domain=True,
                        grid=False
                    )),
            tooltip=[
                alt.Tooltip('peak_label:N', title='Site'),
                alt.Tooltip('chemical_shift:Q', title='\u03b4/ppm', format='.3f'),
                alt.Tooltip('atom_shielding:Q', title='Shielding/ppm', format='.3f')
            ]
        )


        # --- peak ticks at top ---
        peak_ticks = alt.Chart(peaks_df).mark_tick(
            size=15, thickness=2
        ).encode(
            x=alt.X('chemical_shift:Q', scale=alt.Scale(reverse=True)),
            y=alt.value(height - 20)  # position from top of plot area
        )

        # --- peak labels above ticks ---
        peak_labels = alt.Chart(peaks_df).mark_text(
            dy=-25, fontSize=11, clip=False
        ).encode(
            x=alt.X('chemical_shift:Q', scale=alt.Scale(reverse=True)),
            y=alt.value(height - 20),
            text='label:N'
        )

        # --- combine all with proper resolution ---
        chart_tuple = line_chart + peak_ticks
        if show_labels:
            chart_tuple += peak_labels
        chart = chart_tuple.add_params(
            alt.selection_interval(bind='scales', encodings=['x'])
        ).resolve_scale(
            x='shared',
            y='shared'
        ).properties(
            title=alt.TitleParams(text=title, fontSize=16, anchor='start'),
            width=width,
            height=height,
            padding={"top": 10, "left": 80, "right": 20, "bottom": 10}
        ).configure_view(
            continuousWidth=width,
            continuousHeight=height,
            clip=False  # disable clipping to show labels outside plot area
        )

        return chart


    def get_available_species(df, species_column='species'):
        """
        Get a list of unique species available in the DataFrame.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the species data
        species_column : str
            Name of the column containing species information

        Returns:
        --------
        list
            List of unique species found in the column
        """
        if species_column not in df.columns:
            return []

        return sorted(df[species_column].dropna().unique().tolist())
    return create_nmr_spectrum, get_available_species, plot_nmr_spectrum


@app.cell
def _():
    # Other useful imports

    import ase
    from ase import io as ase_io

    from soprano.properties.nmr import MSIsotropy, MSAnisotropy, MSAsymmetry, MSSpan, MSSkew, EFGVzz, EFGTensor, EFGAsymmetry, EFGAnisotropy, EFGSpan, EFGQuaternion, MSTensor, MSEuler, EFGQuadrupolarConstant
    from soprano.properties.nmr.dipolar import DipolarCoupling
    from soprano.nmr import NMRTensor
    from soprano.properties.labeling import MagresViewLabels


    from soprano.scripts.cli_utils import reload_as_molecular_crystal
    from soprano.scripts.nmr import build_nmr_df

    import matplotlib.pyplot as plt
    return (
        EFGTensor,
        MSTensor,
        MagresViewLabels,
        NMRTensor,
        ase_io,
        build_nmr_df,
        reload_as_molecular_crystal,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load a magres file

    Upload a `.magres` file to visualise NMR properties. The file should contain magnetic shielding (`ms`) data.

    ASE can read the Magres file format and is usually used here. However Soprano just requires that an Atoms object has the relevant data attached. For example, to work with magnetic shielding information, the Atoms object should have an 'ms' array.
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload a .magres file",
        filetypes=[".magres"],
        multiple=False
    )
    return (file_upload,)


@app.cell
def _(file_upload, mo):
    mo.vstack([
        file_upload,
        mo.md(f"✅ Loaded: **{file_upload.value[0].name}**") if file_upload.value else mo.md("_Upload a `.magres` file to begin_")
    ])
    return


@app.cell
def _(ase_io, file_upload, mo, np, reload_as_molecular_crystal):
    import tempfile
    from pathlib import Path as TempPath

    def parse_magres_file(uploaded_file):
        """Parse uploaded .magres file using ASE.

        Returns ASE Atoms object with NMR data, or None if parsing fails.
        """
        if uploaded_file is None:
            return None, None

        with tempfile.NamedTemporaryFile(suffix=".magres", delete=False) as tmp:
            tmp.write(uploaded_file.contents)
            temp_path = TempPath(tmp.name)

        try:
            atoms = ase_io.read(temp_path)
            atoms.center()
            atoms = reload_as_molecular_crystal(atoms)

            # hack to make build_nmr_df happy
            if not atoms.has('multiplicity'):
                atoms.set_array('multiplicity', np.ones(len(atoms), dtype=int))
                atoms.set_array('tags', np.arange(len(atoms)))

            return atoms, uploaded_file.name
        except Exception as e:
            mo.output.append(mo.md(f"**Error parsing file:** {e}"))
            return None, None
        finally:
            temp_path.unlink()

    if file_upload.value:
        atoms, magres_filename = parse_magres_file(file_upload.value[0])
    else:
        atoms, magres_filename = None, None

    atoms_not_loaded_correctly = (atoms is None) or (len(atoms) == 0) or (not atoms.has('ms'))
    mo.stop(atoms_not_loaded_correctly, mo.md(r"""**Please upload a `.magres` file above to load the data and continue with the tutorial.**"""))
    return atoms, magres_filename


@app.cell
def _(atoms, mo, view_atoms, viewer_model_style, viewer_show_bonded):
    mo.vstack([view_atoms(atoms, model_style=viewer_model_style.value, show_bonded_atoms=viewer_show_bonded.value), mo.hstack([viewer_model_style, viewer_show_bonded])])
    return


@app.cell
def _(atoms, create_atom_selection_ui, create_selection_forms, mo):
    sel_ui_elements = create_atom_selection_ui(atoms)
    selection_forms = create_selection_forms(sel_ui_elements)

    sel_method_ui = sel_ui_elements['method']

    mo.vstack([mo.md("## Select subset by"), sel_method_ui])
    return sel_method_ui, selection_forms


@app.cell
def _(get_current_form, sel_method_ui, selection_forms):
    current_sel_form = get_current_form(sel_method_ui.value, selection_forms)
    current_sel_form 
    return (current_sel_form,)


@app.cell
def _(
    atoms,
    create_selection_from_form,
    current_sel_form,
    mo,
    sel_method_ui,
    view_atoms,
):
    selection = create_selection_from_form(sel_method_ui.value, current_sel_form, atoms)
    sel_subset = selection.subset(atoms)

    mo.accordion({"View selected subset of atoms": view_atoms(atoms, highlight_indices=selection.indices)}, lazy=False)
    return (sel_subset,)


@app.cell
def _(np, sel_subset):
    atoms_labels = sel_subset.get_array('labels')
    print(f'There are {len(atoms_labels)} labels - one for each site')
    print(f'but there are only {len(np.unique(atoms_labels))} unique labels.')
    print(f'\nThe unique labels are: \n{np.unique(atoms_labels)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Magnetic shieldings and chemical shifts

    All the NMR tensors stored in the original .magres file are saved as arrays in the Atoms object and can be accessed directly. However, Soprano also provides a set of properties to express the tensors in the form of parameters useful to compute the spectrum.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Shielding conventions

    There are many different conventions used to describe the magnetic shielding tensors. These are nicely explained here:
    http://anorganik.uni-tuebingen.de/klaus/nmr/index.php?p=conventions/csa/csa

    and in the Soprano documentation.

    There are some convenience methods in Soprano to get descriptions of the MS tensor according to different conventions. First we can extract a list of MagneticShielding objects from the Atoms object, then we can use different methods to get the tensor in different conventions.
    """)
    return


@app.cell
def _(MagresViewLabels, mo, sel_subset):
    subset_labels = MagresViewLabels.get(sel_subset)

    # Dropdown with the subset labels
    label_selector_ui = mo.ui.dropdown(options=subset_labels, label='Select atom label:', value=subset_labels[0])

    label_selector_ui
    return label_selector_ui, subset_labels


@app.cell
def _(MSTensor, label_selector_ui, mo, np, sel_subset, subset_labels):
    mstensors = MSTensor.get(sel_subset)
    # Index of selected label
    index = np.where(np.array(subset_labels) == label_selector_ui.value)[0][0]
    ms0 = mstensors[index]
    mo.ui.tabs(tabs={
        "IUPAC": ms0.iupac_values,
        "Haeberlen": ms0.haeberlen_values,
        "Herzfeld-Berger": ms0.herzfeldberger_values})
    return


@app.cell
def _(build_nmr_df, magres_filename, mo, sel_subset, subset_labels):
    df = build_nmr_df(sel_subset, fname = magres_filename)
    # Add in the magresview labels
    df['magresview_labels'] = subset_labels
    dataframe = mo.ui.dataframe(df, page_size=10)

    mo.vstack([
        mo.md(r"""## NMR DataFrame"""),
        mo.md(r"""The DataFrame below contains all the NMR parameters computed by Soprano for the selected atoms. You can further filter sites and also sort the DataFrame as needed. The filtered sites are then used to compute the NMR spectrum above (by default it includes all the selection."""),
        mo.md(r"""You can also select rows in the DataFrame to highlight the corresponding atoms in the structure view above."""),
        dataframe
    ])
    return dataframe, df


@app.cell
def _(dataframe, mo, sel_subset, view_atoms):
    df_filtered = dataframe.value
    filtered_indices = df_filtered['original_index']
    filtered_labels = df_filtered['labels']


    mo.accordion({"Click here to view the filtered subset of atoms": view_atoms(sel_subset[filtered_indices])}, lazy=False)
    return filtered_indices, filtered_labels


@app.cell
def _(
    broadening_slider,
    create_nmr_spectrum,
    df,
    mo,
    plot_nmr_spectrum,
    points_slider,
    reference_shielding_ui,
    show_labels_toggle,
    species_picker,
):
    # Create spectrum with current parameters
    spectrum_data = create_nmr_spectrum(
        df,
        'MS_shielding',  # Replace with your actual column name
        broadening=broadening_slider.value,
        points=points_slider.value,
        species=species_picker.value,
        reference_shielding=reference_shielding_ui.value,
        labels_column='magresview_labels',
    )

    chart = plot_nmr_spectrum(spectrum_data, show_labels=show_labels_toggle.value)



    mo.vstack([
        mo.md(r"""### Simulated NMR Spectrum"""),
        mo.hstack([species_picker, reference_shielding_ui]),
        mo.hstack([broadening_slider, points_slider]),
        show_labels_toggle,
        chart
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## General NMR tensor

    Soprano provides a general `NMRTensor` class that can be used in isolation, but also some convenience methods for dealing with NMR tensors from e.g. a `.magres` file. First let's look at the general case:
    """)
    return


@app.cell
def _(mo, np):
    # Create a 3x3 grid of number inputs
    default_tensor_data = np.array([
        [30.29817962,  1.20510693,  3.67274493],
        [1.96313295, 27.57652505,  2.57545224],
        [4.21834132,  2.16271308, 30.90315252]
    ])

    def create_tensor_grid():
        inputs = {}
        for i in range(3):
            for j in range(3):
                inputs[f"{i}{j}"] = mo.ui.number(
                    value=default_tensor_data[i, j],
                    label=f"[{i},{j}]",
                )
        return inputs
    tensor_inputs = create_tensor_grid()

    # The grid approach doesn't work so manually create the elements
    tensor_input_ui_00 = tensor_inputs["00"]
    tensor_input_ui_01 = tensor_inputs["01"]
    tensor_input_ui_02 = tensor_inputs["02"]
    tensor_input_ui_10 = tensor_inputs["10"]
    tensor_input_ui_11 = tensor_inputs["11"]
    tensor_input_ui_12 = tensor_inputs["12"]
    tensor_input_ui_20 = tensor_inputs["20"]
    tensor_input_ui_21 = tensor_inputs["21"]
    tensor_input_ui_22 = tensor_inputs["22"]


    # Display as a grid
    mo.md(f"""
    ### Enter 3x3 Tensor Values:
    |       |       |       |
    |-------|-------|-------|
    | {tensor_input_ui_00} | {tensor_input_ui_01} | {tensor_input_ui_02} |
    | {tensor_input_ui_10} | {tensor_input_ui_11} | {tensor_input_ui_12} |
    | {tensor_input_ui_20} | {tensor_input_ui_21} | {tensor_input_ui_22} |
    """)
    return (
        tensor_input_ui_00,
        tensor_input_ui_01,
        tensor_input_ui_02,
        tensor_input_ui_10,
        tensor_input_ui_11,
        tensor_input_ui_12,
        tensor_input_ui_20,
        tensor_input_ui_21,
        tensor_input_ui_22,
    )


@app.cell
def _(
    np,
    tensor_input_ui_00,
    tensor_input_ui_01,
    tensor_input_ui_02,
    tensor_input_ui_10,
    tensor_input_ui_11,
    tensor_input_ui_12,
    tensor_input_ui_20,
    tensor_input_ui_21,
    tensor_input_ui_22,
):
    # Convert to numpy array
    tensor = np.array([
        [tensor_input_ui_00.value, tensor_input_ui_01.value, tensor_input_ui_02.value],
        [tensor_input_ui_10.value, tensor_input_ui_11.value, tensor_input_ui_12.value],
        [tensor_input_ui_20.value, tensor_input_ui_21.value, tensor_input_ui_22.value]
    ])
    return (tensor,)


@app.cell
def _(NMRTensor, mo, tensor):

    t = NMRTensor(tensor)
    # Now t is an NMRTensor object. We can extract a variety of properties of this tensor 
    # and perform updates to e.g. the tensor ordering

    # These are the ordering conventions available in Soprano:
    orders = ['Increasing', 'Decreasing', 'NQR','Haeberlen']
    convention_order_ui = mo.ui.dropdown(options=orders, label='Select tensor ordering convention:', value='Haeberlen')
    return convention_order_ui, t


@app.cell
def _(
    convention_order_ui,
    euler_angle_convention_ui,
    euler_convention,
    euler_passive,
    mo,
    np,
    t,
):
    t.order = convention_order_ui.value.lower()[0]
    mo.vstack([
        mo.hstack([convention_order_ui, euler_angle_convention_ui], justify='start'),
        mo.md(r"""### Tensor properties"""),
        mo.md(f"""The tensor eigenvalues are: \n
    {t.eigenvalues[0]:.2f}, {t.eigenvalues[1]:.2f}, {t.eigenvalues[2]:.2f}"""),
        mo.md(f"""The Euler angles (in degrees) according to the chosen convention are: \n
    α={t.euler_angles(convention=euler_convention, passive=euler_passive)[0]*180/np.pi:.1f}, β={t.euler_angles(convention=euler_convention, passive=euler_passive)[1]*180/np.pi:.1f}, γ={t.euler_angles(convention=euler_convention, passive=euler_passive)[2]*180/np.pi:.1f}""")
    ])
    return


@app.cell
def _(EFGTensor, MSTensor, sel_subset):
    # get a list of NMRTensor objects with Haeberlen ordering:
    ms_tensors = MSTensor.get(sel_subset)
    efg_tensors = EFGTensor.get(sel_subset)
    return efg_tensors, ms_tensors


@app.cell
def _(df, get_available_species, mo):
    available_species = get_available_species(df)
    broadening_slider = mo.ui.slider(0.05, 2.5, value=0.5, step=0.05, label="Broadening (ppm)", include_input=True)
    points_slider = mo.ui.number(start=512, stop=8192, value=2048, label="Resolution (number of bins)")
    species_picker = mo.ui.dropdown(options=available_species, label='Select species to plot', value=available_species[0] if available_species else None)
    reference_shielding_ui = mo.ui.number(value=0.0, step=0.1, label='Reference shielding (ppm)')
    show_labels_toggle = mo.ui.checkbox(label='Show peak labels', value=True)
    return (
        broadening_slider,
        points_slider,
        reference_shielding_ui,
        show_labels_toggle,
        species_picker,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Equivalent Euler angles

    In the general case, the Euler angles that describe the rotation from one NMR tensor to another are not unique. In fact, for a given choice of conventions, there can be up to 16 equivalent Euler angle sets all correctly describing the same rotation.

    For example, we can look at the relative Euler angles between two MS tensors from our filtered selection of atoms.
    """)
    return


@app.cell
def _(filtered_indices, mo):
    euler_angle_convention_ui = mo.ui.dropdown(options={'ZYZ (active)': ('zyz', False), 'ZYZ (passive)': ('zyz', True), 'ZXZ (active)': ('zxz', False), 'ZXZ (passive)': ('zxz', True)}, label='Euler angle convention:', value='ZYZ (active)')
    if len(filtered_indices) > 1:
        euler_selection_start_ui = mo.ui.dropdown(options=filtered_indices, label='MS tensor from:', value=filtered_indices.tolist()[0])
        euler_selection_end_ui = mo.ui.dropdown(options=filtered_indices, label='MS tensor to:', value=filtered_indices.tolist()[1])
        mo.output.append(mo.vstack([
            mo.hstack([euler_selection_start_ui, euler_selection_end_ui], justify='start'),
            euler_angle_convention_ui
        ]))
    return (
        euler_angle_convention_ui,
        euler_selection_end_ui,
        euler_selection_start_ui,
    )


@app.cell
def _(
    euler_angle_convention_ui,
    euler_columns,
    euler_selection_end_ui,
    euler_selection_start_ui,
    filtered_indices,
    mo,
    ms_tensors,
    np,
    pd,
):
    euler_convention, euler_passive = euler_angle_convention_ui.value
    if len(filtered_indices) > 1:
        euler_selection_start_index = euler_selection_start_ui.value
        euler_selection_end_index = euler_selection_end_ui.value
        equivalent_eulers_0_to_1 = ms_tensors[euler_selection_start_index].equivalent_euler_to(ms_tensors[euler_selection_end_index], euler_convention, passive=euler_passive)

        equivalent_eulers_0_to_1_vis = np.round(equivalent_eulers_0_to_1 * 180 / np.pi, 3)

        df_equivalent_eulers = pd.DataFrame({
            'α (deg)': equivalent_eulers_0_to_1_vis[:,0],
            'β (deg)': equivalent_eulers_0_to_1_vis[:,1],
            'γ (deg)': equivalent_eulers_0_to_1_vis[:,2],
        })

        mo.output.append(mo.vstack([
            mo.md(f"""The equivalent Euler angles (in degrees) to rotate from the MS tensor of site {euler_selection_start_index} to the MS tensor of site {euler_selection_end_index} according to the chosen convention."""),
            mo.ui.table(
                df_equivalent_eulers,
                page_size=16,
                selection=None,
                format_mapping={col: "{:.1f}".format for col in euler_columns},
                text_justify_columns={col: 'right' for col in euler_columns},
                show_column_summaries=True)
        ]))
    return euler_convention, euler_passive


@app.cell
def _(efg_tensors, filtered_indices, filtered_labels, mo, ms_tensors, np, pd):
    ms_to_efg_eulers = [ms.euler_to(efg) for ms, efg in zip(ms_tensors, efg_tensors)]
    # convert to numpy array and to degrees
    ms_to_efg_eulers = np.degrees(ms_to_efg_eulers)[filtered_indices]
    # Display as dataframe
    euler_columns = ['α (deg)', 'β (deg)', 'γ (deg)']
    df_ms_to_efg_eulers = pd.DataFrame(ms_to_efg_eulers, columns=euler_columns, index=filtered_labels)
    mo.vstack([
        mo.md(r"""### Relative Euler angles between MS and EFG tensors"""),
        mo.md(r"""The table below shows the Euler angles (in degrees) to rotate from the MS tensor to the EFG tensor for each of the selected atoms."""),
        mo.ui.table(
            df_ms_to_efg_eulers,
            page_size=50,
            selection=None,
            format_mapping={col: "{:.1f}".format for col in euler_columns},
            text_justify_columns={col: 'right' for col in euler_columns},
            show_column_summaries=True
        ),
    ])
    return (euler_columns,)


if __name__ == "__main__":
    app.run()
