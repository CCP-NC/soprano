# CLI Cookbook

## NMR data extraction
The `nmr` subcommand has a number of options to extract NMR data from a Magres file. You can see the full help by running `soprano nmr -h`. Here are some common examples:

* Extract a full summary (will look for both EFG and MS data):

    ```bash
    soprano nmr seedname.magres
    ```

* Output summary to a CSV file:

    ```bash
    soprano nmr seedname.magres -o summary.csv
    ```

* Output summary to a JSON file:

    ```bash
    soprano nmr seedname.magres -o summary.json
    ```

* Extract a full summary for multiple files:

    ```bash
    soprano nmr *.magres
    ```

* Extract a full summary for multiple files, merging into one table:

    ```bash
    soprano nmr --merge *.magres
    ```

* Extract just the MS data:

    ```bash
    soprano nmr -p ms seedname.magres
    ```

* Extract just the MS data for Carbon:

    ```bash
    soprano nmr -p ms -s C seedname.magres
    ```

* Or just the first 4 Carbon atoms:

    ```bash
    soprano nmr -p ms -s C.1-4 seedname.magres
    ```

* Extract just the MS data for Carbon and Nitrogen:

    ```bash
    soprano nmr -p ms -s C,N seedname.magres
    ```

* Extract just MS data for the sites with label H1a:

    ```bash
    soprano nmr -p ms -s H1a seedname.magres
    ```

* Set chemical shift references and gradients (non-specified references are set to zero and non-specified gradients are set to -1):

    ```bash
    soprano nmr -p ms --references C:170,H:100 --gradients C:-1,H:-0.95 seedname.magres
    ```

* Set custom isotope

    ```bash
    soprano nmr -p efg --isotopes 13C,2H seedname.magres
    ```

* By default, Soprano will reduce the structure to the uniques sites (based either on CIF labels or symmetry operations. If you want to disable this, you can use the `--no-reduce` option:

    ```bash
    soprano nmr --no-reduce seedname.magres
    ```

* You can construct queries that are applied to all loaded magres files using the pandas dataframe query syntax. For example, to extract the MS data for all H sites with a chemical shielding between 100 and 200 ppm *and* an asymmetry parameter greater than 0.5:

    ```bash
    soprano nmr -s H --query "10 < MS_shielding < 30 and MS_asymmetry > 0.5" *.magres 
    ```

## 2D NMR plots

The `plotnmr` subcommand can be used to generate 2D NMR plots from a magres file. Most of the options are the same as for the `nmr` subcommand in terms of filtering sites, setting references, isotopes etc. You can see the full help by running `soprano plotnmr --help`. 

Here are some common examples:

* Plot proton-proton correlation spectrum:

    ```bash
    soprano plotnmr -p 2D -x H -y H seedname.magres
    ```

* Plot C-H correlation spectrum with marker sizes proportional to the dipolar coupling strength. Plot the chemical shift rather than shielding by supplying reference values:

    ```bash
    soprano plotnmr -x C -y H --scale-marker-by dipolar --references C:180,H:30 seedname.magres
    ```

* As previous, but plot a heatmap and contour lines in addition to the markers:

    ```bash
    soprano plotnmr -x C -y H --scale-marker-by dipolar --references C:180,H:30 --heatmap --contour seedname.magres
    ```

* Plot the H-H double quantum correlation spectrum:

    ```bash
    soprano plotnmr -p 2D -x H -y H --yaxis-order 2Q seedname.magres
    ```

* As previous, but averaging over dynamic CH3 and NH3 sites:

    ```bash
    soprano plotnmr -p 2D -x H -y H --yaxis-order 2Q -g CH3,NH3 seedname.magres
    ```

* By default, Soprano will reduce the system to the inequivalent sites first (e.g. those with the same CIF label or a symmetrically equivalent position). To prevent this, use the `--no-reduce` option:

    ```bash
    soprano plotnmr -p 2D -x H -y H --yaxis-order 2Q -g CH3,NH3 --no-reduce seedname.magres
    ```

* Impose a distance cut-off (in Å) between pairs of sites:

    ```bash
    soprano plotnmr -p 2D -x C -y H --rcut 1.5 seedname.magres
    ```

* Combining several of these options:

    ```bash
    soprano plotnmr -p 2D -x C -y H \
            -g CH3 \
            --rcut 1.5 \
            --scale-marker-by dipolar \
            --no-markers \
            --references C:180,H:30 \
            --heatmap \
            --colormap "viridis" \
            --contour \
            --contour-levels 15 \
            --contour-color "black" \
            --contour-linewidth 0.5 \
            seedname.magres
    ```

## Spin Systems
The `spinsys` subcommand can be used to extract spin systems from a magres file. You can see the full help by running `soprano spinsys --help`. 
If you include the magnetic shielding information, you have to specify the shielding reference values for each isotope you want to include in the spin system.

Here are some examples of how you might use this command:

* Extract a spin system in Simpson format for all the H sites in the magres file:

    ```bash
    soprano spinsys seedname.magres -s H --ref H:30
    ```

* Extract a spin system in MRSimulator format for all the H sites in the magres file:

    ```bash
    soprano spinsys seedname.magres -s H --ref H:30 -f mrsimulator
    ```

* Extract a set of individual spin systems for each of the H sites in the magres file, outputting to separate files:

    ```bash
    soprano spinsys seedname.magres -s H --ref H:30 --split
    ```
* Include the dipolar couplings in the spin system output:

    ```bash
    soprano spinsys seedname.magres -s H --ref H:30 --dip
    ```
* Include only the dipolar couplings between the C and H sites in the spin system output:

    ```bash
    soprano spinsys seedname.magres -s H.1-4 --ref H:30 --dip --select_i C --select_j H
    ```
* Set the isotope to use for the spin system:

    ```bash
    soprano spinsys seedname.magres -s H --ref H:30 -i 2H
    ```


## Dipolar Couplings

* Extract dipolar couplings between all pairs of sites:

    ```bash
    soprano dipolar seedname.magres
    ```

* Extract dipolar couplings between all pairs of sites, outputting to a CSV file:

    ```bash
    soprano dipolar seedname.magres -o dipolar.csv
    ```

* Extract dipolar couplings between all pairs of sites, and print out those whose absolute value is greater than 10 kHz:

    ```bash
    soprano dipolar --query "abs(D) > 10.0" seedname.magres
    ```


## Split up molecules

The `splitmols` command can be used to split up a structure into its components (e.g. molecules, framework) based on a connectivity matrix. You can see the full help by running `soprano splitmols --help`. This should work with structure files in any format that ASE can read (= almost all structure formats).

By default the command will output the components to separate extended xyz files. For example

* Split up a structure into molecules within the same unit cell etc. and output to separate .xyz files:

    ```bash
    soprano splitmols seedname.cif
    ```

* Split up a structure into molecules use the ASE GUI to view the structures (no files are written):

    ```bash
    soprano splitmols seedname.cif --view --no-write
    ```

* Split up a structure into molecules and output to a directory in the CASTEP .cell format:

    ```
        soprano splitmols seedname.cif -o output_directory -f cell
    ```

* Center the molecules in a new cell with a 10 Å vacuum spacing:

    ```bash
    soprano splitmols seedname.cif -c --vacuum 10.0
    ```

* Split a zeolite framework with a molecule in a pore into separate files. Here the `--vdw-scale` option is used to increase the van der Waals radii of the atoms by 30% to ensure that the framework is intact and the molecule is separate. The `--no-cell-indices` option is used to prevent the framework atoms from crossing the cell boundaries. These settings work for the tests/test_data/ZSM-5_withH2O.cif example. In other cases you might need to tweak the vdW values manually using the ` --vdw-custom` flag. Use the `-vvv` verbosity flag to see the vdW radii used.

    ```bash
    soprano splitmols seedname.cif --vdw-scale 1.3 --no-cell-indices
    ```
    
* Split the molecules into a new cell defined manually. We can provide the cell as a single float (= cubic cell with that lattice parameter) or as a string with three floats separated by spaces (e.g. `"10 10 20"` for a 10x10x20 Å cell or `"10 10 10 90 90 90"` for a 10x10x10 Å cell with 90° angles) or as a list of 9 floats (e.g. `"10 0 0 0 10 0 0 0 10"`) for a general cell.

    ```bash
    soprano splitmols seedname.cif --cell "10 10 20"
    ```