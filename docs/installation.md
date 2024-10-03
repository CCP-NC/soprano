# Installation


## Dependencies

Soprano requires Python 3 or higher and the libraries Numpy, Scipy and [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/). If you have no experience with the latter, you might want to familiarise yourself before using Soprano - it's an excellent library with support for loading and manipulating lots of chemical structure formats, like .cif, as well as interfacing with many of the most used classical molecular dynamics, tight binding and ab initio simulation software packages.

These dependencies should be install automatically when you install Soprano using pip (see below). If you want to install them manually, you can do so using pip. 

## Virtual environments

It is recommended that you install Soprano in a virtual environment. This is a way of isolating the Python packages you install from the system Python packages, which can be useful if you have multiple projects that require different versions of the same package. You can read more about virtual environments [here](https://docs.python.org/3/tutorial/venv.html).

Popular virtual environment managers include [virtualenv](https://virtualenv.pypa.io/en/latest/) and [conda](https://docs.conda.io/en/latest/).

To set up a new virtual environment using virtualenv, you can do the following:

`virtualenv -p python3 soprano_env`

This will create a new virtual environment called `soprano_env` in the current directory. You can then activate the environment using:

`source soprano_env/bin/activate`

and install Soprano using pip (see below).

To set up a new virtual environment using conda, you can do the following:

`conda create -n soprano_env python=3`

This will create a new virtual environment called `soprano_env` in the current directory. You can then activate the environment using:

`conda activate soprano_env`

and install Soprano using pip (see below).




## Installation using pip

There are several ways to install and work with Soprano depending on your needs and computer setup (e.g. whether you have admin rights or not).

```{note}
If you are using a virtual environment, make sure you have activated it before installing Soprano.
```


### Installation from PyPI (recommended)
You can install the latest stable version of Soprano using pip:   

```pip install soprano```

```{note}
For now Soprano is mainly being developed on the `jkshenton` fork of the repository, which is where the latest features will be added, particularly for NMR-related functionality. For this, see the instructions below for the [bleeding edge-version](#bleeding-edge-version).
```


### Installation from latest source code (git)

You can clone the Soprano repository from Github and install it like this:

`git clone https://github.com/CCP-NC/soprano.git`

and then install it using pip:

`pip install .`

from within the `soprano` directory. 

### Installation from latest source code (zip)

If you don't have git installed, you can grab the source code from Github and install it via a zipped file, e.g. using this `pip` command:

`pip install https://github.com/CCP-NC/soprano/archive/refs/heads/master.zip`

You may want to add in the `--upgrade` flag to make sure you have the latest version.


### Bleeding edge-version
For now Soprano is mainly being developed on the `jkshenton` fork of the repository, which is where the latest features will be added. This should be considered unstable. You can install this version using the following command:

`pip install --upgrade https://github.com/jkshenton/soprano/archive/refs/heads/master.zip`

Changes made to this fork will eventually make it to the main repository (and then to PyPI), but this may take some time.

## Other useful flags
### Installation for development

You can combine any of the above `pip` commands with the `-e` flag to install in editable mode, which means that you can edit the source code and the changes will be reflected in the installed version:

`pip install -e https://github.com/CCP-NC/soprano/archive/refs/heads/master.zip`

If you do this and are working with Jupyter notebooks, it's useful to add in the following lines to the top of your notebook:

```python
%load_ext autoreload
%autoreload 2
```

so that you don't need to restart your kernel every time you make a change to the source code.

### No admin rights

If you don't have admin rights on your computer, you can install Soprano in your home directory by adding the `--user` flag to any of the above `pip` commands. For example:

`pip install --user soprano`


## Testing your installation

You can test your installation by first going into the tests directory:

`cd tests`

and  running the following command:

`python -m unittest *.py --verbose`

If everything is working correctly, you should see a message like this:

```
test_file_not_found (cli_tests.TestCLI) ... ok
test_read_valid_ms (cli_tests.TestCLI) ... ok
test_arrays (collection_tests.TestCollection) ... ok
test_calculator (collection_tests.TestCollection) ... ok
test_chunks (collection_tests.TestCollection) ... ok
test_loadres (collection_tests.TestCollection) ... ok
test_save (collection_tests.TestCollection) ... ok
test_slices (collection_tests.TestCollection) ... ok
test_sorting (collection_tests.TestCollection) ... ok
test_sum (collection_tests.TestCollection) ... ok
test_tree (collection_tests.TestCollection) ... 100% of structures loaded successfully. ok
test_gamma (data_tests.TestData) ... ok
test_quadrupole (data_tests.TestData) ... ok
test_spin (data_tests.TestData) ... ok
test_vdw (data_tests.TestData) ... ok
test_airss (generator_tests.TestGenerate) ... WARNING - The AIRSS generator could not be tested as no AIRSS installation has been found on this system. ok
test_defect (generator_tests.TestGenerate) ... ok
test_linspace (generator_tests.TestGenerate) ... ok
test_molneigh (generator_tests.TestGenerate) ... ok
test_rattle (generator_tests.TestGenerate) ... ok
test_coordhist (gene_tests.TestGenes) ... ok
test_dipolar (nmr_tests.TestNMR) ... ok
test_diprotavg (nmr_tests.TestNMR) ... ok
test_efg (nmr_tests.TestNMR) ... ok
test_shielding (nmr_tests.TestNMR) ... ok
test_tensor (nmr_tests.TestNMR) ... ok
test_cluster (phylogen_tests.TestPhylogen) ... ok
test_customgene (phylogen_tests.TestPhylogen) ... ok
test_gene (phylogen_tests.TestPhylogen) ... ok
test_genefail (phylogen_tests.TestPhylogen) ... ok
test_instantiate (phylogen_tests.TestPhylogen) ... ok
test_loadarray (phylogen_tests.TestPhylogen) ... ok
test_loadgene (phylogen_tests.TestPhylogen) ... ok
test_basicprop (properties_tests.TestPropertyLoad) ... ok
test_dummyprop (properties_tests.TestPropertyLoad) ... ok
test_labelprops (properties_tests.TestPropertyLoad) ... ok
test_linkageprops (properties_tests.TestPropertyLoad) ... ok
test_propertymap (properties_tests.TestPropertyLoad) ... ok
test_remap (properties_tests.TestPropertyLoad) ... ok
test_transformprops (properties_tests.TestPropertyLoad) ... ok
test_randomness (random_tests.TestRandom) ... ok
test_seed (random_tests.TestRandom) ... ok
test_arrays (selection_tests.TestSelection) ... ok
test_basic (selection_tests.TestSelection) ... ok
test_iterate (selection_tests.TestSelection) ... ok
test_mapsel (selection_tests.TestSelection) ... ok
test_operators (selection_tests.TestSelection) ... ok
test_selectors (selection_tests.TestSelection) ... 37 ok
test_queueint (submit_tests.TestSubmit) ... ok
test_symdataset (symmetry_tests.TestSymmetry) ... ok
test_wyckoff (symmetry_tests.TestSymmetry) ... ok
test_merge_tagged_sites (test_cli_utils.TestCLIUtils) ... ok
test_tag_functional_groups (test_cli_utils.TestCLIUtils) ... Averaging over functional groups: CH3 ok
test_abc2cart (utils_tests.TestLatticeMethods) ... ok
test_cart2abc (utils_tests.TestLatticeMethods) ... ok
test_merge_sites (utils_tests.TestMergeSites) ... ok
test_seedname (utils_tests.TestOthers) ... ok
test_specsort (utils_tests.TestOthers) ... ok
test_swing_twist (utils_tests.TestOthers) ... ok
test_min_periodic (utils_tests.TestSupercellMethods) ... ok
test_min_supcell (utils_tests.TestSupercellMethods) ... ok
test_func_interface (xrd_tests.TestXRDCalculator) ... ok
test_lebail_fit (xrd_tests.TestXRDCalculator) ... ok
test_powder_peaks (xrd_tests.TestXRDCalculator) ... ok
test_sel_rules (xrd_tests.TestXRDRules) ... ok

----------------------------------------------------------------------
Ran 65 tests in 0.958s

OK
```


<!--TODO: For more information on contributing to Soprano, see the [contributing guide]() -->

