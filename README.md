# soprano
Soprano - a Python library to crack crystals!

[![Python Tests](https://github.com/CCP-NC/soprano/actions/workflows/python-test.yml/badge.svg)](https://github.com/CCP-NC/soprano/actions/workflows/python-test.yml)
[![PyPI version](https://badge.fury.io/py/Soprano.svg)](https://badge.fury.io/py/Soprano)

## Introduction
Soprano is a Python library developed and maintained by the CCP for NMR Crystallography as a tool to help scientists
working with crystallography and simulations to generate, manipulate, run calculations on and analyse large data sets of
crystal structures, with a particular attention to the output of ab-initio random structure searching, or [AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS). It provides a number of functionalities to help automate many common tasks in computational crystallography.

## How to install
Soprano is now available on the Python Package Index. You can install the latest stable release by using `pip`:

    pip install soprano

This will install Soprano with the latest available versions of ASE and NumPy.

### Installation Options

#### Installing with the latest ASE from git (recommended for magres files with CIF-style labels)
ASE versions 3.23 to 3.25 cannot read magres files with CIF-style labels due to a regression. This will be fixed in ASE 3.26 when it is released. Until then, you can install Soprano together with the latest development version of ASE from git:

    pip install soprano git+https://gitlab.com/ase/ase.git@master

This will ensure full compatibility with magres files using CIF-style labels.


#### For Legacy Systems
If you need compatibility with older systems, you can use the legacy installation which pins ASE to version < 3.23 and NumPy to <2.0:

    pip install soprano[legacy]

**Note:** This is useful for environments where you need more controlled dependency versions.



### Development Installation
To get the latest development version (not guaranteed to be stable) from GitHub:

    git clone https://github.com/CCP-NC/soprano.git
    cd soprano
    pip install -e .

For development purposes, install additional tools:

    pip install -e ".[dev]"

This installs Soprano in development mode, where changes to the code take effect immediately without reinstallation. The `dev` option includes tools for code formatting, linting, and testing. It also includes the latest development version of ASE from git, which is necessary for reading MAGRES files with CIF-style labels. 

You can also combine installation options. For example, to install the development version with the docs dependencies:

    pip install -e ".[dev,docs]"
This installs Soprano in development mode with additional dependencies for building documentation.

These approaches should work even on machines without admin privileges (such as HPC clusters), as long as Python and `pip` are present. You can also use the `--user` flag with `pip` to install packages for your user account only.

## Requirements
Soprano's dependencies are automatically handled by `pip` during installation. The core requirements include:

* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/)
* [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
* [Pandas](https://pandas.pydata.org/) (≥2.0)
* [Matplotlib](https://matplotlib.org/)
* [Spglib](https://spglib.github.io/spglib/) (≥2.4)
* [Pydantic](https://docs.pydantic.dev/) (≥2.0)
* [Click](https://click.palletsprojects.com/) (for CLI functionality)
* [Bottleneck](https://pypi.org/project/Bottleneck/) (≥1.3.6)
* [adjustText](https://github.com/Phlya/adjustText)

Additional, optional dependencies are available through feature sets:

* **docs**: Dependencies for building documentation (`jupyter-book`, `sphinx-click`, etc.)
* **dev**: Dependencies for development (`black`, `flake8`, `pytest`, etc.)
* **legacy**: Pinned versions of ASE and NumPy for compatibility with older systems

## Testing

Soprano uses `pytest` for testing and `hatch` for environment management. To run the test suite:

```bash
# Install hatch first if you don't have it
pip install hatch

# Run tests with the default dev environment
hatch run test:test

# Run tests with legacy dependencies
hatch run legacy:test

# For more verbose output
hatch run test:test -v
# To run tests without the latest ASE (if you have issues with it)
hatch run test:test-no-git
```


For contributors running tests locally with GitHub Actions, consider using [Act](https://github.com/nektos/act).

## Getting started
Soprano ships with several Jupyter notebooks that illustrate its core functionality and how to use it. Being familiar with the use of `ase` - the Atomic Simulation Environment - is a good starting point. To use Jupyter notebooks you only need to have Jupyter installed, then launch the notebook server in the tutorials folder:

    pip install jupyter
    cd tutorials
    jupyter notebook
    
Additional information is available in the auto-generated documentation in the docs folder, and the same information can be retrieved by using the Python `help` function.

## Functionality

Here we show a basic rundown - not by any means exhaustive - of Soprano functionality and features.

### Mass manipulation of structure datasets with AtomsCollection
The AtomsCollection class generalises ASE's Atoms class by treating groups of structures together and making it easier to retrieve information about all of them at once. Combined with the large number of AtomProperties, which extract chemical and structural information and more, it provides a simple, powerful tool to look quickly at the results of an AIRSS search.

### Accurate treatment of periodic boundaries
Many functions in Soprano require to compute interatomic distances, such as when computing bonds, or estimating NMR dipolar couplings. Soprano always takes the utmost care in dealing with periodic boundaries, using algorithms that ensure that the closest periodic copies are always properly accounted for in a fast and efficient way. This approach can also be used in custom functions as the algorithm can be found in the function `soprano.utils.minimum_periodic`.

### Easy processing of NMR parameters and spectral simulations
ASE can read NMR parameters in the `.magres` file format, but Soprano can turn them to more meaningful physical quantities such as isotropies, anisotropies and asymmetries. In addition, with a full database of NMR active nuclei, Soprano can compute quadrupolar and dipolar couplings for specific isotopes. Finally, Soprano can produce a fast approximation of a powder spectrum - both MAS and static - in the diluted atoms approximation, or if that is not enough for your needs, provide an interface to NMR simulation software [Simpson](https://inano.au.dk/about/research-centers-and-projects/nmr/software/simpson).

### Machine learning and phylogenetic analysis
The `soprano.analyse.phylogen` module contains functionality to classify collections of structures based on relevant parameters of choice and identify similarities and patterns using Scipy's hierarchy and k-means clustering algorithms. This can be of great help when analysing collections of potential crystal structure looking for polymorphs, finding defect sites, or analysing disordered systems.

### HPC Submitters
Soprano provides a Submitter class, which can be inherited from by people with some experience in Python coding to create their own scripts running as background processes and able to process large amounts of calculations automatically. Files can be copied, sent to remote HPC machines, submitted for calculations to any of the major queue managing systems, and then downloaded back to the local machine - all or just the significant results, if space is an issue. While not the most user-friendly functionality provided by Soprano, Submitters have the potential to be extremely powerful tools that can save a lot of time when working with large batches of computations.
