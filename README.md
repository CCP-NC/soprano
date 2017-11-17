# soprano
Soprano - a Python library to crack crystals!

## Introduction
Soprano is a Python library developed and maintained by the CCP for NMR Crystallography as a tool to help scientists
working with crystallography and simulations to generate, manipulate, run calculations on and analyse large data sets of
crystal structures, with a particular attention to the output of ab-initio random structure searching, or [AIRSS](https://www.mtg.msm.cam.ac.uk/Codes/AIRSS). It provides a number of functionalities to help automate many common tasks in computational crystallography.

## How to install
You can install Soprano simply by cloning this repository and using `pip`:

    git clone https://github.com/CCP-NC/soprano.git
    pip install ./soprano --user
 
This approach should work even on machines on which one does not possess admin privileges (such as HPC clusters), as long as Python and `pip` are present.

## Requirements
Soprano has a few requirements that should be installed automatically by `pip` when used. Installing with `pip` is strongly advised. The core Soprano requirements are:

* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* The [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)

Additional, optional requirements are `pyspglib` (used for spacegroup detection in `soprano.properties.symmetry` and `soprano.calculate.xrd`) and `paramiko` (used for remote SSH operation in `soprano.hpc.submitter`).

## Getting started
Soprano ships with five Jupyter notebooks that illustrate its core functionality and how to use it. Being accustomed already with the use of `ase` - the Atomic Simulation Environment - is a good starting point. To use Jupyter notebooks you only need to have Jupyter installed, then launch the notebook server in the tutorials folder:

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
ASE can read NMR parameters in the `.magres` file format, but Soprano can turn them to more meaningful physical quantities such as isotropies, anisotropies and asymmetries. In addition, with a full database of NMR active nuclei, Soprano can compute quadrupolar and dipolar couplings for specific isotopes. Finally, Soprano can produce a fast approximation of a powder spectrum - both MAS and static - in the diluted atoms approximation, or if that is not enough for your needs, provide an interface to NMR simulation software [Simpson](http://inano.au.dk/about/research-centers/nmr/software/simpson/).

### Machine learning and phylogenetic analysis
The `soprano.analyse.phylogen` module contains functionality to classify collections of structures based on relevant parameters of choice and identify similarities and patterns using Scipy's hierarchy and k-means clustering algorithms. This can be of great help when analysing collections of potential crystal structure looking for polymorphs, finding defect sites, or analysing disordered systems.

### HPC Submitters
Soprano provides a Submitter class, which can be inherited from by people with some experience in Python coding to create their own scripts running as background processes and able to process large amounts of calculations automatically. Files can be copied, sent to remote HPC machines, submitted for calculations to any of the major queue managing systems, and then downloaded back to the local machine - all or just the significant results, if space is an issue. While not the most user-friendly functionality provided by Soprano, Submitters have the potential to be extremely powerful tools that can save a lot of time when working with large batches of computations.
