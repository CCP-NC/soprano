Tutorials
=======================

Here is a list of tutorials you can go through in order to learn the
ropes of how Soprano works, and how it can help you in your work. All
the tutorials are available both as online text and in the form of
[Jupyter Notebooks](http://jupyter.org/) to help you run them
interactively on your computer. 
They can also be run using the free cloud-based virtual machine: Binder wherever you see this symbol: <i class="fas fa-rocket"></i>


-  [](tutorials/00-python-quickstart.ipynb)    
    If you are new to Python, this tutorial will help you get Python
    installed and running on your computer. It has a particular focus on
    getting you set up to do computational materials science. 

-   [](tutorials/01-basic_concepts.ipynb)    
    How to load multiple structures into a single collection, how to
    access them individually or in groups, assign data arrays to them,
    and sort them;

-   [](tutorials/02-generators_properties_calculators.ipynb)    
    In Soprano, Generators are iterators that help you to construct many
    structures quickly - for example, by interpolating linearly between
    two extreme configurations in a number of steps, or by generating
    structures in which a random defect has been injected. If you use
    AIRSS, there are also bindings to its random structure generator,
    buildcell. Properties are classes that help extracting information
    from collections of structures, ranging from the simple (their
    lattice parameters) to the very complex (NMR dipolar couplings,
    Steinhardt bond order parameters, and so on). Finally, Calculators
    are the same as the ones from ASE and provide interfaces to commonly
    used software packages. Here we see how they can be used within
    Soprano;

-   [](tutorials/03-atomselection_transforms.ipynb)    
    Selections are instances of a special class that allows one to pick
    a subset of atoms within a structure, based for example on their
    species, or distance from a given point. They can also be operated
    with using Boolean logic to create complex choices (did you ever
    want to be able to pick all carbon atoms within 2 Angstroms of an
    oxygen atom in your sample in a couple of lines of code? Now you
    can!). Transforms are special properties that use Selections to
    modify a structure. The chosen selection can be rotated, translated,
    reflected and so on, creating new structures;

-   [](tutorials/04-clustering.ipynb)    
    One of the core functionalities of Soprano is
    the \"phylogenetic\" clustering approach. This is an analysis method
    meant to help one to find similar structures among a large number of
    candidates - for example, grouping the output structures of an AIRSS
    run based on which polymorph or phase they ended up converging to.
    The PhylogenCluster class allows to do this by providing a vast
    number of \"genes\" that embody various properties. These properties
    can be calculated on a set of structures and used to classify them
    based on similarity thanks to Scipy\'s implementation of
    hierarchical and k-means clustering. In addition, the user can
    create custom \"genes\" depending on the need. The reason why this
    analysis is called \"phylogenetic\" is that it draws inspiration
    from biology - chemical structures are grouped in trees of
    similarity using a string of information that describes them just
    like animals can be grouped in species, families and so on based on
    the resemblance of their DNA.
-   [](tutorials/05a-nmr-parameters.ipynb)    
    The first of three NMR tutorials, and the one most readers need.
    ASE can read the `.magres` file format, the standard output format
    for CASTEP and Quantum ESPRESSO GIPAW calculations; here we use
    Soprano to turn those tensors into the parameters you compare with
    experiment. These include referenced chemical shifts (from a single
    reference or a linear calibration), shielding anisotropy in the
    Haeberlen and Herzfeld-Berger conventions, quadrupolar couplings
    ($C_Q$ and $\eta_Q$), dipolar and J couplings, a per-site summary
    table, and simple 1D powder spectra in the dilute (non-interacting
    spins) limit.

-   [](tutorials/05b-nmr-2d-spectra.ipynb)    
    Builds on Tutorial 5a to construct predicted 2D correlation maps,
    both heteronuclear (e.g. ¹H–¹³C) and ¹H–¹H
    double-quantum/single-quantum. It covers choosing and weighting
    correlations, averaging over rapidly rotating groups such as CH₃,
    combining 1D and 2D plots, and exporting peaks and contour data to
    external tools.

-   [](tutorials/05c-nmr-tensors-spin-systems.ipynb)    
    For readers who want to run full spin-dynamics simulations. A
    lineshape depends on how the tensors are oriented as well as on
    their principal values, so this tutorial explains tensor ordering
    and Euler angle conventions, then builds spin systems (sites plus
    couplings) from ab initio data and exports them to SIMPSON and
    MRSimulator.

-   [](tutorials/06-defect_calculations.ipynb)    
    A series of new tools
    has been developed recently, dedicated to the specific task of
    creating defect structures, such as interstitials, substitutions, or
    localised additions of one specific atom. This tutorial illustrates
    some examples of how to use these tools and manipulate the resulting
    structures. Combination of these tools has the potential to be a
    very powerful way of creating complex defect structures, such as one
    can need when studying zeolites.
