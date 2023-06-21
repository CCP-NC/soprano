# CLI Cookbook

## NMR data extraction
The `nmr` subcommand has a number of options to extract NMR data from a Magres file. You can see the full help by running `soprano nmr -h`. Here are some common examples:

* Extract a full summary (will look for both EFG and MS data):

    ```soprano nmr seedname.magres```

* Output summary to a CSV file:

    ```soprano nmr seedname.magres -o summary.csv```

* Output summary to a JSON file:

    ```soprano nmr seedname.magres -o summary.json```

* Extract a full summary for multiple files:

    ```soprano nmr *.magres```

* Extract a full summary for multiple files, merging into one table:

    ```soprano nmr *.magres --merge```

* Extract just the MS data:

    ```soprano nmr seedname.magres -p ms```

* Extract just the MS data for Carbon:

    ```soprano nmr seedname.magres -p ms -s C```

* Or just the first 4 Carbon atoms:

    ```soprano nmr seedname.magres -p ms -s C.1-4```

* Extract just the MS data for Carbon and Nitrogen:

    ```soprano nmr seedname.magres -p ms -s C,N```

* Extract just MS data for the sites with label H1a:

    ```soprano nmr seedname.magres -p ms -s H1a```

* Set chemical shift references and gradients (non-specified references are set to zero and non-specified gradients are set to -1):

    ```soprano nmr seedname.magres -p ms --references C:170,H:100 --gradients C:-1,H:-0.95```

* Set custom isotope

    ```soprano nmr seedname.magres -p efg --isotopes 13C,2H```

* By default, Soprano will reduce the structure to the uniques sites (based either on CIF labels or symmetry operations. If you want to disable this, you can use the `--no-reduce` option:

    ```soprano nmr seedname.magres --no-reduce```

* You can construct queries that are applied to all loaded Magres files using the pandas dataframe query syntax. For example, to extract the MS data for all sites with a chemical shielding between 100 and 200 ppm *and* an asymmetry parameter greater than 0.5:

    ```soprano nmr *.magres -p ms --query "100 < MS_shielding < 200 and MS_asymmetry > 0.5"```

## 2D NMR plots

The `plotnmr` subcommand can be used to generate 2D NMR plots from a Magres file. Most of the options are the same as for the `nmr` subcommand in terms of filtering sites, setting references, isotopes etc. You can see the full help by running `soprano plotnmr --help`. 

Here are some common examples:

* Plot proton-proton correlation spectrum:

    ```soprano plotnmr seedname.magres -p 2D -x H -y H```

* Plot C-H correlation spectrum with marker sizes proportional to the dipolar coupling strength:

    ```soprano plotnmr seedname.magres -p 2D -x C -y H --scale-marker-by dipolar```

* Plot the H-H double quantum correlation spectrum:

    ```soprano plotnmr seedname.magres -p 2D -x H -y H --yaxis-order 2Q```

* As previous, but averaging over dynamic CH3 and NH3 sites:

    ```soprano plotnmr seedname.magres -p 2D -x H -y H --yaxis-order 2Q -g CH3,NH3```

* By default, Soprano will reduce the system to the inequivalent sites first (e.g. those with the same CIF label or a symmetrically equivalent position). To prevent this, use the `--no-reduce` option:

    ```soprano plotnmr seedname.magres -p 2D -x H -y H --yaxis-order 2Q -g CH3,NH3 --no-reduce```

* Impose a distance cut-off (in Å) between pairs of sites:

    ```soprano plotnmr seedname.magres -p 2D -x H -y H --yaxis-order 2Q -g CH3,NH3 -r --rcut 3.5```



## Dipolar Couplings

* Extract dipolar couplings between all pairs of sites:

    ```soprano dipolar seedname.magres```

* Extract dipolar couplings between all pairs of sites, outputting to a CSV file:

    ```soprano dipolar seedname.magres -o dipolar.csv```

* Extract dipolar couplings between all pairs of sites, and print out those whose absolute value is greater than 10 kHz:

    ```soprano dipolar seedname.magres --query "abs(D) > 10.0"```
