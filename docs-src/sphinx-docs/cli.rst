Soprano Command Line Interface
=======================================================

.. click:: soprano.scripts.cli:cli
  :prog: soprano
  :show-nested:


Examples
==========
* Extract a full summary (will look for both EFG and MS data):
    ``soprano nmr seedname.magres``
* Output summary to a CSV file:
    ``soprano nmr seedname.magres -o summary.csv``
* Output summary to a JSON file:
    ``soprano nmr seedname.magres -o summary.json``
* Extract a full summary for multiple files:
    ``soprano nmr *.magres``
* Extract a full summary for multiple files, merging into one table:
    ``soprano nmr *.magres --merge``
* Extract just the MS data:
    ``soprano nmr seedname.magres -p ms``
* Extract just the MS data for Carbon:
    ``soprano nmr seedname.magres -p ms -s C``
* Or just the first 4 Carbon atoms:
    ``soprano nmr seedname.magres -p ms -s C.1-4``
* Extract just the MS data for Carbon and Nitrogen:
    ``soprano nmr seedname.magres -p ms -s C,N``
* Extract just MS data for the sites with label H1a:
    ``soprano nmr seedname.magres -p ms -s H1a``
* Set chemical shift references and gradients (non-specified references are set to zero and non-specified gradients are set to -1):
    ``soprano nmr seedname.magres -p ms --references C:170,H:100 --gradients C:-1,H:-0.95``
* Set custom isotope
    ``soprano nmr seedname.magres -p efg --isotopes 13C,2H``
* If you want to get only the unique sites in the structure, use the reduce option:
    ``soprano nmr seedname.magres -r``






