Pang : Pattern Mining for the Classification of Public Procurement Fraud
-------------------------------------------------------------------------

Pang is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see licence.txt

## Description
Pang is an algorithm which represent and classify a collection of graphs according to their frequent patterns.

The data available in the data repository are extracted from FOPPA, a database of French public procurement. 

# Organization
This repository is composed of the following elements:
* `requirements.txt` : List of Python packages used in pang.py.
* `PANG.py` : Python script in order to use the algorithm.
* `ProcessingPattern.py` : Python script in order to compute the number of occurences and the set of induced patterns
* `data` : folder with the input files needed.


# Installation
You first need to install `python` and the required packages:

1. Install the [`python` language](https://www.python.org)
2. Download this project from GitHub and unzip.
3. Execute `pip install -r requirements.txt` to install the required packages (see also the *Dependencies* Section).

The source code of SPMF in order to use TKG is available [here](https://www.philippe-fournier-viger.com/spmf/index.php?link=download.php)

## Use
In order to use Pang:
1. Open the Python console.
2. Run `pang.py`.

## Dependencies
Tested with `SPMF` version 2.54
Tested with `python` version 3.8.0, with the following packages:
* [`pandas`](https://pypi.org/project/pandas/): version 1.3.5
* [`numpy`](https://pypi.org/project/numpy/): version 1.22.4
* [`networkx`](https://pypi.org/project/numpy/): version 2.6.3
* [`sklearn`](https://pypi.org/project/numpy/): version 0.0
* [`matplotlib`](https://pypi.org/project/numpy/): version 3.6.0

## References

[EGC'2023] Potin, L., Figueiredo, R., Labatut, V. & Largeron, C., Extraction de motifs pour la détection d’anomalies
dans des graphes : application à la fraude dans les
marchés publics. My journal, X(X)/XX-XX, 201X. doi: XXXXXXXXXX - ⟨hal-XXXXXXXX⟩

Fournier-Viger, P., Gomariz, A., Gueniche, T., Soltani, A., Wu., C., Tseng, V. S. (2014). SPMF: a Java Open-Source Pattern Mining Library. Journal of Machine Learning Research (JMLR), 15: 3389-3393.
