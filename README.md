Pang
=======
*Pattern-Based Anomaly Detection in Graphs*

Pang is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see licence.txt

-----------------------------------------------------------------------

# Description
Pang is an algorithm which represents and classifies a collection of graphs according to their frequent patterns (subgraphs).


# Organization
This repository is composed of the following elements:
* `requirements.txt` : List of Python packages used in pang.py.
* `PANG.py` : Python script in order to use the algorithm.
* `EMCL.py` : Python script in order to compute the results of the experiments of the ECML paper.
* `ProcessingPattern.py` : Python script in order to compute the number of occurences and the set of induced patterns
* `data` : folder with the input data files. There is one folder for each dataset, which are described in the [Datasets](#datasets) section.


# Installation
You first need to install `python` and the required packages:

1. Install the [`python` language](https://www.python.org)
2. Download this project from GitHub and unzip.
3. Execute `pip install -r requirements.txt` to install the required packages (see also the *Dependencies* Section).

The source code of SPMF in order to use gSpan and cgSpan is available [here](https://www.philippe-fournier-viger.com/spmf/index.php?link=download.php).
SPMF is available in two versions:
* a jar file that can be run from the command line. Actually, this version can be use with gSpan, but not with cgSpan.
* a source code. The installation of this version is more complicated, but it allows to use cgSpan. You can find the instructions [here](https://www.philippe-fournier-viger.com/spmf/how_to_install.php).

In order to use Pang, you need to unzip the datasets in each folder of the `data` folder.

# Use
We provide two scripts to use Pang:
* `ECML.py` : a python script in order to compute the results of the ECML paper.
* `PANG.py` : a python script in order to use Pang with your own data.

## To Replicate the Paper Experiments
In order to use Pang:
1. Open the Python console.
2. Run `EMCL.py`

The script will compute the results of the experiments and save the results associated with Table 2, 5 and 6 in the `results` folder.


## To Apply PANG to Other Data
If you want to use Pang with your own data, you need to create an `XXX` folder in the `data` folder and put your data in it. This folder must contain the following files:
* `XXX_graph.txt` : a file containing the graphs.
* `XXX_label.txt` : a file containing the labels of the graphs.

Then you need to run a script to produce the data files that will be used by Pang:
1. Open the Python console.
2. Run the script `Patterns.sh` in order to create the files `XXX_patterns.txt`.
3. Run `ProcessingPattern.py`with the option `-d XXX` in order to create the files `XXX_mono.txt` and `XXX_iso.txt`.
4. Run `PANG.py` with the option `-d XXX` in order to run Pang on the data `XXX`.

For each value of the parameter `k`, Pang will create a file `KResults.txt` containing the results of the classification and a file `KPatterns.txt` containing the patterns.

## Data Format
We use the same format as SPMF for the graph input files. Each graph is defined as follows:

1. `t # N  N`: graph id
2. `v M L  M`: node id, L: node label
3. `e P Q L P`: source node id, Q: destination node id, L: edge label

For the patterns output files, each pattern contains one more line than the graphs:

4. `x A B C A,B,C` : graphs containing the pattern

## Datasets
The datasets used in the paper are available in the `data` folder. The following datasets are available:
* `MUTAG` : MUTAG dataset, representing chemical compounds and their mutagenic properties [[D'91](#references)],
* `NCI1` : NCI1 dataset, representing molecules and classified according to carcinogenicity [[W'06](#references)],
* `PTC` : PTC dataset, representing molecules and classified according to carcinogenicity  [[T'03](#references)], 
* `DD` : DD dataset, representing amino acids and their interactions [[D'03](#references)],

Each of these datasets can be found [here](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php).
* `FOPPA` : dataset extracted from FOPPA, a database of French public procurement notices [[P'22](#references)]. 
# Dependencies
Tested with `SPMF` version 2.54, and `python` version 3.6.13 with the following packages:
* [`pandas`](https://pypi.org/project/pandas/): version 1.1.5
* [`numpy`](https://pypi.org/project/numpy/): version 1.19.5
* [`networkx`](https://pypi.org/project/numpy/): version 2.5.1
* [`sklearn`](https://pypi.org/project/numpy/): version 0.24.2
* [`matplotlib`](https://pypi.org/project/numpy/): version 3.3.4
* [`grakel`](https://pypi.org/project/numpy/): version 0.1.8
* [`karateclub`](https://pypi.org/project/numpy/): version 1.3.3
* [`stellargraph`](https://pypi.org/project/numpy/): version 1.2.1


The VF2 and ISMAGS algortihms are included in the [`Networkx` library](https://networkx.org/)

For the baselines:
* The WL and WLOA algorithms are included in the Grakel library, documentation available [here](https://ysig.github.io/GraKeL/0.1a8/benchmarks.html)
* Graph2Vec is included in the karateclub library, documentation available [here](https://karateclub.readthedocs.io/en/latest/)
* DGCNN is included in the stellargraph library, documentation available [here](https://stellargraph.readthedocs.io/en/stable/).
* We use the implementation of CORK from Marisa Thoma. This implementation is available in the `CORKcpp.zip` archive.


# References
* **[P'22]** L. Potin, V. Labatut, R. Figueiredo, C. Largeron, P.-H. Morand. *FOPPA: A database of French Open Public Procurement Award notices*, Technical Report, Avignon University, 2022.  [⟨hal-03796734⟩](https://hal.archives-ouvertes.fr/hal-03796734)
* **[D'91]** A.S. Debnath, R.L. Lopez, G. Debnath, A. Shusterman, C. Hansch. *Structure-
activity relationship of mutagenic aromatic and heteroaromatic nitro compounds.
correlation with molecular orbital energies and hydrophobicity*, Journal of Medic-
inal Chemistry 34(2), 786–797, 1991.
* **[W'06]** N.Wale, G. Karypis. *Comparison of descriptor spaces for chemical compound
retrieval and classification*, 6th International Conference on Data Mining, pp.
678–689, 2006.
* **[T'03]** H . Toivonen, A. Srinivasan, R.D. King, S. Kramer, C. Helma.*Statistical eval-
uation of the predictive toxicology challenge 2000-2001*, Bioinformatics 19(10),
1183–1193, 2003.
* **[D'03]** P.D. Dobson, A.J. Doig. *Distinguishing enzyme structures from non-enzymes
without alignments*, Journal of Molecular Biology 330(4), 771–783 ,2003.
