Pang v1.0.0
-------------------------------------------------------------------------
*Pattern-Based Anomaly Detection in Graphs*

* Copyright 2021-2023 Lucas Potin & Rosa Figueiredo & Vincent Labatut & Christine Largeron

Pang is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see licence.txt

* **Lab site:** http://lia.univ-avignon.fr
* **GitHub repo:** https://github.com/CompNet/Pang
* **Contact:** Lucas Potin <lucas.potin@univ-avignon.fr>
 

# Description
Pang is an algorithm which represents and classifies a collection of graphs according to their frequent patterns (subgraphs). The detail of this algorithm are described in an article [[P'23](#references)].
This work was conducted in the framework of the DeCoMaP ANR project (Detection of corruption in public procurement markets -- ANR-19-CE38-0004). 

If you use this source code, please cite article [[P'23](#references)]:
```bibtex
@InProceedings{Potin2023c,
  author    = {Potin, Lucas and Figueiredo, Rosa and Labatut, Vincent and Largeron, Christine},
  title     = {Pattern Mining for Anomaly Detection in Graphs: Application to Fraud in Public Procurement},
  booktitle = {Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  year      = {2023},
  volume    = {14174},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  pages     = {69-87},
  doi       = {10.1007/978-3-031-43427-3_5},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-43427-3_5},
}
```

**Content**
* [Organization](#organization)
* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [References](#references)

# Organization
This repository is composed of the following elements:

* `requirements.txt`: List of required Python packages.
* `src`: folder containing the source code
  * `EMCL.py`: script that reproduces the experiments of our paper submitted to ECML PKDD.
  * `PANG.py`: script that implements the Pang method.
  * `ProcessingPattern.py`: script that computes the number of occurences and the set of induced patterns.
  * `Pattern.sh`: script that computes the patterns of a dataset.
  * `CORKcpp.zip`: archive containing the CORK source code (used in `EMCL.py`), cf. Section [Installation](#installation).
* `data`: folder containing the input data. Each subfolder corresponds to a distinct dataset, cf. Section [Datasets](#datasets).
* `results`: files produced by the processing.


# Installation

## Python and Packages
First, you need to install the `Python` language and the required packages:

1. Install the [`Python` language](https://www.python.org)
2. Download this project from GitHub and unzip.
3. Execute `pip install -r requirements.txt` to install the required packages (see also Section [Dependencies](#dependencies)).

## Non-Python Dependencies
Second, one of the dependencies, SPMF, is not a Python package, but rather a Java program, and therefore requires a specific installation process:

* Download its source code on [Philippe Fournier-Viger's website](https://www.philippe-fournier-viger.com/spmf/index.php?link=download.php).
* Follow the installation instructions provided on the [same website](https://www.philippe-fournier-viger.com/spmf/how_to_install.php).

Note that SPMF is available both as a JAR and as source code archive. However, the former does not contain all the features required by Pang, so one should use only the latter.

In order to run the script that reproduces our ECML PKDD experiments, you also need to install CORK. This is done by unzipping the archive `CORKcpp.zip` in the `src` folder. File `Readme` in this archive contains the instruction for compiling the C++ source code.


## Data
Third, you need to set up the data to which you want to apply Pang. This can be the dataset from our paper, in which case you will need to unzip several archives, or your own data, in which case they need to be respect the appropriate format. In both cases, see cf. Section [Usage](#usage).

# Use
We provide two scripts to use Pang:

* `ECML.py`: reproduces the experiments described in our paper submitted to ECML PKDD.
* `PANG.py`: applies Pang in the general case, possibly to your own data.

## To Replicate the Paper Experiments
To replicate the experiments in our Paper[[P'23](#references)], first unzip the provided datasets, and run Pang on them. 

### Data Preparation
To unzip the datasets used in our experiments:

1. Go to the `data` folder.
2. In each subfolder, you will find an archive that you need to unzip.

We retrieved the benchmark datasets from the [SPMF website](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php); they include:
* `MUTAG` : MUTAG dataset, representing chemical compounds and their mutagenic properties [[D'91](#references)]
* `NCI1` : NCI1 dataset, representing molecules and classified according to carcinogenicity [[W'06](#references)]
* `PTC` : PTC dataset, representing molecules and classified according to carcinogenicity [[T'03](#references)] 
* `DD` : DD dataset, representing amino acids and their interactions [[D'03](#references)]

The public procurement dataset contains graphs extracted from the FOPPA database:
* `FOPPA` : dataset extracted from FOPPA, a database of French public procurement notices [[P'23b](#references)].


### Processing
Then, run the appropriate script:

3. Open the Python console.
4. Run `EMCL.py`

The script will compute the results of the experiments and save the results associated with Table 2, 4, 5 and 6 of the paper, in the `results` folder.


## To Apply Pang to Other Data
If you want to use Pang with your own data, you need to set up the data, then identify the patterns, and finally perform the classification.

### Data Preparation
Create an `XXX` folder in the `data` folder (where `XXX` is the name of your dataset), in order to host your data. This folder must contain the following files:

* `XXX_graph.txt` : a file containing all the graphs.
* `XXX_label.txt` : a file indicating the labels (classes) of these graphs.

We use the same format as SPMF for the graph input files, i.e.:

1. `t # N  N`: graph id
2. `v M L  M`: node id, L: node label
3. `e P Q L P`: source node id, Q: destination node id, L: edge label

For information, the files produced by our scripts to list the identified patterns are similar, except they contain an extra line:

4. `x A B C A,B,C` : graphs containing the pattern

The format of the file containing the graph labels is as follows: each line contains an unique integer, corresponding to the label of the graph in the same line in the graph file.

### Processing

Once the data are ready, you need to run a script to identify the patterns, and produce the files required by Pang:

1. Open the `Python` console.
2. Run the script `Patterns.sh` in order to create the files `XXX_patterns.txt`.
3. Run `ProcessingPattern.py`with the option `-d XXX` in order to create the files `XXX_mono.txt` and `XXX_iso.txt`.
4. Run `PANG.py`, specifying both following parameters:
    * `-d XXX`: name of the dataset
    * `-k k`: number of patterns to consider in the PANG representations. User can provide a single value, or a list of values separated by commas.

For each value of the parameter `k`, Pang will create a file `KResults.txt` containing the results of the classification and a file `KPatterns.txt` containing the patterns.


# Dependencies
Tested with `python` version 3.6.13 and the following packages:
* [`pandas`](https://pypi.org/project/pandas/): version 1.1.5
* [`numpy`](https://pypi.org/project/numpy/): version 1.19.5
* [`networkx`](https://pypi.org/project/numpy/): version 2.5.1
* [`sklearn`](https://pypi.org/project/numpy/): version 0.24.2
* [`matplotlib`](https://pypi.org/project/numpy/): version 3.3.4
* [`grakel`](https://pypi.org/project/numpy/): version 0.1.8
* [`karateclub`](https://pypi.org/project/numpy/): version 1.3.3
* [`stellargraph`](https://pypi.org/project/numpy/): version 1.2.1

The VF2 [[C'04](#references)] and ISMAGS [[H'14](#references)] algorithms are included in the [`Networkx` library](https://networkx.org/)

Tested with `SPMF` version 2.54, which implements gSpan [[Y'02](#references)] (to mine frequent patterns) and cgSpan [[S'21](#references)] (closed frequent patterns).

For the ECML PKDD assessment, we use the following algorithms for the sake of comparison:

* The `WL` [[S'11](#references)] and `WL_OA` [[K'16](#references)] algorithms are included in the `Grakel` library, documentation available [here](https://ysig.github.io/GraKeL/0.1a8/benchmarks.html)
* `Graph2Vec` [[N'17](#references)] is included in the `karateclub` library, documentation available [here](https://karateclub.readthedocs.io/en/latest/)
* `DGCNN` [[Z'18](#references)] is included in the `stellargraph` library, documentation available [here](https://stellargraph.readthedocs.io/en/stable/).
* We use the implementation of `CORK` [[T'09](#references)] by Marisa Thoma, which is available [here](http://www.dbs.ifi.lmu.de/~thoma/pub/sam2010/sam2010.zip).

# References
* **[C'04]** L. P. Cordella, P. Foggia, C. Sansone, M. Vento. *A (sub)graph isomorphism algorithm for matching large graphs*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 26(10):1367-1372, 2004. DOI: [10.1109/tpami.2004.75](https://doi.org/10.1109/tpami.2004.75)
* **[D'91]** A. S. Debnath, R. L. Lopez, G. Debnath, A. Shusterman, C. Hansch. *Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. Correlation with molecular orbital energies and hydrophobicity*, Journal of Medicinal Chemistry 34(2):786–797, 1991. DOI: [10.1021/jm00106a046](https://doi.org/10.1021/jm00106a046)
* **[D'03]** P. D. Dobson, A. J. Doig. *Distinguishing enzyme structures from non-enzymes without alignments*, Journal of Molecular Biology 330(4):771–783, 2003. DOI: [10.1016/S0022-2836(03)00628-4](https://doi.org/10.1016/S0022-2836(03)00628-4)
* **[H'14']** M. Houbraken, S. Demeyer, T. Michoel, P. Audenaert, D. Colle, M. Pickavet. *The Index-Based Subgraph Matching Algorithm with General Symmetries (ISMAGS): Exploiting Symmetry for Faster Subgraph Enumeration*, PLoS ONE 9(5):e97896, 2014. DOI: [10.1371/journal.pone.0097896](https://doi.org/10.1371/journal.pone.0097896).
* **[K'16]** N. M. Kriege, P. L. Giscard, R. Wilson. *On Valid Optimal Assignment Kernels and Applications to Graph Classification*, 30th International Conference on Neural Information Processing Systems, pp. 1623–1631, 2016. URL: [here](https://proceedings.neurips.cc/paper_files/paper/2016/hash/0efe32849d230d7f53049ddc4a4b0c60-Abstract.html)
* **[N'17]** A. Narayanan, M. Chandramohan, R. Venkatesan, L. Chen, Y. Liu, S. Jaiswal. *graph2vec: Learning Distributed Representations of Graphs*, 13th International Workshop on Mining and Learning with Graphs, p. 21, 2017. URL: [here](https://arxiv.org/abs/1707.05005)
* **[P'23]** L. Potin, V. Labatut, R. Figueiredo & C. Largeron *Pattern Mining for Anomaly Detection in Graphs: Application to Fraud in Public Procurement*, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), 2023. [⟨hal-04131485⟩](https://hal.archives-ouvertes.fr/hal-04131485)
* **[P'23b]** L. Potin, V. Labatut, P. H. Morand & C. Largeron. *FOPPA: An Open Database of French Public Procurement Award Notices From 2010–2020*, Scientific Data, 2023, 10:303. DOI: [10.1038/s41597-023-02213-z](https://dx.doi.org/10.1038/s41597-023-02213-z) [⟨hal-04101350⟩](https://hal.archives-ouvertes.fr/hal-04101350)
* **[S'11]** N. Shervashidze, P. Schweitzer, E. J. van Leeuwen, K. Mehlhorn, K. M. Borgwardt. *Weisfeiler-Lehman Graph Kernels*, Journal of Machine Learning Research 12:2539–2561, 2011. URL: [here](https://dl.acm.org/citation.cfm?id=2078187)
* **[S'21]** Z. Shaul, S. Naaz. *cgSpan: Closed Graph-Based Substructure Pattern Mining, IEEE International Conference on Big Data, pp. 4989-4998, 2021. DOI: [10.1109/bigdata52589.2021.9671995](https://doi.org/10.1109/bigdata52589.2021.9671995)
* **[T'03]** H. Toivonen, A. Srinivasan, R. D. King, S. Kramer, C. Helma. *Statistical evaluation of the predictive toxicology challenge 2000-2001*, Bioinformatics 19(10):1183–1193, 2003. DOI: [10.1093/bioinformatics/btg130](https://doi.org/10.1093/bioinformatics/btg130)
* **[T'09]** M. Thoma, H. Cheng, A. Gretton, J. Han, H.-P. Kriegel, A. Smola, S. Le, P. S. Yu, X. Yan, K. Borgwardt. *Near-optimal supervised feature selection among frequent subgraphs*, SIAM International Conference on Data Mining, pp. 1076-1087, 2009. DOI: [10.1137/1.9781611972795.92](http://doi.org/10.1137/1.9781611972795.92)
* **[W'06]** N. Wale, G. Karypis. *Comparison of descriptor spaces for chemical compound retrieval and classification*, 6th International Conference on Data Mining, pp. 678–689, 2006. DOI: [10.1007/s10115-007-0103-5](https://doi.org/10.1007/s10115-007-0103-5)
* **[Y'02]** X. Yan, J. Han. *gSpan: Graph-based substructure pattern mining*, IEEE International Conference on Data Mining, pp.721-724, 2002. DOI: [10.1109/ICDM.2002.1184038](https://doi.org/10.1109/ICDM.2002.1184038)
* **[Z'18]** D. Zhang, J. Yin, X. Zhu, C. Zhang. *Network Representation Learning: A Survey*, IEEE Transactions on Big Data 6(1):3–28, 2018. DOI: [10.1109/tbdata.2018.2850013](http://doi.org/10.1109/tbdata.2018.2850013)

