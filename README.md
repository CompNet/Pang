Pang : Pattern Mining for the Classification of Public Procurement Fraud
-------------------------------------------------------------------------

Pang is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation. For source availability and license information see licence.txt

## Description
Pang is an algorithm which represent and classify a collection of graphs according to their frequent patterns. 

The data available are in the FOPPA repository are extracted from FOPPA, a database of French public procurement. 

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

The VF2 and ISMAGS algortihms are implemented into the Networkx library(https://networkx.org/)
## Data Format
We use the same format as SPMF for the graph input files. Each graph is defined as follow.

1. t # N  N: graph id
2. v M L  M: node id, L: node label
3. e P Q L P: source node id, Q: destination node id, L: edge label

For the patterns output files, each pattern contains one more line than the graphs.

4. x A B C A,B,C : graphs containing the pattern
## Use

In order to use Pang:
1. Open the Python console.
2. Run `PANG.py`.

If you want to use Pang with your own data, you need to create a `XXX` folder in the `data` folder and put your data in it. This folder must contain the following files:
* `XXX_graph.txt` : a file containing the graphs.
* `XXX_pattern.txt` : a file containing the patterns.
* `XXX_label.txt` : a file containing the labels of the graphs.

1. Open the Python console.
2. Run `ProcessingPattern.py`with the option -d XXX in order to create the files XXX_mono.txt and XXX_iso.txt.
3. Run `PANG.py` with the option -d XXX in order to run Pang on the data XXX.

For each value of the parameter `k`, Pang will create a file `KResults.txt` containing the results of the classification and a file `KPatterns.txt` containing the patterns.

## Dependencies
Tested with `SPMF` version 2.54
Tested with `python` version 3.8.0, with the following packages:
* [`pandas`](https://pypi.org/project/pandas/): version 1.3.5
* [`numpy`](https://pypi.org/project/numpy/): version 1.22.4
* [`networkx`](https://pypi.org/project/numpy/): version 2.6.3
* [`sklearn`](https://pypi.org/project/numpy/): version 0.0
* [`matplotlib`](https://pypi.org/project/numpy/): version 3.6.0
