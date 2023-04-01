#!/bin/bash 

java -Xmx8192m -jar spmf.jar run GSPAN ../data/MUTAG/MUTAG_graph.txt ../data/MUTAG/MUTAG_pattern.txt 0.00001 8 true false true 
java -Xmx8192m -jar spmf.jar run CGSPANSupport ../data/MUTAG/MUTAG_graph.txt ../data/MUTAG/MUTAG_CG.txt 0.00001 8 true false true 
python3 ProcessingPatterns.py -d MUTAG