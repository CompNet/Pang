#!/bin/bash 

name=$1
input="../data/"$name"/"$name"_graph.txt"
outputGSPAN="../data/"$name"/"$name"_pattern.txt"
outputCGSPAN="../data/"$name"/"$name"_CG.txt"
echo "$input"
java -Xmx8192m -jar spmf.jar run GSPAN $input $outputGSPAN 0.9 8 true false true 
java -Xmx8192m -jar spmf.jar run CGSPAN $input $outputCGSPAN 0.9 8 true false true 