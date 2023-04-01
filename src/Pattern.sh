#!/bin/bash 

name=$1
input="../data/"$name"/"$name"_graph.txt"
outputGSPAN="../data/"$name"/"$name"_pattern.txt"
outputCGSPAN="../data/"$name"/"$name"_CG.txt"
echo "$input"
java -Xmx8192m -jar spmf.jar run GSPAN $input $outputGSPAN 0.1 10 true false true 
#java -Xmx8192m -jar spmf.jar run CGSPANMNI $input $outputCGSPAN 1 1 true false true 