
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
import copy
import networkx as nx
import numpy as np
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm

from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def read_Sizegraph(fileName):
    """Read the number of graphs in a file.
    Input: fileName (string) : the name of the file
    Ouptut: TAILLE (int) : the number of graphs in the file"""
    
    file = open(fileName, "r")
    nbGraph=0
    for line in file:
       if line[0]=="t":
            nbGraph=nbGraph+1
    return nbGraph

def load_graphs(fileName,TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    nom (list of string) : the list of names of each graph)"""
    
    nbV=[]
    nbE=[]
    numbers = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
                nbV.append(len(labelVertices[compteur-1]))
                nbE.append(len(labelEdges[compteur-1]))
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            #for j in range(1,len(b)-1):
            for j in range(1,len(b)-1):
                if not(b[j]=="#"):
                    val = b[j]
                    val = re.sub("\n","",val)
                    val = int(val)
                    temp.append(val)
            numbers[numero]=temp  
    noms.append(temptre)
    nbV.append(len(labelVertices[compteur-1]))
    nbE.append(len(labelEdges[compteur-1]))
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            #tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,noms

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
def estimate_importance(features, labels):
	clf = ExtraTreesClassifier()
	clf = clf.fit(features, labels)
	return clf.feature_importances_


def compute_results(features_train, labels_train, features_test, labels_test):
	clf = SVC(class_weight='balanced')
	clf = CalibratedClassifierCV(clf)
	clf.fit(features_train,labels_train)
	res = clf.predict(features_test)

	ok = 0
	fp, fn = 0, 0
	tp, tn = 0, 0
	nAbuses = 0
	f1 = 0
	for r in range(len(res)):
		if res[r] == labels_test[r]:
			ok += 1
		if labels_test[r] == 1:
			if res[r] == 1:
				tp += 1
			if res[r] == 0:
				fn += 1
			nAbuses += 1
		if labels_test[r] == 0:
			if res[r] == 1:
				fp += 1
			if res[r] == 0:
				tn += 1

	rec = tp / float(nAbuses)
	pre = tp / float(tp + fp)
	if pre + rec > 0:
		f1 = 2 * (pre * rec) / float(pre + rec)
	return f1


def load_patterns(fileName,TAILLE):
    """ This function loads the post-processed patterns, i.e with occurences.
    fileName (string) : the name of the file
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of patterns
            numbers (list of list of int) : the list of occurences of each graph
            numberoccurences (list of list of int) : the list of occurences of each pattern
    """
    numbers = []
    numberoccurences = []
    numbercoverage = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
        numberoccurences.append([])
        numbercoverage.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            tempOccu = []
            tempCoverage = []
            for j in range(1,len(b)-1):
                val = b[j]
                val = re.sub("\n","",val)
                if not(val=="#" or val==""):
                    val = str(val).split("/")
                    numeroGraph = int(val[0])
                    val = str(val[1]).split(":")
                    coverage=1
                    if len(val)>1:
                        coverage = float(val[1])
                    occurences = int(float(val[0]))
                    temp.append(numeroGraph)
                    tempOccu.append(occurences)
                    tempCoverage.append(coverage)
            numbers[numero]=temp 
            numberoccurences[numero]=tempOccu
            numbercoverage[numero]=tempCoverage
    noms.append(temptre)
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,numberoccurences


## create a function which compute the closed patterns of a list of patterns
## we have as input a list of patterns, a list of graphs and for each pattern the list of graphs where it appears


def induced_patterns(patterns,graphs,occurences):
    res=[]
    for i in tqdm.tqdm(range(len(patterns))):
        res.append([])
        p1=patterns[i]
        for j in range(len(occurences[i])):
            bool_induced = False
            g=graphs[occurences[i][j]]
            #check if p1 is a subgraph of g using networkx. Consider nodes and edges labels. We check for induced subgraph
            #use GraphMatcher and consider color as node label and color as edge label
            GM = nx.algorithms.isomorphism.GraphMatcher(g,p1,node_match=nx.algorithms.isomorphism.categorical_node_match('color', None),edge_match=nx.algorithms.isomorphism.categorical_edge_match('color', None))
            #list all isomorphisms
            ss = GM.subgraph_isomorphisms_iter()
            for s in ss:
                if len(s)==0:
                    break
                else:
                    bool_induced = True
                    break
            #print each isomorphism
            if bool_induced:
                res[i].append(occurences[i][j])
    return res
            
def write_induced(fileName,patterns,occurences,listePatt):
    """ This function writes the induced patterns in a file.

    Args:
        patterns (list of networkx graphs): the list of patterns
        occurences (list of list of int): the list of occurences of each pattern
    """
    file=open(fileName,"w")
    for i in range(len(patterns)):
        file.write("t # "+str(i)+" \n")
        if len(occurences[i])>0 and i in listePatt:
            stre="x "
            for j in range(len(occurences[i])):
                stre=stre+str(occurences[i][j])+" " 
            stre=stre+"\n"
            file.write(stre)
       

def closed_patterns(patterns,graphs,occurences):
    ## we create a list of closed patterns, which contains the ids of each pattern
    closed_patterns=[]
    for i in range(len(patterns)):
        closed_patterns.append(i)
    
    #we drop the patterns with no occurences
    for i in range(len(patterns)):
        if len(occurences[i])==0:
            closed_patterns.remove(i)
    print(closed_patterns)
    #for each pattern.
    for i in tqdm.tqdm(range(0,len(patterns))):
        if i in closed_patterns:
            p1 = patterns[i]
            for j in range(len(patterns)):
                if i!=j:
                    if j in closed_patterns:
                        bool_closed=False
                        p2 = patterns[j]
                        GM = nx.algorithms.isomorphism.GraphMatcher(p1,p2,node_match=nx.algorithms.isomorphism.categorical_node_match('color', None),edge_match=nx.algorithms.isomorphism.categorical_edge_match('color', None))
                        #list all isomorphisms
                        ss = GM.subgraph_isomorphisms_iter()
                        for s in ss:
                            if len(s)==0:
                                break
                            else:
                                bool_closed = True
                                break
                        if bool_closed:
                            #if p1 is a subgraph of p2, we check if p1 appears in more graphs than p2
                            if not(len(occurences[j])>len(occurences[i])):
                                #if no, we removep2 from the list of closed patterns
                                closed_patterns.remove(j)       
    return closed_patterns
        
        
        
        

def main(dataset):
    arg=str(dataset)
    folder="../data/"+str(arg)+"/"
    FILEGRAPHS=folder+str(arg)+"_graph.txt"
    FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
    FILEMONOSET=folder+str(arg)+"_mono.txt"
    FILEISOSET=folder+str(arg)+"_iso.txt"
    FILELABEL =folder+str(arg)+"_label.txt"
    TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
    TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
    keep= []
    dele =[]
    for i in range(TAILLEGRAPHE):
        if i not in dele:
            keep.append(i)
            
    """loading graphs"""
    print("Reading graphs")
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    print("Reading patterns")
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)
    xx,id_graphs,xxxx = load_patterns(FILEISOSET,TAILLEPATTERN)

    #res = induced_patterns(Subgraphs,Graphes,id_graphs)
    #res=[]
    res2 = closed_patterns(Subgraphs,Graphes,id_graphs)

    fileName = arg+"_closedInduced.txt"
    write_induced(fileName,Subgraphs,id_graphs,res2)

import sys
import time
#check if we are in main
if __name__ == "__main__":
    #get the dataset name
    start = time.time()
    dataset = sys.argv[1]
    print(dataset)
    main(dataset)
    end = time.time()
    timing = end - start  
    print("Time taken: ", timing)