
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

import scipy as sp

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

from sklearn import metrics
def compute_results(features_train, labels_train, features_test, labels_test):
	clf = SVC(C=1000)
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
	pre = tp / max(float(tp + fp),1)
	if pre + rec > 0:
		f1 = 2 * (pre * rec) / float(pre + rec)
	return metrics.f1_score(res,labels_test, average=None)[1]


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

def computeCorrelation(patterns,id_graphs,numberoccurences,TAILLEGRAPHE,NAMEFILE):
    MatriceCorrelation = np.zeros((len(patterns),TAILLEGRAPHE))
    for j in range(len(patterns)):
        nbPattern = patterns[j]
        for graphe in range(TAILLEGRAPHE):
            if graphe in id_graphs[nbPattern]:
                if numberoccurences==1:
                    MatriceCorrelation[j][graphe]=1
                else:
                    MatriceCorrelation[j][graphe]=numberoccurences[nbPattern][id_graphs[nbPattern].index(graphe)]
    print(MatriceCorrelation)
    alpha = np.corrcoef(MatriceCorrelation)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.imshow(alpha, interpolation='nearest',label='Correlation')
    plt.colorbar()
    plt.savefig(NAMEFILE+".pdf")
    np.savetxt(NAMEFILE+".txt",MatriceCorrelation)
    

def computeScoreMono(keep,labels,id_graphs,TAILLEPATTERN):
    """ this function computes the discrimination score of each pattern using the Binary Criterion"""
    """ Input : keep (list of int) : the list of patterns to keep
                labels (list of int) : the list of labels of the graphs
                id_graphs (list of list of int) : the list of occurences of each pattern
                TAILLEPATTERN (int) : the number of patterns
        Output : diffDelta (list of int) : the list of discrimination scores of each pattern
    """
    g=5
    nb1 = sum(labels)
    nb0=len(labels)-nb1
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    growthRate=np.zeros(TAILLEPATTERN)
    supportDiffernt=np.zeros(TAILLEPATTERN)
    unusualness=np.zeros(TAILLEPATTERN)
    generalizationQuotient=np.zeros(TAILLEPATTERN)
    OddsRatio=np.zeros(TAILLEPATTERN)
    gain=np.zeros(TAILLEPATTERN)
    supMaxK=np.zeros(TAILLEPATTERN)
    MutualInformation=np.zeros(TAILLEPATTERN)
    ChiSquare=np.zeros(TAILLEPATTERN)
    pvalue=np.zeros(TAILLEPATTERN)
    TruePositiveRate=np.zeros(TAILLEPATTERN)
    FalsePositiveRate=np.zeros(TAILLEPATTERN)
    Strength=np.zeros(TAILLEPATTERN)
    diffDeltaPlus=np.zeros(TAILLEPATTERN)
    diffDeltaMoins=np.zeros(TAILLEPATTERN)
    ##Delta Criterion
    for i in tqdm.tqdm(range(len(diffDelta))):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                            aa=id_graphs[i][j]
                            if labels[aa]==0:
                                        diffDelta[i]=diffDelta[i]+1
                                        diffDeltaMoins[i]=diffDeltaMoins[i]+1
                            elif labels[aa]==1:
                                        diffDelta[i]=diffDelta[i]-1
                                        diffDeltaPlus[i]=diffDeltaPlus[i]+1
        diffDelta[i] = abs(diffDelta[i])
        n11 = diffDeltaPlus[i]
        n12 = diffDeltaMoins[i]
        n1 = n11+n12
        n21 = nb1-n11
        n22 = nb0-n12
        n2=n21+n22
        supDp = diffDeltaPlus[i]/nb1
        supDm = diffDeltaMoins[i]/nb0
        supD = (diffDeltaPlus[i]+diffDeltaMoins[i])/(nb1+nb0)


        ##Growth Rate
        growthRate[i] = max(diffDeltaPlus[i],diffDeltaMoins[i])/min(diffDeltaPlus[i]+1,diffDeltaMoins[i]+1)

        ##Support Difference
        supportDiffernt[i]= abs(diffDelta[i])

        ##Unusualness
        unusualness[i]=(n1/len(labels))*(n11/n1-nb1/len(labels))

        ##Generalization Quotient
        g=1
        generalizationQuotient[i]=n11/(n12+g)

        ##Odds Ratio
        OddsRatio[i]=(supDp/(1-supDp))/(supDm/(1-supDm))

        ##Gain
        gain[i]=diffDelta[i]/(diffDeltaPlus[i]+diffDeltaMoins[i])

        ##Maximal Support

        ##Mutual Information
        

        ##Chi Square
        e11 =((n11+n12)*(n11+n21))/len(labels)
        e21 =((n21+n22)*(n11+n21))/len(labels)
        e12 =((n11+n12)*(n12+n22))/len(labels)
        e22 =((n21+n22)*(n12+n22))/len(labels)

        ChiSquare[i] = 0
        ChiSquare[i]=ChiSquare[i]+(n11-e11)*(n11-e11)/e11
        ChiSquare[i]=ChiSquare[i]+(n21-e21)*(n21-e21)/e21
        ChiSquare[i]=ChiSquare[i]+(n12-e12)*(n12-e12)/e12
        ChiSquare[i]=ChiSquare[i]+(n22-e22)*(n22-e22)/e22

        ##p-value

        ##False Positive Rate
        FalsePositiveRate[i]=n2/n21

        ##True Positive Rate
        TruePositiveRate[i]=n11/n1

        ##Strength
        Strength[i]=(supDp*supDp)/(supDp+supDm)
    
    
    
    
    
    return growthRate,supportDiffernt,unusualness,generalizationQuotient,OddsRatio,TruePositiveRate,FalsePositiveRate,Strength
import tqdm   
###################################
            
def computeScoreOccurences(keep,labels,id_graphs,occurences,TAILLEPATTERN):
    """ this function computes the discrimination score of each pattern using the Integer Criterion"""
    """ Input : keep (list of int) : the list of patterns to keep
                occurences (list of list of int) : the list of number occurences of each pattern in each graph
                labels (list of int) : the list of labels of the graphs
                id_graphs (list of list of int) : the list of graphs containing of each pattern
                TAILLEPATTERN (int) : the number of patterns
        Output : diffDelta (list of int) : the list of discrimination scores of each pattern
    """
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    diffCORK=np.zeros(TAILLEPATTERN)
    tailleRed =[]
    tailleNonRed=[]
    g=5
    nb1 = sum(labels)
    nb0=len(labels)-nb1
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    growthRate=np.zeros(TAILLEPATTERN)
    supportDiffernt=np.zeros(TAILLEPATTERN)
    unusualness=np.zeros(TAILLEPATTERN)
    generalizationQuotient=np.zeros(TAILLEPATTERN)
    OddsRatio=np.zeros(TAILLEPATTERN)
    gain=np.zeros(TAILLEPATTERN)
    supMaxK=np.zeros(TAILLEPATTERN)
    MutualInformation=np.zeros(TAILLEPATTERN)
    ChiSquare=np.zeros(TAILLEPATTERN)
    pvalue=np.zeros(TAILLEPATTERN)
    TruePositiveRate=np.zeros(TAILLEPATTERN)
    FalsePositiveRate=np.zeros(TAILLEPATTERN)
    Strength=np.zeros(TAILLEPATTERN)
    diffDeltaPlus=np.zeros(TAILLEPATTERN)
    diffDeltaMoins=np.zeros(TAILLEPATTERN)
    for i in range(TAILLEPATTERN):
        tailleRed.append(0)
        tailleNonRed.append(0)
    ##Delta Criterion
    for i in tqdm.tqdm(range(len(diffDelta))):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                    if labels[id_graphs[i][j]]==0:
                            diffDelta[i]=diffDelta[i]+occurences[i][j]
                            diffDeltaMoins[i]=diffDeltaMoins[i]+occurences[i][j]
                    elif labels[id_graphs[i][j]]==1:
                            diffDelta[i]=diffDelta[i]-occurences[i][j]
                            diffDeltaPlus[i]=diffDeltaPlus[i]+occurences[i][j]
        diffDelta[i] = abs(diffDelta[i])
        n11 = diffDeltaPlus[i]
        n12 = diffDeltaMoins[i]
        n1 = n11+n12
        n21 = nb1-n11
        n22 = nb0-n12
        n2=n21+n22
        supDp = diffDeltaPlus[i]/nb1
        supDm = diffDeltaMoins[i]/nb0
        supD = (diffDeltaPlus[i]+diffDeltaMoins[i])/(nb1+nb0)
        growthRate[i] = max(diffDeltaPlus[i],diffDeltaMoins[i])/min(diffDeltaPlus[i]+1,diffDeltaMoins[i]+1)
        supportDiffernt[i]= abs(diffDelta[i])
        unusualness[i]=(n1/len(labels))*(n11/(n1+1)-nb1/len(labels))
        generalizationQuotient[i]=n11/(n12+g)
        OddsRatio[i]=(supDp/(1-supDp))/(supDm/(1-supDm))
        gain[i]=diffDelta[i]/(diffDeltaPlus[i]+diffDeltaMoins[i])
        TruePositiveRate[i]=n11/n1
        FalsePositiveRate[i]=n2/n21
        Strength[i]=(supDp*supDp)/(supDp+supDm)
    print("Scores :")
    print(sum(growthRate))
    print(sum(supportDiffernt))
    print(sum(unusualness))
    print(sum(generalizationQuotient))
    return growthRate,supportDiffernt,unusualness,generalizationQuotient,OddsRatio,TruePositiveRate,FalsePositiveRate,Strength
    
def readLabels(fileLabel):
    """ this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1
        
    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""
    
    file=open(fileLabel,"r")
    labels = []
    numero=0
    for line in file:
        line = str(line).split("\t")[0]
        if int(line)==-1:
            labels.append(0)
        elif int(line)>-1:
            labels.append(min(int(line),1))
        numero=numero+1
    return labels

def KVector(keep,K,diff,id_graphs,numberoccurences,LENGTHGRAPH,labels):
    """ this fuction creates the vectorial representation of the graphs
        Input : K (int) : the number of patterns to keep
        diff (list of int) : the list of discrimination scores of each pattern
        id_graphs (list of list of int) : the list of graphs containing of each pattern
        numberoccurences (list of list of int) : the list of number occurences of each pattern in each graph
        LENGTHGRAPH (int) : the number of graphs
        labels (list of int) : the list of labels of the graphs
        
        
        Output : X (list of list of int) : the vectorial representation of the graphs
        Y (list of int) : the list of labels of the graphs""" 
    keepPatterns = []
    for i in tqdm.tqdm(range(K)):
        if sum(diff)==0:
            break
        bestScore = np.max(diff)
        bestPattern = np.argmax(diff)
        keepPatterns.append(bestPattern)
        diff[bestPattern]=0
    vectorialRep = []
    newLabels = []
    c=0
    for j in tqdm.tqdm(range(LENGTHGRAPH)):#330
        if j in keep:
            vectorialRep.append([])
            for k in keepPatterns:
                if j in id_graphs[k]:
                    for t in range(len(id_graphs[k])):
                        if id_graphs[k][t]==j:
                            if numberoccurences==None:
                                occu=1
                            else:
                                occu = numberoccurences[k][t]
                    vectorialRep[c].append(occu)
                else:
                    vectorialRep[c].append(0)
            c=c+1
    X = vectorialRep
    return X,keepPatterns,labels

import sys, getopt

def graphKeep(Graphes,labels):
    """Equilibrate the number of graphs in each class"""
    ### Equilibre dataset
    if len(labels)-sum(labels)>sum(labels):
        minority=1
        NbMino=sum(labels)
    else:
        minority =0
        NbMino=len(labels)-sum(labels)
    keep = []
    count=0
    graphs=[]
    for i in range(len(labels)):
        if labels[i]==minority:
            keep.append(i)
    complete=NbMino
    for i in range(len(labels)):   
        if labels[i]!=minority:
            if count<complete:
                count=count+1
                keep.append(i)
    return keep



def cross_validation(X,Y,cv,classifier):
    #store for each fold the F1 score of each class
    F1_score0 = []
    F1_score1 = []
    # for each fold
    for train_index, test_index in cv.split(X,Y):
        #split the dataset into train and test
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def pangProcessing(Ks,keep,labels,id_graphs_mono,id_graphs_iso,occurences_mono,occurences_iso,LENGTHPATTERN,LENGTHGRAPHS):
    cv = StratifiedKFold(n_splits=10,shuffle=True)
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,LENGTHPATTERN)
    scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,LENGTHPATTERN)
    scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,LENGTHPATTERN)
    for K in Ks:
        X_GenBin = KVector(keep,K,scoresGenBin,id_graphs_mono,None,LENGTHGRAPHS,labels)
        X_GenOcc = KVector(keep,K,scoresGenOcc,id_graphs_mono,occurences_mono,LENGTHGRAPHS,labels)
        X_IndBin = KVector(keep,K,scoresIndBin,id_graphs_iso,None,LENGTHGRAPHS,labels)
        X_IndOcc = KVector(keep,K,scoresIndOcc,id_graphs_iso,occurences_iso,LENGTHGRAPHS,labels)
        results = np.zeros((4,2,2))
        representations = [[X_GenBin,labels],[X_GenOcc,labels],[X_IndBin,labels],[X_IndOcc,labels]]
        for i in range(len(representations)):
            results[i][0][0],results[i][0][1],results[i][1][0],results[i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=0.1))
        outputResults(K,results,"../data/"+str(K)+"results.txt")
    return 0

def outputResults(K,results, output_file):
    """ function to output the results of the cross validation
        Input : K : number of patterns
        results : array containing the results of the cross validation
        output_file : file where the results will be written"""
    f = open(output_file,"w")
    f.write("Results for K = "+str(K)+ " \n")
    for i in range(len(results)):
            f.write("Representation "+str(i)+"\n")
            f.write("F1 score for class 0: "+str(results[i][0][0])+" +/- "+str(results[i][0][1])+"\n")
            f.write("F1 score for class 1: "+str(results[i][1][0])+" +/- "+str(results[i][1][1])+"\n")
    f.close()

def printBestPattern(patterns,labels):
    for i in range(len(patterns)):
        if labels[i]==1:
            print(patterns[i])
def main(argv):
    opts, args = getopt.getopt(argv,"hd:o:k:",["ifile=","ofile="])
    for opt, arg in opts:
      if opt == '-h':
         print ('PANG.py -d <dataset> -k<values of K> -m<mode>')
         sys.exit()
      elif opt in ("-d"):
            folder="../data/"+str(arg)+"/"
            FILEGRAPHS=folder+str(arg)+"_graph.txt"
            FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
            FILEMONOSET=folder+str(arg)+"_mono.txt"
            FILEISOSET=folder+str(arg)+"_iso.txt"
            FILELABEL =folder+str(arg)+"_label.txt"
            TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
            TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)
      elif opt in ("-k"):
         Ks=[]
         temp=arg
         temp=temp.split(",")
         for j in range(len(temp)):
             Ks.append(int(temp[j]))
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
    
    """loading processed patterns"""
    print("Reading processed patterns")
    xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
    xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
    keep = graphKeep(PatternsRed,FILELABEL)
    print("Processing PANG")
    pangProcessing(Ks,keep,labels,id_graphsMono,id_graphsIso,numberoccurencesMono,numberoccurencesIso,TAILLEPATTERN,TAILLEGRAPHE)



import os

def plot_figures(arg, ress, originalSS, feature, scoressss, dataset_name):
    # create the directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    if not os.path.exists("results/" + arg):
        os.makedirs("results/" + arg)

    # plot the figures
    position = []
    position2 = []
    position3 = []
    position4 = []
    for i in range(len(ress[0])):
        #check the position of the feature in the list of features deleted  
        position.append(originalSS[0].index(ress[0][i]))
        position2.append(originalSS[1].index(ress[1][i]))
        position3.append(originalSS[2].index(ress[2][i]))
        position4.append(originalSS[3].index(ress[3][i]))
    print(len(position))
    print(len(position2))
    print(len(position3))
    print(len(position4))
    plt.plot(feature,position,label="GenBin")
    plt.plot(feature,position2,label="GenOcc")
    plt.plot(feature,position3,label="IndBin")
    plt.plot(feature,position4,label="IndOcc") 
    #Name the x axis : features number
    plt.xlabel('Features number')
    plt.ylabel('Step of deletion')
    #Affiche la légende
    plt.legend()
    plt.title('Features importance')

    # save the plot in a specific directory
    name = "results/" + arg + "/" + dataset_name + str(len(ress)) + "NN.pdf"
    plt.savefig(name)

    # plot the second figure
    scoress0 = scoressss[0]
    scoress1 = scoressss[1]
    scoress2 = scoressss[2]
    scoress3 = scoressss[3]
    print(len(scoress0))
    print(len(scoress1))
    print(len(scoress2))
    print(len(scoress3))
    plt.figure()
    plt.xlabel('Number of features')
    plt.ylabel('FA-Score of the anomalous class')
    plt.plot([len(feature)+1]+feature,scoress0,label="GenBin")
    plt.plot([len(feature)+1]+feature,scoress1,label="GenOcc")
    plt.plot([len(feature)+1]+feature,scoress2,label="IndBin")
    plt.plot([len(feature)+1]+feature,scoress3,label="IndOcc")
    plt.legend()

    # save the second plot in a specific directory
    name = "results/" + arg + "/" + dataset_name + str(len(ress)) + "F1Score.pdf"
    plt.savefig(name)


def tableScore(K,patterns,file,h):
    f=open(file,"a")
    eachPatterns = []
    for i in range(len(patterns)):
         for j in range(len(patterns[i])):
             eachPatterns.append(patterns[i][j])
    uniquePatterns=len(set(eachPatterns))
    lenPatterns=len(eachPatterns)
    ratio = uniquePatterns/lenPatterns
    ratio = 1-ratio
    #ne garder que 3 chiffres apres la virgule
    ratio = round(ratio,3)
    #On ecrit la valeur de K
    f.write("K="+str(K)+"\n")
    #Pour chaque score, on ecrit le ratio UniquePatterns/lenPatterns

    for i in range(len(patterns)):
        f.write("K="+str(K)+" : Score "+str(h)+ ": "+str(ratio)+"\n")
    f.close()

def read_cluster_bis(file):
    dicoCluster = {}
    listeClu=[]
    file=pd.read_csv(file,sep=",")
    for i in range(len(file)):
        dicoCluster[file["NumeroPatt"][i]]=int(file["Cluster"][i])
        if int(file["Cluster"][i]) not in listeClu:
            listeClu.append(int(file["Cluster"][i]))
    return dicoCluster,len(listeClu)

def read_cluster(file,dataset,K):
    """ this function reads the file containing the clusters of the patterns
        Input : file (string) : the name of the file containing the clusters
        Output : dicoCluster (dictionary) : the dictionary containing the clusters of the patterns"""
    dicoCluster = {}
    file=pd.read_csv(file,sep=";")
    file=file[file["Dataset"]==dataset]
    file=file[file["K"]==K]
    print(dataset,K)
    NBCLUSTERS = file["NbClusters"].iloc[0] 
    listeCluster = file["Clusters"].iloc[0]
    listeCluster = re.sub("\n","",listeCluster)
    listeCluster = re.sub("\[","",listeCluster)
    listeCluster = re.sub("\]","",listeCluster)
    listeCluster = listeCluster.split(" ")

    listePatterns = file["Patterns"].iloc[0]
    listePatterns = re.sub("\n","",listePatterns)
    listePatterns = re.sub("\[","",listePatterns)
    listePatterns = re.sub("\]","",listePatterns)
    listePatterns = listePatterns.split(",")
    for i in range(len(listePatterns)):
        if listeCluster[i]!="":
            dicoCluster[int(listePatterns[i])]=int(listeCluster[i])
    return dicoCluster,int(NBCLUSTERS)
import matplotlib.pyplot as plt

def proceedAblation(X_train,X_test,y_train,y_test,motifs,NBPATTERNS,NBCLUSTERS,dicoCluster,NamePlotAblationStep1):
    print("NEW ABLATION")
    clustersDejaPris = []
    clustersAnePasPrendre = []
    numbers = [i for i in range(NBCLUSTERS)]
    clusters = []
    for mot in motifs:
         clusters.append(dicoCluster[mot])
    print(clusters)
    # count the number of clusters
    NbClusterInThisCase = len(set(clusters))
    #Re-order the clusters
    setclusters = sorted(set(clusters))
    dicoOrdonate = {}
    compt=0
    for var in setclusters:
        dicoOrdonate[var]=compt 
        compt=compt+1
    for i in range(len(clusters)):
        clusters[i]=dicoOrdonate[clusters[i]]
    print(clusters)
    NbClusterActuel = NbClusterInThisCase
    newDicoCluster = {}
    for key in dicoCluster.keys():
        if key in motifs:
            newDicoCluster[key]=dicoOrdonate[dicoCluster[key]]
    print(newDicoCluster)
    print(dicoCluster)
    res=[]
    scoress=[]
    print("NbClusterInThisCase",NbClusterInThisCase)
    for j in range(0,NbClusterInThisCase):
        print("Clusters Selectionnes")
        print(clustersDejaPris)
        valueee=[]
        X_train =  np.array(X_train)
        X_test =  np.array(X_test)
        use_column = [True for ndx in range(NBPATTERNS)]
        for k in range(len(use_column)):
            if newDicoCluster[motifs[k]] in clustersDejaPris:
                use_column[k]=False
        print(clusters)
        print(use_column)
        base_score = compute_results(X_train[:, use_column],y_train,X_test[:, use_column], y_test)
        scores = []
        scoress.append(base_score)
        featuresCluster = []
        if j==NbClusterInThisCase-1:
            res.append(numbers[0])
            scoress.append(0)  
        else:
            for i in range(NbClusterInThisCase):
                clustersAnePasPrendre = copy.deepcopy(clustersDejaPris)
                if i not in clustersDejaPris:
                    clustersAnePasPrendre.append(i)
                    #create a list of boolean where the features to delete is False
                    #features to delete are the features associated to the cluster i in dicoCluster
                    use_column = [True for ndx in range(NBPATTERNS)]
                    for k in range(len(use_column)):
                        if newDicoCluster[motifs[k]] in clustersAnePasPrendre:
                            use_column[k]=False
                    scores.append(compute_results(X_train[:, use_column],
                                            y_train,
                                            X_test[:, use_column],
                                            y_test))
                    valueee.append(scores[i]-base_score)
                else: 
                    scores.append(-300)
                    valueee.append(-300)
            alpha = sorted(enumerate([s for s in scores]),
                key=lambda ndx_score: ndx_score[1],
                reverse=True)
            print(alpha)
            #delete the features with the lowest importance score                                       
            feat_num = alpha[0][0]
            scoress.append(alpha[0][1])
            res.append(numbers[feat_num])
            #delete from numbers the feature
            clustersDejaPris.append(feat_num)
            NbClusterActuel = NbClusterActuel-1
    return scoress,NbClusterInThisCase,res

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if __name__ == '__main__':
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    dfResu = pd.DataFrame(index=range(0,5000),columns=["Dataset","K","ScoreDiscriminant","TypePattern","TypeRepresentation","NbClustersTotal","NbClustersContenus","Score","ScoreMax","ScoreLast"])
    numero=0
    for arg in ["FOPPA"]:
        folder="../data/"+str(arg)+"/"
        FILEGRAPHS=folder+str(arg)+"_graph.txt"
        FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
        FILEMONOSET=folder+str(arg)+"_mono.txt"
        FILEISOSET=folder+str(arg)+"_iso.txt"
        FILELABEL =folder+str(arg)+"_label.txt"
        FILECLUSTER = "ClustersFOPPA.csv"
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

        """loading processed patterns"""
        print("Reading processed patterns")
        xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
        xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
        KsPossible=[len(id_graphsMono)]     
        labelss = readLabels(FILELABEL)
        keep = graphKeep(PatternsRed,labelss)
        
        labels=[]
        for i in range(len(labelss)):
            if i in keep:
                labels.append(labelss[i])
        print("TAILLE KEEP")
        print(len(keep))
        for NBPATTERNS in KsPossible:
            dicoCluster, NBCLUSTERS = read_cluster_bis(FILECLUSTER)
            importance_feat = [0.0]*NBPATTERNS
            scoresGenBin=[0,0,0,0,0,0,0,0]
            scoresGenOcc=[0,0,0,0,0,0,0,0]
            scoresIndBin=[0,0,0,0,0,0,0,0]
            scoresIndOcc=[0,0,0,0,0,0,0,0]
            scoresGenBin[0],scoresGenBin[1],scoresGenBin[2],scoresGenBin[3],scoresGenBin[4],scoresGenBin[5],scoresGenBin[6],scoresGenBin[7] = computeScoreMono(keep,labelss,id_graphsMono,TAILLEPATTERN)
            #scoresGenOcc[0],scoresGenOcc[1],scoresGenOcc[2],scoresGenOcc[3],scoresGenOcc[4],scoresGenOcc[5],scoresGenOcc[6],scoresGenOcc[7] = computeScoreOccurences(keep,labelss,id_graphsMono,numberoccurencesMono,TAILLEPATTERN)
            #scoresIndBin[0],scoresIndBin[1],scoresIndBin[2],scoresIndBin[3],scoresIndBin[4],scoresIndBin[5],scoresIndBin[6],scoresIndBin[7] = computeScoreMono(keep,labelss,id_graphsIso,TAILLEPATTERN)
            #scoresIndOcc[0],scoresIndOcc[1],scoresIndOcc[2],scoresIndOcc[3],scoresIndOcc[4],scoresIndOcc[5],scoresIndOcc[6],scoresIndOcc[7] = computeScoreOccurences(keep,labelss,id_graphsIso,numberoccurencesIso,TAILLEPATTERN)
            results=[]
            import upsetplot     
            for h in range(len(scoresGenBin)):
                noooo = "FileLogs_"+str(arg)+"_"+str(h)+".txt"
                fileLogs = open(noooo,"w")
                ress=[[],[],[],[]]
                scoressss=[[],[],[],[]]
                originalSS= [[],[],[],[]]
                X_GenBin=[]
                for QQ in range(0,4):
                    motifs=[]
                    if QQ==0:
                        X_GenBin,numbers,labels = KVector(keep,NBPATTERNS,scoresGenBin[h],id_graphsMono,None,TAILLEGRAPHE,labels)
                        #computeCorrelation(numbers,id_graphsMono,1,TAILLEGRAPHE,"Type"+str(QQ)+"Score"+str(h)+str(arg)+"_GenBinMatrix0905")
                        for toto in range(len(numbers)):
                            motifs.append(int(numbers[toto]))
                    if QQ==1:
                        print("Passage")
                        X_GenBin,numbers,labels = KVector(keep,NBPATTERNS,scoresGenOcc[h],id_graphsMono,numberoccurencesMono,TAILLEGRAPHE,labels)
                        #computeCorrelation(numbers,id_graphsMono,numberoccurencesMono,TAILLEGRAPHE,"Type"+str(QQ)+"Score"+str(h)+str(arg)+"_GenOccMatrix0905")
                        for toto in range(len(numbers)):
                            motifs.append(int(numbers[toto]))
                    if QQ==2:
                        X_GenBin,numbers,labels = KVector(keep,NBPATTERNS,scoresIndBin[h],id_graphsIso,None,TAILLEGRAPHE,labels)
                        #computeCorrelation(numbers,id_graphsIso,1,TAILLEGRAPHE,"Type"+str(QQ)+"Score"+str(h)+str(arg)+"_IndBinMatrix0905")
                        for toto in range(len(numbers)):
                            motifs.append(int(numbers[toto]))
                    if QQ==3:
                        X_GenBin,numbers,labels = KVector(keep,NBPATTERNS,scoresIndOcc[h],id_graphsIso,numberoccurencesIso,TAILLEGRAPHE,labels)
                        #computeCorrelation(numbers,id_graphsIso,numberoccurencesIso,TAILLEGRAPHE,"Type"+str(QQ)+"Score"+str(h)+str(arg)+"_IndoccMatrix0905")
                        for toto in range(len(numbers)):
                            motifs.append(int(numbers[toto]))
                    boolean=True
                    for train_index, test_index in cv.split(X_GenBin,labels):
                        X_train=[]
                        X_test=[]
                        y_train=[]
                        y_test=[]
                        for l in train_index:
                            X_train.append(X_GenBin[l])
                            y_train.append(labels[l])
                        for l in test_index:
                            X_test.append(X_GenBin[l])
                            y_test.append(labels[l])
                        #only for the first fold
                        if boolean==True:
                            boolean=False
                            NamePlotAblationStep1 = "AblationStep1_"+str(arg)+"_Score"+str(h)+"_Type"+str(QQ)+"_K"+str(NBPATTERNS)+".pdf"
                            scores,NbClusterInThisCase,res= proceedAblation(X_train,X_test,y_train,y_test,motifs,NBPATTERNS,NBCLUSTERS,dicoCluster,NamePlotAblationStep1)   
                            print(res) 
                            dfResu.iloc[numero]["Dataset"]=arg
                            dfResu.iloc[numero]["K"]=NBPATTERNS
                            if h == 0:
                                 dfResu.iloc[numero]["ScoreDiscriminant"] = "GrowthRate"
                            if h == 1:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "SupportDifferent"
                            if h == 2:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "Unusualness"
                            if h == 3:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "GeneralizationQuotient"
                            if h == 4:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "OddsRatio"
                            if h == 5:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "TruePositiveRate"
                            if h == 6:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "FalsePositiveRate"
                            if h == 7:
                                    dfResu.iloc[numero]["ScoreDiscriminant"] = "Strength"

                            if QQ==0:
                                dfResu.iloc[numero]["TypePattern"]="Extracted"
                                dfResu.iloc[numero]["TypeRepresentation"]="Binary"
                            if QQ==1:
                                dfResu.iloc[numero]["TypePattern"]="Extracted"
                                dfResu.iloc[numero]["TypeRepresentation"]="Occurence"
                            if QQ==2:
                                dfResu.iloc[numero]["TypePattern"]="Induced"
                                dfResu.iloc[numero]["TypeRepresentation"]="Binary"
                            if QQ==3:
                                dfResu.iloc[numero]["TypePattern"]="Induced"
                                dfResu.iloc[numero]["TypeRepresentation"]="Occurence"

                            dfResu.iloc[numero]["Score"]=scores[0]
                            dfResu.iloc[numero]["NbClustersTotal"]=NBCLUSTERS
                            dfResu.iloc[numero]["NbClustersContenus"]=NbClusterInThisCase
                            dfResu.iloc[numero]["ScoreMax"]=max(scores)
                            #pas la derniere valeur, la valeur avant
                            dfResu.iloc[numero]["ScoreLast"]=scores[-2]
                            numero=numero+1
                            dfResu.to_csv("dfResuNEW2.csv",sep=";")
                            
