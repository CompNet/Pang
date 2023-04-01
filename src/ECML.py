# Reproduce ECML results
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
import sys, getopt

from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC

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
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],feature=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'feature': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,noms

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
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],feature=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'feature': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,numberoccurences


def computeScoreMono(keep,labels,id_graphs,TAILLEPATTERN):
    """ this function computes the discrimination score of each pattern using the Binary Criterion"""
    """ Input : keep (list of int) : the list of patterns to keep
                labels (list of int) : the list of labels of the graphs
                id_graphs (list of list of int) : the list of occurences of each pattern
                TAILLEPATTERN (int) : the number of patterns
        Output : diffDelta (list of int) : the list of discrimination scores of each pattern
    """
    NbRed = sum(labels)
    NbNonRed = len(labels)-NbRed
    diffDelta=np.zeros(TAILLEPATTERN)
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                    if j in keep:
                            aa=id_graphs[i][j]
                            if labels[aa]==0:
                                        diffDelta[i]=diffDelta[i]+1
                            elif labels[aa]==1:
                                        diffDelta[i]=diffDelta[i]-1
        diffDelta[i] = abs(diffDelta[i])
    return diffDelta
    
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
    for i in range(TAILLEPATTERN):
        tailleRed.append(0)
        tailleNonRed.append(0)
    ##Delta Criterion
    for i in range(len(diffDelta)):
        if len(id_graphs[i])>0:
                for j in range(len(id_graphs[i])):
                    if j in keep:
                        if labels[id_graphs[i][j]]==0:
                                diffDelta[i]=diffDelta[i]+occurences[i][j]
                                tailleNonRed[i]=tailleNonRed[i]+occurences[i][j]
                        elif labels[id_graphs[i][j]]==1:
                                diffDelta[i]=diffDelta[i]-occurences[i][j]
                                tailleRed[i]=tailleRed[i]+occurences[i][j]
        diffDelta[i] = abs(diffDelta[i])
    return diffDelta
    
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
    for i in range(K):
        if sum(diff)==0:
            break
        bestScore = np.max(diff)
        bestPattern = np.argmax(diff)
        keepPatterns.append(bestPattern)
        diff[bestPattern]=0
    vectorialRep = []
    newLabels = []
    for j in range(LENGTHGRAPH):#330
            vectorialRep.append([])
            for k in keepPatterns:
                if j in keep and j in id_graphs[k]:
                    for t in range(len(id_graphs[k])):
                        if id_graphs[k][t]==j:
                            if numberoccurences==None:
                                occu=1
                            else:
                                occu = numberoccurences[k][t]
                    vectorialRep[j].append(occu)
                else:
                    vectorialRep[j].append(0)
    
    X=[]
    Y=[]
    for i in range(len(labels)):
        if i in keep:
            X.append(vectorialRep[i])
            Y.append(labels[i])
    return X,Y

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
        for l in train_index:
            X_train.append(X[l])
            y_train.append(Y[l])
        for l in test_index:
            X_test.append(X[l])
            y_test.append(Y[l])
        #fit the classifier on the train set
        classifier.fit(X_train,y_train)
        #predict the labels of the test set
        y_predict = classifier.predict(X_test)
        #find the F1 score of each class on the test set
        # use the function for finding the f score between two labels sets 
        # using sklearn.metrics
        
        F1_score0.append(metrics.f1_score(y_test, y_predict, average=None)[0])
        F1_score1.append(metrics.f1_score(y_test, y_predict, average=None)[1])
    #compute the mean and standard deviation of the F1 score of each class
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    print(F1_score1_mean)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def pangProcessing(Ks,keep,labels,id_graphs_mono,id_graphs_iso,occurences_mono,occurences_iso,LENGTHPATTERN,LENGTHGRAPHS):
    cv = StratifiedKFold(n_splits=10,shuffle=True)
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,LENGTHPATTERN)
    scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,LENGTHPATTERN)
    scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,LENGTHPATTERN)
    for K in Ks:
        X_GenBin,Y = KVector(keep,K,scoresGenBin,id_graphs_mono,None,LENGTHGRAPHS,labels)
        X_GenOcc,Y = KVector(keep,K,scoresGenOcc,id_graphs_mono,occurences_mono,LENGTHGRAPHS,labels)
        X_IndBin,Y = KVector(keep,K,scoresIndBin,id_graphs_iso,None,LENGTHGRAPHS,labels)
        X_IndOcc,Y = KVector(keep,K,scoresIndOcc,id_graphs_iso,occurences_iso,LENGTHGRAPHS,labels)
        results = np.zeros((4,2,2))
        representations = [[X_GenBin,Y],[X_GenOcc,Y],[X_IndBin,Y],[X_IndOcc,Y]]
        for i in range(len(representations)):
            results[i][0][0],results[i][0][1],results[i][1][0],results[i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=100))
    return results

from karateclub import Graph2Vec
from grakel import graph_from_networkx
from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, WeisfeilerLehmanOptimalAssignment

def Baselines(index,DATASET,Graphes,cv,labels,results):
    """ this function computes the baseline results for the graph classification task
        baselines are : WL, WLOA, Graph2Vec, DGCNN.""" 
    dicoC = {"MUTAG":1000,"NCI1":100,"NCI109":1,"PTC":100,"DD":1,"FOPPA":100}
    model=Graph2Vec(attributed=True,epochs=1000)
    fitGraph = copy.deepcopy(Graphes)
    model.fit(fitGraph)
    GraphesB=model.get_embedding()  
    
    #create 6 numpy arrays with ten zeros.
    #each numpy array will contain the F1 score of each class for each baseline
    f0WL = np.zeros(10)
    f1WL = np.zeros(10)
    f0WLOA = np.zeros(10)
    F1WLOA = np.zeros(10)
    F0G2V = np.zeros(10)
    F1G2V = np.zeros(10)
    i=-1
    for train_index, test_index in cv.split(Graphes,labels):
        i=i+1
        X_train=[]
        y_train=[]
        X_test=[]
        y_test=[]
        g2v_train=[]
        g2v_test=[]
        for l in train_index:
            X_train.append(Graphes[l])
            y_train.append(labels[l])
            g2v_train.append(GraphesB[l])
        for l in test_index:
            X_test.append(Graphes[l])
            y_test.append(labels[l])
            g2v_test.append(GraphesB[l])
        G_train = graph_from_networkx(X_train, node_labels_tag='feature',edge_labels_tag='feature')
        G_test = graph_from_networkx(X_test, node_labels_tag='feature',edge_labels_tag='feature')
        
        #########WL
        clf=SVC(C=dicoC[DATASET],kernel='linear')
        gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        f0WL[i]=metrics.f1_score(y_test, y_pred, average=None)[0]
        f1WL[i]=metrics.f1_score(y_test, y_pred, average=None)[1]

        #########WLOA
        clf=SVC(C=dicoC[DATASET],kernel='linear')
        G_train = graph_from_networkx(X_train, node_labels_tag='feature',edge_labels_tag='feature')
        G_test = graph_from_networkx(X_test, node_labels_tag='feature',edge_labels_tag='feature')
        gk = WeisfeilerLehmanOptimalAssignment(n_iter=2)
        K_train = gk.fit_transform(G_train)
        K_test = gk.transform(G_test)
        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)
        # Computes and prints the classification accuracy
        f0WLOA[i]=metrics.f1_score(y_test, y_pred, average=None)[0]
        F1WLOA[i]=metrics.f1_score(y_test, y_pred, average=None)[1]
        
        ###Graph2Vec
        clf=SVC(C=dicoC[DATASET])
        clf.fit(g2v_train, y_train)
        y_pred = clf.predict(g2v_test)
        F0G2V[i]=metrics.f1_score(y_test, y_pred, average=None)[0]
        F1G2V[i]=metrics.f1_score(y_test, y_pred, average=None)[1]
    
    #compute the mean of the F1 score of each class for each baseline
    #compute the standard deviation of the F1 score of each class for each baseline
    #store the results in the results array following the order 
    #first index: 0 for WL, 1 for WLOA, 2 for Graph2Vec, 3 for DGCNN
    #second index : the class
    #third index : 0 for the mean, 1 for the standard deviation
    results[index][6][0][0]=np.mean(f0WL)
    results[index][6][0][1]=np.std(f0WL)
    results[index][6][1][0]=np.mean(f1WL)
    results[index][6][1][1]=np.std(f1WL)
    results[index][7][0][0]=np.mean(f0WLOA)
    results[index][7][0][1]=np.std(f0WLOA)
    results[index][7][1][0]=np.mean(F1WLOA)
    results[index][7][1][1]=np.std(F1WLOA)
    results[index][8][0][0]=np.mean(F0G2V)
    results[index][8][0][1]=np.std(F0G2V)
    results[index][8][1][0]=np.mean(F1G2V)
    results[index][8][1][1]=np.std(F1G2V)
    results[index][9][0][0]=0
    results[index][9][0][1]=0
    results[index][9][1][0]=0
    results[index][9][1][1]=0
    return results


import stellargraph as sg

def Table2():
    """ this function computes the results of the table 1 of the paper
        results are saved in a csv file in the folder results"""
    DATASETS = ["MUTAG"]
    Ks = {"MUTAG": 150, "NCI1": 3, "DD": 3, "PTC": 150, "FOPPA": 500}
    results = np.zeros((len(DATASETS),10,2,2))
    for DATASET in DATASETS:
        arg=DATASET
        folder="../data/"+str(arg)+"/"
        FILEGRAPHS=folder+str(arg)+"_graph.txt"
        FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
        FILEMONOSET=folder+str(arg)+"_mono.txt"
        FILEISOSET=folder+str(arg)+"_iso.txt"
        FILELABEL =folder+str(arg)+"_label.txt"
        FILECGGRAPH =folder+str(arg)+"_CG.txt"
        FILECGMONO =folder+str(arg)+"_CG_MONO.txt"
        GRAPHLENGTH=read_Sizegraph(FILEGRAPHS)
        PATTERNLENGTH=read_Sizegraph(FILESUBGRAPHS)
        CGLENGTH=read_Sizegraph(FILECGGRAPH)
        print("DATASET : "+str(arg))

        Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,GRAPHLENGTH)
        Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,PATTERNLENGTH)
        xx,id_graphs_mono,occurences_mono = load_patterns(FILEMONOSET,PATTERNLENGTH)
        xx,id_graphs_iso,occurences_iso = load_patterns(FILEISOSET,PATTERNLENGTH)
        cgSubgraphs,id_graphs_cg,noms = load_graphs(FILECGGRAPH,CGLENGTH)
        xx,id_cg_mono,occurences_cg_mono = load_patterns(FILECGMONO,CGLENGTH)
        
        
        labels = readLabels(FILELABEL)
        keep = graphKeep(PatternsRed,labels)
        cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
        scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,PATTERNLENGTH)
        scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,PATTERNLENGTH)
        scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,PATTERNLENGTH)
        scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,PATTERNLENGTH)
        scoresCloBin = computeScoreMono(keep,labels,id_graphs_cg,CGLENGTH)
        scoresCloOcc = computeScoreOccurences(keep,labels,id_graphs_cg,occurences_cg_mono,CGLENGTH)
        X_GenBin,Y = KVector(keep,Ks[DATASET],scoresGenBin,id_graphs_mono,None,GRAPHLENGTH,labels)
        X_GenOcc,Y = KVector(keep,Ks[DATASET],scoresGenOcc,id_graphs_mono,occurences_mono,GRAPHLENGTH,labels)
        X_IndBin,Y = KVector(keep,Ks[DATASET],scoresIndBin,id_graphs_iso,None,GRAPHLENGTH,labels)
        X_IndOcc,Y = KVector(keep,Ks[DATASET],scoresIndOcc,id_graphs_iso,occurences_iso,GRAPHLENGTH,labels)
        X_CloBin,Y = KVector(keep,Ks[DATASET],copy.deepcopy(scoresCloBin),id_graphs_cg,None,GRAPHLENGTH,labels)
        X_CloOcc,Y = KVector(keep,Ks[DATASET],copy.deepcopy(scoresCloOcc),id_graphs_cg,occurences_cg_mono,GRAPHLENGTH,labels) 
        representations = [[X_GenBin,Y],[X_GenOcc,Y],[X_IndBin,Y],[X_IndOcc,Y],[X_CloBin,Y],[X_CloOcc,Y]]
        for i in range(len(representations)):
            #fill in results for :
            # first index : dataset
            #second index : representation
            results[DATASETS.index(DATASET)][i][0][0],results[DATASETS.index(DATASET)][i][0][1],results[DATASETS.index(DATASET)][i][1][0],results[DATASETS.index(DATASET)][i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=100))
        #keep only graphs which are in keep
        Graphs = [Graphes[i] for i in keep]
        results = Baselines(DATASETS.index(DATASET),DATASET,Graphs,cv,Y,results)
        print(results)
    data = pd.DataFrame(index=range(len(results[0])),columns=DATASETS)
    for i in range(len(results[0])):
        for j in DATASETS:
            data[j][i]=str(results[DATASETS.index(j)][i][1][0])[0:4]+" ("+str(results[DATASETS.index(j)][i][1][1])[0:4]+")"
    data.to_csv("../results/Table2.csv",index=False)
          
    
        
        
        
        
        



def Table5(Ks,keep,labels,id_graphs_mono,id_graphs_iso,id_graphs_cg,occurences_mono,occurences_iso,occurences_cg,LENGTHPATTERN,LENGTHGRAPHS,cv):
    """ this function computes the results of the table 5 of the paper
        results are saved in a csv file in the folder results"""
    
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    results = np.zeros((len(Ks),1,2,2))
    for K in Ks:
        X_GenBin,Y = KVector(keep,K,copy.deepcopy(scoresGenBin),id_graphs_mono,None,LENGTHGRAPHS,labels)
        representations = [[X_GenBin,Y]]
        for i in range(len(representations)):
            KValue = Ks.index(K)
            results[KValue][i][0][0],results[KValue][i][0][1],results[KValue][i][1][0],results[KValue][i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=1000))
    data = pd.DataFrame(index=range(len(Ks)),columns=["K","F1_score0","F1_score1"])
    for i in range(len(results)):
        data["K"][i] = Ks[i]
        data["F1_score0"][i] = str(results[i][0][0][0])[0:4] +" ("+str(results[i][0][0][1])[0:4]+")"
        data["F1_score1"][i] = str(results[i][0][1][0])[0:4] +" ("+str(results[i][0][1][1])[0:4]+")"
    data.to_csv("../results/Table5.csv",index=False)
    return results



#####################################
def Table6(Ks,keep,labels,id_graphs_mono,id_graphs_iso,id_graphs_cg,occurences_mono,occurences_iso,occurences_cg,LENGTHPATTERN,LENGTHGRAPHS,LENGTHCG,cv):
    """ This function computes the results of Table 6 in the paper
    Results are saved in the folder results on a csv file"""
    
    scoresGenBin = computeScoreMono(keep,labels,id_graphs_mono,LENGTHPATTERN)
    scoresGenOcc = computeScoreOccurences(keep,labels,id_graphs_mono,occurences_mono,LENGTHPATTERN)
    scoresIndBin = computeScoreMono(keep,labels,id_graphs_iso,LENGTHPATTERN)
    scoresIndOcc = computeScoreOccurences(keep,labels,id_graphs_iso,occurences_iso,LENGTHPATTERN)
    scoresCloBin = computeScoreMono(keep,labels,id_graphs_cg,LENGTHCG)
    scoresCloOcc = computeScoreOccurences(keep,labels,id_graphs_cg,occurences_cg,LENGTHCG)
    
    results = np.zeros((len(Ks),6,2,2))
    namesRepresentation=["GenBin","GenOcc","IndBin","IndOcc","CloBin","CloOcc"]
    for K in Ks:
        X_GenBin,Y = KVector(keep,K,copy.deepcopy(scoresGenBin),id_graphs_mono,None,LENGTHGRAPHS,labels)
        X_GenOcc,Y = KVector(keep,K,copy.deepcopy(scoresGenOcc),id_graphs_mono,occurences_mono,LENGTHGRAPHS,labels)
        X_IndBin,Y = KVector(keep,K,copy.deepcopy(scoresIndBin),id_graphs_iso,None,LENGTHGRAPHS,labels)
        X_IndOcc,Y = KVector(keep,K,copy.deepcopy(scoresIndOcc),id_graphs_iso,occurences_iso,LENGTHGRAPHS,labels)
        X_CloBin,Y = KVector(keep,K,copy.deepcopy(scoresCloBin),id_graphs_cg,None,LENGTHGRAPHS,labels)
        X_CloOcc,Y = KVector(keep,K,copy.deepcopy(scoresCloOcc),id_graphs_cg,occurences_cg,LENGTHGRAPHS,labels) 
        representations = [[X_GenBin,Y],[X_GenOcc,Y],[X_IndBin,Y],[X_IndOcc,Y],[X_CloBin,Y],[X_CloOcc,Y]]
        for i in range(len(representations)):
            KValue = Ks.index(K)
            results[KValue][i][0][0],results[KValue][i][0][1],results[KValue][i][1][0],results[KValue][i][1][1] = cross_validation(representations[i][0],representations[i][1],cv,SVC(C=1000))
    data = pd.DataFrame(index=range(len(representations)),columns=["Representation","F1_score0","F1_score1"])
    for i in range(len(results[0])):
        data["Representation"][i] = str(namesRepresentation[i])
        data["F1_score0"][i] = str(results[0][i][0][0])[0:4] +" ("+str(results[0][i][0][1])[0:4]+")"
        data["F1_score1"][i] = str(results[0][i][1][0])[0:4] +" ("+str(results[0][i][1][1])[0:4]+")"
    data.to_csv("../results/Table6.csv",index=False)
    return results


if __name__ == '__main__':
    print("Computing results for Table 2")
    Table2()
    print("Computing results for Table 5")
    arg="FOPPA"
    folder="../data/"+str(arg)+"/"
    FILEGRAPHS=folder+str(arg)+"_graph.txt"
    FILESUBGRAPHS=folder+str(arg)+"_pattern.txt"
    FILEMONOSET=folder+str(arg)+"_mono.txt"
    FILEISOSET=folder+str(arg)+"_iso.txt"
    FILELABEL =folder+str(arg)+"_label.txt"
    GRAPHLENGTH=read_Sizegraph(FILEGRAPHS)
    PATTERNLENGTH=read_Sizegraph(FILESUBGRAPHS)
    FILECGGRAPH =folder+str(arg)+"_CG.txt"
    FILECGMONO =folder+str(arg)+"_CG_MONO.txt"
    CGLENGTH=read_Sizegraph(FILECGGRAPH)
    """loading graphs"""
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,GRAPHLENGTH)
    """loading patterns"""
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,PATTERNLENGTH)
    
    """loading processed patterns"""
    xx,id_graphs_mono,occurences_mono = load_patterns(FILEMONOSET,PATTERNLENGTH)
    xx,id_graphs_iso,occurences_iso = load_patterns(FILEISOSET,PATTERNLENGTH)
    cgSubgraphs,id_graphs_cg,noms = load_graphs(FILECGGRAPH,CGLENGTH)
    xx,id_cg_mono,occurences_cg_mono = load_patterns(FILECGMONO,CGLENGTH)
    labels = readLabels(FILELABEL)
    keep = graphKeep(PatternsRed,labels)
    cv = StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
    table5 = Table5([10,50,100,150,15793],keep,labels,id_graphs_mono,id_graphs_iso,id_graphs_cg,occurences_mono,occurences_iso,occurences_cg_mono,PATTERNLENGTH,GRAPHLENGTH,cv)
    print("Computing results for Table 6")
    table5 = Table6([100],keep,labels,id_graphs_mono,id_graphs_iso,id_graphs_cg,occurences_mono,occurences_iso,occurences_cg_mono,PATTERNLENGTH,GRAPHLENGTH,CGLENGTH,cv)