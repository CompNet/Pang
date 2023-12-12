
import xgboost
from sklearn.linear_model import LinearRegression

import shap
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
            FalsePositiveRate[i]=n21/(n2+0.01)

            ##True Positive Rate
            TruePositiveRate[i]=n11/(n1+0.01)

            ##Strength
            Strength[i]=(supDp*supDp)/(supDp+supDm)
    
    
    
    return growthRate,supportDiffernt,TruePositiveRate,FalsePositiveRate,Strength


def MetricOnlyPatterns(metric,patterns):
    return metric[np.array(patterns)],patterns
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
        
    F1_score0_mean = np.mean(F1_score0)
    F1_score0_std = np.std(F1_score0)
    F1_score1_mean = np.mean(F1_score1)
    F1_score1_std = np.std(F1_score1)
    #return the mean and standard deviation of the F1 score of each class
    return F1_score0_mean,F1_score0_std,F1_score1_mean,F1_score1_std


def UniqueOnlyVectors(UniqueID,id_graphs):
    result = []
    for id in UniqueID:
        result.append(id_graphs[id])
    return result
        
def ComputeRepresentation(keepPatterns,id_graphs,labels,LENGTHGRAPH):
    numberoccurences=None
    vectorialRep = []
    newLabels = []
    c=0
    for j in range(LENGTHGRAPH):#330
            if j in keep:
                newLabels.append(labels[j])
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
                vectorialRep[c] = np.array(vectorialRep[c])
                c=c+1
    X = vectorialRep
    return X,newLabels
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
            FILECLOSED =folder+str(arg)+"_CG.txt"
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
    xx,id_graphsClosed,numberoccurencesClosed = load_patterns(FILECLOSED,TAILLEPATTERN)
    keep = graphKeep(PatternsRed,FILELABEL)
    print("Processing PANG")
    pangProcessing(Ks,keep,labels,id_graphsMono,id_graphsIso,numberoccurencesMono,numberoccurencesIso,TAILLEPATTERN,TAILLEGRAPHE)


def createVectorialRepresentation(uniquePatt,NBGRAPHS):
    res = []
    for i in range(len(uniquePatt)):
        res.append([])
        for j in range(NBGRAPHS):
            if j in uniquePatt[i]:
                res[i].append(1)
            else:
                res[i].append(0)
    return res
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



def findUniquePatterns(id_graphsMono):
    setUniquePatterns = set()
    res = []
    for i in id_graphsMono:
        frozeni=frozenset(i)
        if frozeni not in setUniquePatterns:
            setUniquePatterns.add(frozeni)
            res.append(i)    
    return res
        
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

##############################################################################"
# 
def proceedAblation(X_train,X_test,y_train,y_test,motifs,NBPATTERNS,NBCLUSTERS,dicoCluster,NamePlotAblationStep1):
    """ This function proceed to an ablation study on the specified dataset"""
    clustersDejaPris = []
    clustersAnePasPrendre = []
    numbers = [i for i in range(NBCLUSTERS)]
    clusters = []
    for mot in motifs:
         clusters.append(dicoCluster[mot])
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
    NbClusterActuel = NbClusterInThisCase
    newDicoCluster = {}
    for key in dicoCluster.keys():
        if key in motifs:
            newDicoCluster[key]=dicoOrdonate[dicoCluster[key]]
    res=[]
    scoress=[]
    for j in range(0,NbClusterInThisCase):
        valueee=[]
        X_train =  np.array(X_train)
        X_test =  np.array(X_test)
        use_column = [True for ndx in range(NBPATTERNS)]
        for k in range(len(use_column)):
            if newDicoCluster[motifs[k]] in clustersDejaPris:
                use_column[k]=False
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
            #delete the features with the lowest importance score                                       
            feat_num = alpha[0][0]
            scoress.append(alpha[0][1])
            res.append(numbers[feat_num])
            #save results in a file
            f=open(NamePlotAblationStep1,"a")   
            f.write("Numero de cluster : "+ str(alpha[0][0])+"\n")
            #write score associated to the deleted feature
            f.write("Score : "+ str(alpha[0][1])+"\n")
            f.close()
            #delete from numbers the feature
            clustersDejaPris.append(feat_num)
            NbClusterActuel = NbClusterActuel-1
    return scoress,NbClusterInThisCase


def saveUniqueinCSV(uniqueSet,id_graphs_mono):
    dataframeResults = pd.DataFrame(index=range(0,len(id_graphs_mono)),columns=["OldID","newID"])
    compt=-1
    for uniquePattern in uniqueSet:
        compt=compt+1
        for j in range(len(id_graphs_mono)):
            if id_graphs_mono[j]==uniquePattern:
                dataframeResults["OldID"][j]=j
                dataframeResults["newID"][j]=compt
    dataframeResults.to_csv("DicoUniqueNCI1.csv",sep=";",index=False)

from sklearn.cluster import AgglomerativeClustering
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

def metricDotProduct(X):
    print("Avant")
    a = (len(X[0])-np.matmul(X,np.transpose(X)))/2
    print("Apres")
    return a


def keepOnlyPatterns(clusters):
    #Select 1 pattern per cluster
    #clusters = model.labels_
    nbClusters = max(clusters)+1
    patterns = []
    for i in range(nbClusters):
        for j in range(len(clusters)):
            if clusters[j]==i:
                patterns.append(j)
                break
    return patterns

def performClustering(pattern,distance):
    model = AgglomerativeClustering(distance_threshold=distance,metric="precomputed",n_clusters=None,linkage="complete")
    model = model.fit(pattern)
    return model

#import cdist
from scipy.spatial.distance import cdist
def selectCurrentClustering(pattern,distance,id_graphs,convertisseur):
    """ This function perform the full clustering for one specific value"""
    newID_graphs = []
    convertisseur = {}
    resUnique = []
    model = performClustering(pattern,distance)
    clusters = model.labels_
    n_clusters = max(clusters)+1
    #Create a dictionnary associating to each pattern the cluster it belongs to
    dicoClusterPattern = {}
    for i in range(len(clusters)):
        dicoClusterPattern[i]=clusters[i]
    res = []
    # Créer un dictionnaire qui associe à chaque cluster la liste des id des motifs qui lui appartiennent
    dicoCluster = {}
    # Calculer le centroïde de chaque cluster
    for cluster_id in range(n_clusters):
        cluster_points = []
        id_clusters_points = []
        for i in range(len(model.labels_)):
            if model.labels_[i]==cluster_id:
                cluster_points.append(superMatrice[i])
                id_clusters_points.append(i)
        cluster_centroid = np.mean(cluster_points, axis=0)
        # Calculer la distance de chaque point du cluster au centroïde
        distances = cdist(cluster_points, [cluster_centroid])
        
        # Trouver l'indice du point le plus proche du centroïde
        central_point_index = np.argmin(distances)
        
        # Ajouter l'id du point le plus central au tableau
        dicoCluster[cluster_id] = id_clusters_points[central_point_index]
        res.append(id_clusters_points[central_point_index])
        newID_graphs.append(id_graphs[id_clusters_points[central_point_index]])
    return model,res,convertisseur,newID_graphs

def ExtendDictionnary(dicoClusterPattern,dicoRepetition):
    newDictionnary = {}
    for key in dicoRepetition.keys():
        value=dicoRepetition[key]
        for patt in value:
            #si le pattern est dans le dictionnaire
            if patt in newDictionnary.keys():  
                print("??")
            else:
                newDictionnary[patt]=dicoClusterPattern[key] 
    #display the number of keys in the new dictionnary
    return newDictionnary
        
def proceedAblationForCluster(X_train,X_test,y_train,y_test,NBPATTERNS,NBCLUSTERS,dictionnaryCluster):
    print(dictionnaryCluster)
    """ This function proceed to an ablation study on the specified dataset, removing one cluster at a time
    X_train : training set
    X_test : test set
    y_train : training labels
    y_test : test labels
    motifs : list of patterns
    NBPATTERNS : number of patterns
    NBCLUSTERS : number of clusters
    dicoCluster : dictionnary associating to each pattern its cluster""" 
    #Save all results in a csv file
    # First : compute the score without removing any cluster
    base_score = compute_results(X_train,y_train,X_test, y_test)
    # Now : remove one cluster at a time
    scores = []
    res=[]
    scoress=[]
    for j in tqdm.tqdm(range(0,NBCLUSTERS)):
        valueee=[]
        X_train =  np.array(X_train)
        X_test =  np.array(X_test)
        use_column = [True for ndx in range(NBPATTERNS)]
        #Take all value of the dictionnary with the key j
        vals = [k for k, v in dictionnaryCluster.items() if v == j]
        for k in range(len(use_column)):
            if k in vals:
                use_column[k]=False
        scores = compute_results(X_train[:, use_column],y_train,X_test[:, use_column], y_test)
        valueee = scores
        scoress.append(scores)

        f=open("testAblationPTC6.txt","a")
        f.write("Numero de cluster : "+ str(j)+"\n")
        #write score associated to the deleted feature
        f.write("Score : "+ str(valueee)+"\n")
        f.close()
    #plot the results
    
    plt.plot(range(0,NBCLUSTERS),scoress)
    plt.xlabel('Number of clusters')
    plt.ylabel('FA-Score of the anomalous class')
    plt.legend()
    plt.title('Features importance')
    # save the plot in a specific directory
    plt.savefig("results/FOPPA20.pdf")
    return scoress

def proceedCompleteAblation(X_train,X_test,y_train,y_test,NBPATTERNS,NBCLUSTERS,dictionnaryCluster):
    #Same as proceedAblationForCluster but remove 1 cluster by one cluster
    # At each step remove the cluster with the lowest importance score
    #Save all results in a csv file
    # First : compute the score without removing any cluster
    #create a dataframe to store the results
    #index = number of clusters
    #columns = Step, deleted cluster, score
    dataframeResults = pd.DataFrame(index=range(0,NBCLUSTERS),columns=["Step","DeletedCluster","Score","NBClustersRemaining"])
    base_score = compute_results(X_train,y_train,X_test, y_test)
    # Repeat the process until there is no more cluster
    scores = []
    res=[]
    scoress=[]
    clustersDejaPris = []
    clustersAnePasPrendre = []
    X_train =  np.array(X_train)
    X_test =  np.array(X_test)
    for j in tqdm.tqdm(range(0,NBCLUSTERS)):
        #create a list of boolean where the features to keep is True
        use_column_base = [True for ndx in range(NBPATTERNS)]
        #delete the features associated with a cluster already deleted
        for k in range(len(use_column_base)):
            if dictionnaryCluster[k] in clustersDejaPris:
                use_column_base[k]=False
        # for each cluster : 2 possibilities
        # cluster already deleted : put a score of -1000 in order to not select it
        # cluster not deleted : compute the score associated to the deletion of this cluster
        scores=[]
        for i in range(0,NBCLUSTERS):
            if i in clustersDejaPris:
                scores.append(-1000)
            else:
                #remove pattern associated to the cluster i
                use_column = copy.deepcopy(use_column_base)
                for k in range(len(use_column)):
                    if dictionnaryCluster[k] == i:
                        use_column[k]=False
                gg = X_train[:, use_column]
                scores.append(compute_results(X_train[:, use_column],y_train,X_test[:, use_column], y_test))
        #select the cluster with the lowest importance score
        alpha = sorted(enumerate([s for s in scores]),
            key=lambda ndx_score: ndx_score[1],
            reverse=True)
        print(alpha)
        print(alpha[0][0])
        #Add the cluster with the lowest importance score to the list of clusters to delete
        clustersDejaPris.append(alpha[0][0])
        print(clustersDejaPris)
        #save results in the dataframe
        dataframeResults["Step"][j]=j
        dataframeResults["DeletedCluster"][j]=alpha[0][0]
        dataframeResults["Score"][j]=alpha[0][1]
        dataframeResults["NBClustersRemaining"][j]=len(gg[0])-1
        #save dataframe in a csv file
        dataframeResults.to_csv("results/ResultsMUTAG1.csv",sep=";",index=False)
        
        
    

def selectTopPatterns(diff,K,keep,convertisseur):
    for i in range(len(diff)):
        if i not in keep:
            diff[i]=-1
    keepPatterns = []
    if K==len(diff):
        return range(0,len(diff))
    else:
        for i in range(K):
            if sum(diff)==-1*len(diff):
                break
            bestScore = np.max(diff)
            bestPattern = np.argmax(diff)
            keepPatterns.append(bestPattern)
            diff[bestPattern]=-1
        return keepPatterns

def selectTopPatternsUnique(diff,K,id_graphs):
    keepPatterns = []
    for i in range(K):
        if sum(diff)==-1*len(diff):
            break
        bestScore = np.max(diff)
        bestPattern = np.argmax(diff)
        keepPatterns.append(bestPattern)
        diff[bestPattern]=-1
        for j in range(len(diff)):
            if id_graphs[bestPattern]==id_graphs[j]:
                diff[j]=-1
    return keepPatterns

def RankPatternForAScoreCluster(Scores,dico,TAILLECLUSTER):
    results = []
    dejaPris = []
    #tri par ordre décroissant, en gardant l'indice
    alpha = sorted(enumerate([s for s in Scores]),
        key=lambda ndx_score: ndx_score[1],
        reverse=True)
    #creer une liste avec les indices triés, en enlevant les valeurs égales a 0
    triIndice = []
    for i in range(len(alpha)):
        if alpha[i][1]!=0:
            triIndice.append(alpha[i][0])  
    NbClusterToTake = TAILLECLUSTER
    nbp = 0
    for i in range(len(triIndice)):
        nbp=nbp+1
        if dico[triIndice[i]]==-1:
            continue
        elif dico[triIndice[i]] not in dejaPris:
                results.append(dico[triIndice[i]])
                dejaPris.append(dico[triIndice[i]])
                NbClusterToTake=NbClusterToTake-1
        if NbClusterToTake==0:
            break
    return results,nbp,triIndice,alpha
    
def RankPatternForAScore(Scores,dico,TAILLERANGE):
    results = []
    dejaPris = []
    #tri par ordre décroissant, en gardant l'indice
    alpha = sorted(enumerate([s for s in Scores]),
        key=lambda ndx_score: ndx_score[1],
        reverse=True)
    #creer une liste avec les indices triés, en enlevant les valeurs égales a 0
    triIndice = []
    for i in range(len(alpha)):
            triIndice.append(alpha[i][0])  
    #On prend les K premiers patterns
    triIndice = triIndice[0:TAILLERANGE]
    #Pour chaque pattern, on regarde si son pattern Unique est dans le dico
    #Si oui, on regarde si le pattern Unique a déjà été pris
    #Si non, on le prend
    nbc=0
    for i in range(len(triIndice)):
        if dico[triIndice[i]]==-1:
            continue
        elif dico[triIndice[i]] not in dejaPris:
            results.append(dico[triIndice[i]])
            dejaPris.append(dico[triIndice[i]])
            nbc=nbc+1 
    return results,nbc,triIndice,alpha


def partialRepresentation(X,patterns):
    return X[:,np.array(patterns)]

if __name__ == '__main__':
    #Test different classifiers
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    listeClassifier = ["SVC"]
    TABLERESULTS = pd.DataFrame(index=range(0,100),columns=["Score","PATTERNS"])
    for classifier in listeClassifier:
        for cc in range(0,1):
            cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            arg="FOPPA"
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

            """loading processed patterns"""
            print("Reading processed patterns")
            xx,id_graphsMono,numberoccurencesMono = load_patterns(FILEMONOSET,TAILLEPATTERN)
            xx,id_graphsIso,numberoccurencesIso = load_patterns(FILEISOSET,TAILLEPATTERN)
            labelss = readLabels(FILELABEL)
            keep = graphKeep(PatternsRed,labelss)
            
            #Unique 
            ##Test Unique patterns
            dfr=-1

            for TYPEPATTERN in [id_graphsMono]:
                dfr=dfr+1
                id_graphsMono = copy.deepcopy(TYPEPATTERN)
                #create a dictionnary, for each pattern , indicate the unique pattern it belongs to
                dicoRepetition = {}
                dicoUniqueToPattern = {}
                convertisseur = {}
                patternsUnique=[]
                dejaVu = []
                c=-1
                for i in tqdm.tqdm(range(len(id_graphsMono))):
                    if id_graphsMono[i]==[]:
                        dicoRepetition[i]=-1
                        dicoUniqueToPattern[c] = []
                    else:
                        if id_graphsMono[i] not in dejaVu:
                            patternsUnique.append(i)
                            c = c+1
                            dejaVu.append(id_graphsMono[i]) 
                            dicoRepetition[i]=c
                            dicoUniqueToPattern[c] = []
                            convertisseur[c]=i
                            dicoUniqueToPattern[c].append(i)
                        else:
                            dicoRepetition[i]=dejaVu.index(id_graphsMono[i])
                            dicoUniqueToPattern[dejaVu.index(id_graphsMono[i])].append(i)
            
                dicoRepetition = {}
                patternsUnique=[]
                dejaVu = []
                for i in tqdm.tqdm(range(len(id_graphsMono))):
                    if id_graphsMono[i] not in dejaVu:
                        patternsUnique.append(i)
                        dejaVu.append(id_graphsMono[i]) 
                        dicoRepetition[i]=i
                    else:
                        dicoRepetition[i]=dejaVu.index(id_graphsMono[i])
                
                superMatrice = np.ones((len(dejaVu),TAILLEGRAPHE),dtype=np.int8)*-1
                for i in range(len(dejaVu)):
                    for j in range(len(dejaVu[i])):
                            superMatrice[i][dejaVu[i][j]]=1
                ### transform the matrix into a list of list


                ### Resume : superMatrice is a matrix where each row is a unique pattern
                # Each column is a graph
                # If the graph contains the pattern, the value is 1, else -1
                #Dejavu is the list of ids associated to each unique pattern
                #patternsUnique is the list of unique patterns
                superMatrice = superMatrice.tolist()
                dotProductMat = metricDotProduct(superMatrice)
                for i in range(len(superMatrice)):
                    superMatrice[i]=np.array(superMatrice[i])
                vectRepresentation,vectLabels = ComputeRepresentation(range(0,len(patternsUnique)),dejaVu,labelss,TAILLEGRAPHE)
                vectRepresentation = np.array(vectRepresentation)
                X_train, X_test, y_train, y_test = train_test_split(vectRepresentation, vectLabels, test_size=0.2, random_state=7)
                #On calcule les scores discriminants pour tous les motifs uniques
                growth,suppdiff,TPR,FPR,STR = computeScoreMono(patternsUnique,labelss,dejaVu,len(patternsUnique))
                #Ici on a les X_train et X_test qui sont les vecteurs de représentation des graphes bases sur les motifs uniques
                for nbCluster in [0,5,10,15]: #Pour differentes valeurs de clusters
                    model,patternsStepI,convertisseur,newIdGraphs = selectCurrentClustering(dotProductMat,nbCluster,dejaVu,convertisseur)  
                    growthT=copy.deepcopy(growth)
                    suppdiffT=copy.deepcopy(suppdiff)
                    TPRT=copy.deepcopy(TPR)
                    FPRT=copy.deepcopy(FPR)
                    STRT=copy.deepcopy(STR)
                    #On recupere les motifs UNIQUES qui ont été selectionnés
                    # Ils sont stockés dans patternsStepI
                    #On trie d'abord PatternsStepI par ordre croissant pour ne pas avoir de problemes
                    patternsStepI = sorted(patternsStepI)
                    for j in range(len(patternsStepI)):
                        convertisseur[j]=patternsStepI[j]
                    print(patternsStepI)
                    vectRepresentationB = partialRepresentation(copy.deepcopy(vectRepresentation),patternsStepI)
                    X_trainB = partialRepresentation(copy.deepcopy(X_train),patternsStepI)
                    X_testB = partialRepresentation(copy.deepcopy(X_test),patternsStepI)

                    #Reprentation de X_train et X_test avec les motifs uniques gardes dans l'etape de clustering
                    d_train = xgboost.DMatrix(X_trainB, label=y_train)
                    d_test = xgboost.DMatrix(X_testB, label=y_test)
                    params = {
                    "eta": 0.01,
                    "objective": "binary:logistic",
                    "subsample": 0.5,
                    "base_score": np.mean(y_train),
                    "eval_metric": "logloss",
                    }
                    model = xgboost.train(
                        params,
                        d_train,
                        5000,
                        evals=[(d_test, "test")],
                        verbose_eval=100,
                        early_stopping_rounds=20,
                    )
                    
                    #POUR SHAP
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_trainB)
                    #Display the importance of each feature
                    #Save the shap values in a file
                    shap_values = shap_values.values
                    shap_values = np.abs(shap_values)
                    #compute the mean importance score of each feature
                    meanShap = np.mean(shap_values,axis=0)

                    #1) Sort the effective results and keep the indexes
                    alpha = sorted(enumerate([s for s in meanShap]),
                            key=lambda ndx_score: ndx_score[1],
                            reverse=True)
                        
                    explainer = shap.TreeExplainer(
        model,data=np.array(X_train),feature_perturbation="interventional",model_output="logloss"
    )
                    shap_values = explainer.shap_values(np.array(X_trainB),y_train)
                    shap_values = np.abs(shap_values)
                    meanShap = np.mean(shap_values,axis=0)
                    beta = sorted(enumerate([s for s in meanShap]),
                        key=lambda ndx_score: ndx_score[1],
                        reverse=True)
                    #Display the importance of each feature

                    RandomF = RandomForestClassifier(n_estimators=100,random_state=0)
                    RandomF.fit(X_trainB,y_train)
                    #return the importance score of each feature
                    importances = RandomF.feature_importances_
                    #sort the importance score
                    gamma = sorted(enumerate([s for s in importances]),
                        key=lambda ndx_score: ndx_score[1],
                        reverse=True)
                    rerere = []
                    rererere = []
                    keepP = []
                    keepPP = []
                    keeePPP = []
                    #import spearmanr
                    from scipy.stats import spearmanr
                    #compare the ranking of the features for each metric
                    Matrice = np.zeros((8,8))
                    #keep only the indexes of the features of alpha
                    for i in range(len(alpha)):
                        alpha[i]=alpha[i][0]
                        beta[i]=beta[i][0]
                        gamma[i]=gamma[i][0]
                    alpha = np.array(alpha)
                    beta = np.array(beta)
                    gamma = np.array(gamma)
                    import rbo
                    growthB = selectTopPatterns(growthT,len(patternsStepI)-1,patternsStepI,convertisseur)
                    suppdiffB = selectTopPatterns(suppdiffT,len(patternsStepI)-1,patternsStepI,convertisseur)
                    TPRB = selectTopPatterns(TPRT,len(patternsStepI)-1,patternsStepI,convertisseur)
                    FPRB = selectTopPatterns(FPRT,len(patternsStepI)-1,patternsStepI,convertisseur)
                    STRB = selectTopPatterns(STRT,len(patternsStepI)-1,patternsStepI,convertisseur)
                    Metric = [alpha,beta,gamma,growthB,suppdiffB,TPRB,FPRB,STRB]
                    for i in range(len(Metric)):
                        for j in range(len(Metric)):
                            #compute the number of common features between the two metrics in the top 100 patterns
                            Matrice[i][j] = len(set(Metric[i][0:100]).intersection(set(Metric[j][0:100])))
                
                    #plot the results using a heatmap
                    plt.figure()
                    import seaborn as sns
                    NameMetric = ["SHAP","Sage","RF","GR","SD","TPR","FPR","STR"]
                    #add names to the axes
                    sns.heatmap(Matrice, annot=True, fmt=".2f", cmap="coolwarm",xticklabels=NameMetric,yticklabels=NameMetric)
                    plt.xlabel('Metrics')
                    plt.ylabel('Metrics')
                    # save the plot in a specific directory
                    plt.savefig("results/"+arg+str(dfr)+str(nbCluster)+"heatmapInter.pdf")

                    #2) Compute the F1 score for each number of features
                    #for i in tqdm.tqdm(range(1,len(patternsUnique),int(len(patternsUnique)/100))):
                    for i in tqdm.tqdm(range(1,50)):
                        #select the top i features
                        for dd in range(i):
                            keepP.append(convertisseur[alpha[dd]])
                        #keep only the features selected in keepP
                        #don't use loops, it's too slow
                        X_train2 = X_train[:,keepP]
                        X_test2 = X_test[:,keepP]
                        #compute the F1 score
                        score = compute_results(X_train2,y_train,X_test2, y_test)
                        rerere.append(score)

                        #select the top i features
                        for dd in range(i):
                            keepPP.append(convertisseur[beta[dd]])
                        #Transform X_TRAIN and X_TEST keeping only the features selected in keepP
                        X_train2 =X_train[:,keepPP]
                        X_test2 = X_test[:,keepPP]
                        #compute the F1 score
                        score = compute_results(X_train2,y_train,X_test2, y_test)
                        rerere.append(score)

                        for dd in range(i):
                            keeePPP.append(convertisseur[gamma[dd]])
                        #Transform X_TRAIN and X_TEST keeping only the features selected in keepP
                        X_train2 = X_train[:,keeePPP]
                        X_test2 = X_test[:,keeePPP]
                        score = compute_results(X_train2,y_train,X_test2, y_test)
                        rerere.append(score)

                        #Same thing for other metrics : growth, support, TPR, FPR, ST
                        for metric in [growthB,suppdiffB,TPRB,FPRB,STRB]:
                            patternsUnique2 = selectTopPatterns(copy.deepcopy(metric),i,patternsStepI,convertisseur)
                            patternsUnique2 = np.array(patternsUnique2)
                            X_train2 = X_train[:,patternsUnique2]
                            X_test2 = X_test[:,patternsUnique2]
                            score = compute_results(X_train2,y_train,X_test2, y_test)
                            rerere.append(score)
                    #plot the results
                    plt.figure()
                    #values mod 6 : 0 for SHAP, 1 for growth, 2 for suppdiff, 3 for TPR, 4 for FPR, 5 for STR
                    #plot each curve separately
                    shaps = []
                    sage = []
                    growths = []
                    suppdiffs = []
                    TPRs = []
                    FPRs = []
                    STRs = []
                    randomF = []
                    for j in range(len(rerere)):
                        if j%8==0:
                            shaps.append(rerere[j])
                        elif j%8==1:
                            sage.append(rerere[j])
                        elif j%8==2:
                            randomF.append(rerere[j])
                        elif j%8==3:
                            growths.append(rerere[j])
                        elif j%8==4:
                            suppdiffs.append(rerere[j])
                        elif j%8==5:
                            TPRs.append(rerere[j])
                        elif j%8==6:
                            FPRs.append(rerere[j])
                        elif j%8==7:
                            STRs.append(rerere[j])
                    plt.plot(range(1,len(shaps)+1),shaps,label="SHAP")
                    plt.plot(range(1,len(sage)+1),sage,label="Sage")
                    plt.plot(range(1,len(growths)+1),growths,label="Growth")
                    plt.plot(range(1,len(suppdiffs)+1),suppdiffs,label="SuppDiff")
                    plt.plot(range(1,len(TPRs)+1),TPRs,label="TPR")
                    plt.plot(range(1,len(FPRs)+1),FPRs,label="FPR")
                    plt.plot(range(1,len(STRs)+1),STRs,label="STR")
                    plt.plot(range(1,len(randomF)+1),randomF,label="Random Forest")
                    plt.xlabel('Number of features')
                    plt.ylabel('FA-Score of the anomalous class')
                    plt.legend()
                    # save the plot in a specific directory
                    plt.savefig("results/"+arg+str(dfr)+str(nbCluster)+"F1ScoreNEW.pdf")
                    